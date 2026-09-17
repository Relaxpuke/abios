package com.example.aiobs.ai

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Android-side post-processing for XFeat W8A16.
 *
 * QNN Context Binary outputs are decoded by XFeatQnnContextBinaryRunner into
 * batch-1 logical NCHW float arrays with a runtime grid of modelSize / 8.
 *
 * The sparse path mirrors the validated XFeat detectAndCompute-equivalent flow:
 *   65-way softmax -> pixel-shuffle heatmap -> 5x5 NMS -> reliability scoring
 *   -> Top-K -> bicubic descriptor interpolation -> L2 normalization.
 */
object XFeatPostProcessor {
    private const val FEAT_CHANNELS = 64
    private const val LOGIT_CHANNELS = 65
    private const val NMS_RADIUS = 2
    private const val ALPHA = -0.75f

    data class Result(
        val features: XFeatFeatureSet?,
        val candidateCount: Int,
        val selectedCount: Int,
        val meanScore: Float,
        val descriptorNormMean: Float,
        val elapsedMs: Long
    )

    private data class Candidate(
        val x: Int,
        val y: Int,
        val score: Float
    )

    fun process(
        feats: FloatArray,
        keypointLogits: FloatArray,
        reliability: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        modelSize: Int,
        topK: Int = 1024,
        detectionThreshold: Float = 0.05f
    ): Result {
        val start = System.nanoTime()
        if (
            modelSize <= 0 || modelSize % 8 != 0 ||
            imageWidth <= 0 || imageHeight <= 0
        ) {
            return Result(null, 0, 0, 0f, 0f, elapsedMs(start))
        }

        val cellH = modelSize / 8
        val cellW = modelSize / 8

        if (
            feats.size != FEAT_CHANNELS * cellH * cellW ||
            keypointLogits.size != LOGIT_CHANNELS * cellH * cellW ||
            reliability.size != cellH * cellW
        ) {
            return Result(null, 0, 0, 0f, 0f, elapsedMs(start))
        }

        val fullH = modelSize
        val fullW = modelSize
        val heatmap = FloatArray(fullH * fullW)
        fillKeypointHeatmap(keypointLogits, heatmap, cellH, cellW, fullH, fullW)

        val candidates = ArrayList<Candidate>(modelSize * modelSize / 2)
        for (y in 0 until fullH) {
            for (x in 0 until fullW) {
                val idx = y * fullW + x
                val value = heatmap[idx]
                if (value < detectionThreshold) continue
                if (!isStrictLocalMaximum(heatmap, x, y, NMS_RADIUS, fullW, fullH)) continue

                val keypointScore = nearestSampleHeatmap(
                    heatmap, x.toFloat(), y.toFloat(), fullW, fullH
                )
                val reliabilityScore = bilinearSampleReliability(
                    reliability, x.toFloat(), y.toFloat(), cellW, cellH, fullW, fullH
                )
                val score = keypointScore * reliabilityScore
                if (x == 0 && y == 0) continue
                if (score > 0f && score.isFinite()) {
                    candidates += Candidate(x, y, score)
                }
            }
        }

        candidates.sortByDescending { it.score }
        val selected = candidates.take(topK.coerceIn(1, 4096))
        if (selected.isEmpty()) {
            return Result(null, candidates.size, 0, 0f, 0f, elapsedMs(start))
        }

        val count = selected.size
        val keypoints = FloatArray(count * 2)
        val descriptors = FloatArray(count * FEAT_CHANNELS)
        val scores = FloatArray(count)
        var scoreSum = 0.0
        var descriptorNormSum = 0.0

        val scaleX = imageWidth.toFloat() / fullW.toFloat()
        val scaleY = imageHeight.toFloat() / fullH.toFloat()

        for (i in 0 until count) {
            val c = selected[i]
            val x = c.x.toFloat()
            val y = c.y.toFloat()

            keypoints[i * 2] = x * scaleX
            keypoints[i * 2 + 1] = y * scaleY
            scores[i] = c.score
            scoreSum += c.score.toDouble()

            sampleBicubicDescriptor(
                feats = feats,
                x = x,
                y = y,
                cellW = cellW,
                cellH = cellH,
                fullW = fullW,
                fullH = fullH,
                out = descriptors,
                outOffset = i * FEAT_CHANNELS
            )

            var norm2 = 0.0
            val offset = i * FEAT_CHANNELS
            for (k in 0 until FEAT_CHANNELS) {
                val v = descriptors[offset + k].toDouble()
                norm2 += v * v
            }
            val norm = sqrt(norm2).toFloat().coerceAtLeast(1e-8f)
            for (k in 0 until FEAT_CHANNELS) {
                descriptors[offset + k] /= norm
            }
            descriptorNormSum += 1.0
        }

        val featureSet = XFeatFeatureSet(
            keypointsXY = keypoints,
            descriptors = descriptors,
            descriptorRows = count,
            descriptorCols = FEAT_CHANNELS,
            scores = scores
        )

        return Result(
            features = featureSet,
            candidateCount = candidates.size,
            selectedCount = count,
            meanScore = (scoreSum / count.coerceAtLeast(1)).toFloat(),
            descriptorNormMean = (descriptorNormSum / count.coerceAtLeast(1)).toFloat(),
            elapsedMs = elapsedMs(start)
        )
    }

    private fun fillKeypointHeatmap(
        logits: FloatArray,
        heatmap: FloatArray,
        cellH: Int,
        cellW: Int,
        fullH: Int,
        fullW: Int
    ) {
        val cellCount = cellH * cellW
        var cell = 0
        while (cell < cellCount) {
            var maxLogit = -Float.MAX_VALUE
            var c = 0
            while (c < LOGIT_CHANNELS) {
                val value = logits[c * cellCount + cell]
                if (value > maxLogit) maxLogit = value
                c++
            }

            var denom = 0.0
            c = 0
            while (c < LOGIT_CHANNELS) {
                denom += exp((logits[c * cellCount + cell] - maxLogit).toDouble())
                c++
            }
            val inv = if (denom <= 1e-12) 0f else (1.0 / denom).toFloat()

            val cellY = cell / cellW
            val cellX = cell % cellW
            for (dy in 0..7) {
                val outY = cellY * 8 + dy
                for (dx in 0..7) {
                    val cls = dy * 8 + dx
                    val probability =
                        exp((logits[cls * cellCount + cell] - maxLogit).toDouble()).toFloat() * inv
                    heatmap[outY * fullW + cellX * 8 + dx] = probability
                }
            }
            cell++
        }
    }

    private fun isStrictLocalMaximum(
        heatmap: FloatArray,
        x: Int,
        y: Int,
        radius: Int,
        fullW: Int,
        fullH: Int
    ): Boolean {
        val center = heatmap[y * fullW + x]
        for (dy in -radius..radius) {
            val yy = (y + dy).coerceIn(0, fullH - 1)
            for (dx in -radius..radius) {
                if (dx == 0 && dy == 0) continue
                val xx = (x + dx).coerceIn(0, fullW - 1)
                if (heatmap[yy * fullW + xx] >= center) return false
            }
        }
        return true
    }

    private fun nearestSampleHeatmap(
        heatmap: FloatArray,
        x: Float,
        y: Float,
        fullW: Int,
        fullH: Int
    ): Float {
        val sx = x * fullW.toFloat() / (fullW - 1).toFloat() - 0.5f
        val sy = y * fullH.toFloat() / (fullH - 1).toFloat() - 0.5f
        val nx = floor(sx + 0.5f).toInt()
        val ny = floor(sy + 0.5f).toInt()
        if (nx !in 0 until fullW || ny !in 0 until fullH) return 0f
        return heatmap[ny * fullW + nx]
    }

    private fun bilinearSampleReliability(
        reliability: FloatArray,
        x: Float,
        y: Float,
        cellW: Int,
        cellH: Int,
        fullW: Int,
        fullH: Int
    ): Float {
        val sx = x * cellW.toFloat() / (fullW - 1).toFloat() - 0.5f
        val sy = y * cellH.toFloat() / (fullH - 1).toFloat() - 0.5f
        return bilinear2d(reliability, cellW, cellH, sx, sy)
    }

    private fun bilinear2d(
        data: FloatArray,
        width: Int,
        height: Int,
        x: Float,
        y: Float
    ): Float {
        val x0 = floor(x.toDouble()).toInt()
        val y0 = floor(y.toDouble()).toInt()
        val x1 = x0 + 1
        val y1 = y0 + 1
        val wx = x - x0
        val wy = y - y0
        val v00 = sampleZero(data, width, height, x0, y0)
        val v10 = sampleZero(data, width, height, x1, y0)
        val v01 = sampleZero(data, width, height, x0, y1)
        val v11 = sampleZero(data, width, height, x1, y1)
        val top = v00 * (1f - wx) + v10 * wx
        val bottom = v01 * (1f - wx) + v11 * wx
        return (top * (1f - wy) + bottom * wy).coerceIn(0f, 1f)
    }

    private fun sampleBicubicDescriptor(
        feats: FloatArray,
        x: Float,
        y: Float,
        cellW: Int,
        cellH: Int,
        fullW: Int,
        fullH: Int,
        out: FloatArray,
        outOffset: Int
    ) {
        val sx = x * cellW.toFloat() / (fullW - 1).toFloat() - 0.5f
        val sy = y * cellH.toFloat() / (fullH - 1).toFloat() - 0.5f
        val baseX = floor(sx.toDouble()).toInt()
        val baseY = floor(sy.toDouble()).toInt()

        for (c in 0 until FEAT_CHANNELS) {
            var value = 0.0
            for (j in -1..2) {
                val yy = baseY + j
                val wy = cubicWeight(sy - yy)
                if (wy == 0f) continue
                for (i in -1..2) {
                    val xx = baseX + i
                    val wx = cubicWeight(sx - xx)
                    if (wx == 0f) continue
                    val sample = if (xx in 0 until cellW && yy in 0 until cellH) {
                        feats[c * cellH * cellW + yy * cellW + xx]
                    } else {
                        0f
                    }
                    value += sample.toDouble() * wx.toDouble() * wy.toDouble()
                }
            }
            out[outOffset + c] = value.toFloat()
        }
    }

    private fun cubicWeight(distance: Float): Float {
        val x = abs(distance)
        return when {
            x <= 1f -> (ALPHA + 2f) * x * x * x - (ALPHA + 3f) * x * x + 1f
            x < 2f -> ALPHA * x * x * x - 5f * ALPHA * x * x + 8f * ALPHA * x - 4f * ALPHA
            else -> 0f
        }
    }

    private fun sampleZero(data: FloatArray, width: Int, height: Int, x: Int, y: Int): Float {
        if (x !in 0 until width || y !in 0 until height) return 0f
        return data[y * width + x]
    }

    private fun elapsedMs(startNs: Long): Long = (System.nanoTime() - startNs) / 1_000_000L
}
