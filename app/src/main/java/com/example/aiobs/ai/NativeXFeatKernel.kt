package com.example.aiobs.ai

import android.util.Log

/**
 * ARM64 native hot path for XFeat raw TFLite outputs and 64D descriptor matching.
 * The model itself remains TFLite/GPU; only the expensive JVM post-processing is
 * moved to native code.
 */
object NativeXFeatKernel {
    private const val TAG = "NativeXFeatKernel"
    const val MAX_TOP_K = 1024
    const val DESCRIPTOR_DIM = 64

    @Volatile
    private var loaded = false

    init {
        try {
            System.loadLibrary("aiobs_xfeat_neon")
            loaded = true
            Log.i(TAG, "ARM64 XFeat native hot path loaded")
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to load aiobs_xfeat_neon", t)
        }
    }

    data class PostResult(
        val features: XFeatFeatureSet?,
        val candidateCount: Int,
        val selectedCount: Int,
        val meanScore: Float,
        val descriptorNormMean: Float,
        val elapsedMs: Long
    )

    fun isAvailable(): Boolean = loaded

    fun postProcess(
        feats: FloatArray,
        logits: FloatArray,
        reliability: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        topK: Int,
        detectionThreshold: Float
    ): PostResult? {
        if (!loaded) return null
        val k = topK.coerceIn(1, MAX_TOP_K)
        if (imageWidth <= 0 || imageHeight <= 0 || imageWidth % 8 != 0 || imageHeight % 8 != 0) {
            return null
        }
        val cellH = imageHeight / 8
        val cellW = imageWidth / 8
        if (feats.size != DESCRIPTOR_DIM * cellH * cellW ||
            logits.size != 65 * cellH * cellW ||
            reliability.size != cellH * cellW) {
            return null
        }
        return try {
            val output = FloatArray(4 + k * (2 + DESCRIPTOR_DIM + 1))
            val ok = nativePostProcess(
                feats, logits, reliability, output,
                imageWidth, imageHeight, k, detectionThreshold
            )
            if (!ok) return null

            var offset = 0
            val count = output[offset++].toInt().coerceIn(0, k)
            val candidateCount = output[offset++].toInt().coerceAtLeast(0)
            val meanScore = output[offset++]
            val descriptorNormMean = output[offset++]
            if (count <= 0) {
                return PostResult(null, candidateCount, 0, meanScore, descriptorNormMean, 0L)
            }

            val keypoints = FloatArray(count * 2)
            val descriptors = FloatArray(count * DESCRIPTOR_DIM)
            val scores = FloatArray(count)
            for (i in 0 until count) {
                keypoints[i * 2] = output[offset++]
                keypoints[i * 2 + 1] = output[offset++]
                val d = i * DESCRIPTOR_DIM
                for (j in 0 until DESCRIPTOR_DIM) descriptors[d + j] = output[offset++]
                scores[i] = output[offset++]
            }
            PostResult(
                XFeatFeatureSet(keypoints, descriptors, count, DESCRIPTOR_DIM, scores),
                candidateCount, count, meanScore, descriptorNormMean, 0L
            )
        } catch (t: Throwable) {
            Log.e(TAG, "nativePostProcess failed", t)
            null
        }
    }

    fun mutualNearestMatch(
        reference: XFeatFeatureSet,
        candidate: XFeatFeatureSet
    ): List<Pair<Int, Int>>? {
        if (!loaded || reference.descriptorCols != DESCRIPTOR_DIM || candidate.descriptorCols != DESCRIPTOR_DIM) return null
        return try {
            val raw = nativeMutualNearestMatch(
                reference.descriptors,
                candidate.descriptors,
                reference.count,
                candidate.count,
                DESCRIPTOR_DIM
            )
            ArrayList<Pair<Int, Int>>(raw.size / 2).apply {
                var i = 0
                while (i + 1 < raw.size) {
                    add(raw[i] to raw[i + 1])
                    i += 2
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "nativeMutualNearestMatch failed", t)
            null
        }
    }

    @JvmStatic
    private external fun nativePostProcess(
        feats: FloatArray,
        logits: FloatArray,
        reliability: FloatArray,
        output: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        topK: Int,
        detectionThreshold: Float
    ): Boolean

    @JvmStatic
    private external fun nativeMutualNearestMatch(
        referenceDescriptors: FloatArray,
        candidateDescriptors: FloatArray,
        referenceCount: Int,
        candidateCount: Int,
        descriptorDim: Int
    ): IntArray

    private fun elapsedMs(startNs: Long): Long =
        ((System.nanoTime() - startNs).coerceAtLeast(0L)) / 1_000_000L
}
