package com.example.aiobs.ai

import android.content.Context
import android.util.Log
import org.opencv.android.OpenCVLoader
import kotlin.math.sqrt
import kotlin.math.pow

/**
 * Formal XFeat local identity backend.
 *
 * Production path:
 *   crop -> grayscale -> aspect-preserving INTER_AREA -> centered model-size letterbox
 *        -> FP32 / 255 -> Direct QNN Context Binary (W8A16, SM8550 HTP)
 *        -> FP32 tensors -> XFeatPostProcessor -> sparse feature set
 *
 * There is intentionally no TFLite/GPU path here. W8A16 QNN Context Binary is
 * the canonical implementation used by both
 * the LOCKED anchor capture path and the LOST/RECOVERY verifier.
 */
class XFeatLocalIdentityMatcher(
    context: Context,
    private val assetName: String = W8A16_ASSET,
    identityThreshold: Float = DEFAULT_IDENTITY_THRESHOLD
) { // 🌟 修复：补充了缺失的闭合括号和大括号

    data class MatchResult(
        val passed: Boolean,
        val score: Float,
        val goodMatches: Int,
        val inliers: Int,
        val inlierRatio: Float,
        val bankIndex: Int,
        val quality: Float,
        val spatialCoverage: Float,
        val referenceCount: Int,
        val candidateCount: Int,
        val elapsedMs: Long
    )

    private val appContext = context.applicationContext
    private val identityThreshold = identityThreshold.coerceIn(0.05f, 0.95f)

    private val runner: XFeatQnnContextBinaryRunner?
    private var backend = "UNINITIALIZED"

    @Volatile var lastInputMs: Long = 0L
        private set
    @Volatile var lastInferMs: Long = 0L
        private set
    @Volatile var lastPostprocessMs: Long = 0L
        private set

    val available: Boolean
        get() = runner?.isReady() == true

    val backendName: String
        get() = if (available) "QNN_W8A16_HTP" else backend

    init {
        try {
            OpenCVLoader.initLocal()
        } catch (t: Throwable) {
            Log.w(TAG, "OpenCV init check failed; RANSAC may be unavailable", t)
        }

        runner = try {
            XFeatQnnContextBinaryRunner(
                context = appContext,
                assetName = assetName
            ).also {
                backend = "QNN_W8A16_HTP"
                Log.i(TAG, "XFeat production backend ready=" + it.isReady() + " asset=$assetName")
            }
        } catch (t: Throwable) {
            backend = "UNAVAILABLE"
            Log.e(TAG, "XFeat W8A16 QNN initialization failed", t)
            null
        }
    }

    fun extract(patch: LocalGrayPatch?): XFeatFeatureSet? {
        if (patch == null || patch.pixels.isEmpty()) return null

        val qnn = runner ?: return null
        if (!qnn.isReady()) return null

        return try {
            val inputStart = System.nanoTime()
            val input = toInput(patch)
            val inputMs = (System.nanoTime() - inputStart) / 1_000_000L
            lastInputMs = inputMs

            val result = qnn.infer(input)
            lastInferMs = result.elapsedMs.toLong()

            val nativePostStart = System.nanoTime()
            val nativePost = NativeXFeatKernel.postProcess(
                feats = result.feats,
                logits = result.keypointLogits,
                reliability = result.reliability,
                imageWidth = modelSize,
                imageHeight = modelSize,
                topK = TOP_K,
                detectionThreshold = DETECTION_THRESHOLD
            )
            val post: NativeXFeatKernel.PostResult = nativePost
                ?: run {
                    val fallback = XFeatPostProcessor.process(
                        feats = result.feats,
                        keypointLogits = result.keypointLogits,
                        reliability = result.reliability,
                        imageWidth = modelSize,
                        imageHeight = modelSize,
                        modelSize = modelSize,
                        topK = TOP_K,
                        detectionThreshold = DETECTION_THRESHOLD
                    )
                    NativeXFeatKernel.PostResult(
                        features = fallback.features,
                        candidateCount = fallback.candidateCount,
                        selectedCount = fallback.selectedCount,
                        meanScore = fallback.meanScore,
                        descriptorNormMean = fallback.descriptorNormMean,
                        elapsedMs = fallback.elapsedMs
                    )
                }
            val postMs = if (nativePost != null) {
                (System.nanoTime() - nativePostStart) / 1_000_000L
            } else {
                post.elapsedMs
            }

            lastPostprocessMs = postMs

            post.features
        } catch (t: Throwable) {
            Log.w(TAG, "XFeat W8A16 extraction failed", t)
            null
        }
    }

    fun match(
        reference: XFeatFeatureSet?,
        candidate: XFeatFeatureSet?,
        bankIndex: Int = 0
    ): MatchResult? {
        if (!available || reference == null || candidate == null) {
            return null
        }

        if (reference.count < MIN_GEOMETRY_POINTS || candidate.count < MIN_GEOMETRY_POINTS) {
            return null
        }

        val start = System.nanoTime()

        return try {
            // 1. 调用底层 C++ 极速全景海选匹配 (两万点级纯点积并发)
            val rawMatches = NativeXFeatKernel.mutualNearestMatch(
                reference,
                candidate
            ) ?: mutualNearestMatch(
                reference,
                candidate
            )

            if (rawMatches.isEmpty()) {
                return MatchResult(
                    passed = false, score = 0f, goodMatches = 0, inliers = 0,
                    inlierRatio = 0f, bankIndex = bankIndex, quality = 0f,
                    spatialCoverage = 0f, referenceCount = reference.count, candidateCount = candidate.count,
                    elapsedMs = (System.nanoTime() - start) / 1_000_000L
                )
            }

            // =========================================================================
            // 🌟 XFeat 伪 3D 拓扑引擎：Coarse-to-Fine 架构 🌟
            // =========================================================================

            val chunkSizes = reference.chunkSizes ?: intArrayOf(reference.count)
            val numChunks = chunkSizes.size

            // 建立手账字典 (每个姿态的起始 index)
            val chunkOffsets = IntArray(numChunks)
            var currentOffset = 0
            for (i in 0 until numChunks) {
                chunkOffsets[i] = currentOffset
                currentOffset += chunkSizes[i]
            }

            var finalScore = 0f
            var finalRatio = 0f
            var finalInliers = 0
            var winnerName = "Anchor"
            val breakdownLog = java.lang.StringBuilder()

            // 统一计算空间覆盖率 (依然基于所有候选点的全局覆盖)
            val candidateCoverage = spatialCoverage(
                FloatArray(rawMatches.size * 2) { idx ->
                    val p = rawMatches[idx / 2]
                    if (idx % 2 == 0) candidate.keypointsXY[p.second * 2] else candidate.keypointsXY[p.second * 2 + 1]
                },
                modelSize.toFloat(), modelSize.toFloat()
            )

            if (numChunks > 1) {
                // 👉 第一阶段：粗筛投票 (Coarse Voting) - 极速分桶
                val chunkVotes = IntArray(numChunks)
                // 临时记录属于每个 Chunk 的原始匹配点坐标 (用扁平 FloatArray 直接存放给 RANSAC 用的坐标，省去后续的再次转换)
                val chunkPairCoords = Array(numChunks) { FloatArray(rawMatches.size * 4) }
                val chunkPairIdx = Array(numChunks) { IntArray(rawMatches.size * 2) } // 用来算 meanCos
                val chunkPairCounts = IntArray(numChunks)

                for (i in rawMatches.indices) {
                    val pair = rawMatches[i]
                    val refIdx = pair.first
                    val candIdx = pair.second

                    // 查字典：找出这个点属于哪个记忆姿态
                    var c = 0
                    for (k in numChunks - 1 downTo 0) {
                        if (refIdx >= chunkOffsets[k]) {
                            c = k
                            break
                        }
                    }

                    val count = chunkPairCounts[c]
                    // 提前准备好 RANSAC 需要的坐标格式：srcX, srcY, dstX, dstY
                    chunkPairCoords[c][count * 4] = reference.keypointsXY[refIdx * 2]
                    chunkPairCoords[c][count * 4 + 1] = reference.keypointsXY[refIdx * 2 + 1]
                    chunkPairCoords[c][count * 4 + 2] = candidate.keypointsXY[candIdx * 2]
                    chunkPairCoords[c][count * 4 + 3] = candidate.keypointsXY[candIdx * 2 + 1]

                    // 记录原始索引，留给 meanCos 计算
                    chunkPairIdx[c][count * 2] = refIdx
                    chunkPairIdx[c][count * 2 + 1] = candIdx

                    chunkPairCounts[c]++
                    chunkVotes[c]++ // 该姿态票数 +1
                }

                // =========================================================
                // 👉 第二阶段：极限 O(N) 零分配选出 Top-2 (Zero-GC Top-2 Polling)
                // =========================================================
                var top1Idx = -1
                var top1Votes = -1
                var top2Idx = -1
                var top2Votes = -1

                // 单次遍历打擂台，坚决不排序，不创建任何 List 对象！
                for (c in 0 until numChunks) {
                    val votes = chunkVotes[c]
                    if (votes >= 8) { // 至少有8对匹配才值得进行几何抢救
                        if (votes > top1Votes) {
                            top2Votes = top1Votes
                            top2Idx = top1Idx
                            top1Votes = votes
                            top1Idx = c
                        } else if (votes > top2Votes) {
                            top2Votes = votes
                            top2Idx = c
                        }
                    }
                }

                // 组装最终需要执行 RANSAC 的名额 (最多2个)
                val topCandidates = IntArray(2)
                var topCount = 0
                if (top1Idx != -1) topCandidates[topCount++] = top1Idx
                if (top2Idx != -1) topCandidates[topCount++] = top2Idx

                // 👉 第三阶段：精准 1v1 RANSAC 审判 (Fine RANSAC)
                for (i in 0 until topCount) {
                    val c = topCandidates[i] // 提取出真实的姿态索引
                    val pCount = chunkPairCounts[c]
                    val name = if (c == 0) "Anchor" else "T$c"

                    // 提取属于该姿态的纯净坐标数据
                    val src = FloatArray(pCount * 2)
                    val dst = FloatArray(pCount * 2)
                    for (j in 0 until pCount) {
                        src[j * 2] = chunkPairCoords[c][j * 4]
                        src[j * 2 + 1] = chunkPairCoords[c][j * 4 + 1]
                        dst[j * 2] = chunkPairCoords[c][j * 4 + 2]
                        dst[j * 2 + 1] = chunkPairCoords[c][j * 4 + 3]
                    }

                    // 致命一击：纯净版 1v1 RANSAC 几何校验
                    val ransac = RansacGeometryVerifier.verify(
                        src, dst, modelSize, modelSize, reprojectionError = 4.5
                    )
                    val inlierCount = ransac.inliers

                    // 准备安全分母 (防噪底惩罚)
                    val effectiveDenom = maxOf(20f, pCount.toFloat())
                    val ratio = inlierCount.toFloat() / effectiveDenom

                    // 提取内点计算平均 Cosine
                    val inlierPairs = IntArray(inlierCount * 2)
                    var inlierIdx = 0
                    for (j in 0 until pCount) {
                        if (ransac.inlierMask.getOrNull(j) == true) {
                            inlierPairs[inlierIdx * 2] = chunkPairIdx[c][j * 2]
                            inlierPairs[inlierIdx * 2 + 1] = chunkPairIdx[c][j * 2 + 1]
                            inlierIdx++
                        }
                    }
                    val meanCos = calculateInlierMeanCos(reference.descriptors, candidate.descriptors, inlierPairs)

                    // 动态加权公式计算独立得分
                    // 👇 === 🌟 全新自适应权重模型 (基于 Ratio) === 👇
                    // 核心理念：不仅绝对数量要够（防小样本偏差），比例（ratio）才是衡量几何稳定性的真理！
                    val countFactor = ((inlierCount - 25f) / 70f).coerceIn(0f, 1f)

                    // 2. 匹配质量因子 [0.0, 1.0]：下限 15% 保持不变，上限下调至 35%（密集点集下 35% 已是极其纯净的刚体）
                    val ratioFactor = ((ratio - 0.15f) / 0.20f).coerceIn(0f, 1f)

                    // 3. 综合置信度 (0.0 ~ 1.0)
                    val confidence = countFactor * ratioFactor

                    // 4. 平滑映射 ratioWeight：在 [0.20, 0.50] 之间连续变化
                    val ratioWeight = 0.5f - 0.3f * confidence
                    val cosWeight = 1.0f - ratioWeight

                    val chunkScore = (meanCos.pow(cosWeight) * ratio.pow(ratioWeight)).coerceIn(0f, 1f)

                    // 写入日志简报
                    val ratioPercent = ratio * 100f
                    breakdownLog.append("$name(good=$pCount, inls=$inlierCount, ratio=${"%.1f".format(ratioPercent)}%) | ")

                    // 决出最强王者
                    if (chunkScore > finalScore) {
                        finalScore = chunkScore
                        finalRatio = ratio
                        finalInliers = inlierCount
                        winnerName = name
                    }
                }

                if (finalScore > 0.1f) {
                    Log.d(TAG, "🧬 [PSEUDO-3D DEMUX] 🏆 WINNER: [$winnerName] score=${"%.3f".format(finalScore)}")
                    Log.d(TAG, "🧬 [PSEUDO-3D DEMUX] 📊 DETAILS: $breakdownLog")
                }

            } else {
                // 👉 保底逻辑：当只传入单一姿态时 (如单张图比对验证)
                val src = FloatArray(rawMatches.size * 2)
                val dst = FloatArray(rawMatches.size * 2)
                val rawPairs = IntArray(rawMatches.size * 2)

                for (i in rawMatches.indices) {
                    val pair = rawMatches[i]
                    src[i * 2] = reference.keypointsXY[pair.first * 2]
                    src[i * 2 + 1] = reference.keypointsXY[pair.first * 2 + 1]
                    dst[i * 2] = candidate.keypointsXY[pair.second * 2]
                    dst[i * 2 + 1] = candidate.keypointsXY[pair.second * 2 + 1]
                    rawPairs[i * 2] = pair.first
                    rawPairs[i * 2 + 1] = pair.second
                }

                val ransac = RansacGeometryVerifier.verify(
                    src, dst, modelSize, modelSize, reprojectionError = 4.5
                )

                finalInliers = ransac.inliers
                val effectiveDenom = maxOf(20f, rawMatches.size.toFloat())
                finalRatio = finalInliers.toFloat() / effectiveDenom

                val inlierPairs = IntArray(finalInliers * 2)
                var inlierIdx = 0
                for (i in rawMatches.indices) {
                    if (ransac.inlierMask.getOrNull(i) == true) {
                        inlierPairs[inlierIdx * 2] = rawPairs[i * 2]
                        inlierPairs[inlierIdx * 2 + 1] = rawPairs[i * 2 + 1]
                        inlierIdx++
                    }
                }

                val meanCos = calculateInlierMeanCos(reference.descriptors, candidate.descriptors, inlierPairs)
                val ratioWeight = if (finalInliers >= 30) 0.2f else 0.5f
                val cosWeight = 1.0f - ratioWeight
                finalScore = (meanCos.pow(cosWeight) * finalRatio.pow(ratioWeight)).coerceIn(0f, 1f)
            }

            MatchResult(
                passed = finalScore >= identityThreshold,
                score = finalScore,
                goodMatches = rawMatches.size,
                inliers = finalInliers,
                inlierRatio = finalRatio,
                bankIndex = bankIndex,
                quality = finalScore,
                spatialCoverage = candidateCoverage,
                referenceCount = reference.count,
                candidateCount = candidate.count,
                elapsedMs = (System.nanoTime() - start) / 1_000_000L
            )
        } catch (t: Throwable) {
            Log.w(TAG, "XFeat match failed", t)
            null
        }
    }

    // 辅助计算内点的平均特征余弦相似度 (基于纯点积，因为特征已 L2 归一化)
    private fun calculateInlierMeanCos(aDesc: FloatArray, bDesc: FloatArray, inliers: IntArray): Float {
        if (inliers.isEmpty()) return 0f
        var sumCos = 0f
        val numInliers = inliers.size / 2
        for (i in inliers.indices step 2) {
            val aIdx = inliers[i]
            val bIdx = inliers[i+1]
            var dot = 0f
            val aOffset = aIdx * 64
            val bOffset = bIdx * 64
            // 算 64 维向量的点积
            for (d in 0 until 64) {
                dot += aDesc[aOffset + d] * bDesc[bOffset + d]
            }
            sumCos += dot.coerceIn(-1f, 1f)
        }
        return sumCos / numInliers
    }

    private fun mutualNearestMatch(
        reference: XFeatFeatureSet,
        candidate: XFeatFeatureSet
    ): List<Pair<Int, Int>> {
        val best = ArrayList<Pair<Int, Int>>()
        val reverseBest = IntArray(candidate.count) { -1 }
        val reverseScore = FloatArray(candidate.count) { -Float.MAX_VALUE }

        for (i in 0 until reference.count) {
            var bestIdx = -1
            var bestSim = -Float.MAX_VALUE

            for (j in 0 until candidate.count) {
                val sim = cosine(reference, i, candidate, j)
                if (sim > bestSim) {
                    bestSim = sim
                    bestIdx = j
                }
            }

            if (bestIdx < 0) continue

            if (bestSim > reverseScore[bestIdx]) {
                reverseScore[bestIdx] = bestSim
                reverseBest[bestIdx] = i
            }
        }

        for (j in 0 until candidate.count) {
            val i = reverseBest[j]
            if (i >= 0) {
                best += i to j
            }
        }

        return best
    }

    private fun cosine(a: XFeatFeatureSet, ia: Int, b: XFeatFeatureSet, ib: Int): Float {
        if (a.descriptorCols != b.descriptorCols) return -1f

        val ao = ia * a.descriptorCols
        val bo = ib * b.descriptorCols

        var dot = 0f
        var na = 0f
        var nb = 0f

        for (k in 0 until a.descriptorCols) {
            val av = a.descriptors[ao + k]
            val bv = b.descriptors[bo + k]
            dot += av * bv
            na += av * av
            nb += bv * bv
        }

        if (na < 1e-8f || nb < 1e-8f) return -1f
        return (dot / sqrt(na.toDouble() * nb.toDouble()).toFloat()).coerceIn(-1f, 1f)
    }

    private fun toInput(patch: LocalGrayPatch): FloatArray {
        require(patch.width > 0 && patch.height > 0)
        require(patch.pixels.size == patch.width * patch.height)

        val src = org.opencv.core.Mat(patch.height, patch.width, org.opencv.core.CvType.CV_8UC1)
        val resized = org.opencv.core.Mat()
        val canvas = org.opencv.core.Mat(modelSize, modelSize, org.opencv.core.CvType.CV_8UC1, org.opencv.core.Scalar(0.0))

        return try {
            src.put(0, 0, patch.pixels)

            val scale = minOf(
                modelSize.toDouble() / patch.width.toDouble(),
                modelSize.toDouble() / patch.height.toDouble()
            )

            val resizedWidth = maxOf(1, kotlin.math.round(patch.width * scale).toInt())
            val resizedHeight = maxOf(1, kotlin.math.round(patch.height * scale).toInt())

            org.opencv.imgproc.Imgproc.resize(
                src,
                resized,
                org.opencv.core.Size(resizedWidth.toDouble(), resizedHeight.toDouble()),
                0.0,
                0.0,
                org.opencv.imgproc.Imgproc.INTER_AREA
            )

            val offsetX = (modelSize - resizedWidth) / 2
            val offsetY = (modelSize - resizedHeight) / 2

            resized.copyTo(canvas.submat(org.opencv.core.Rect(offsetX, offsetY, resizedWidth, resizedHeight)))

            val pixels = ByteArray(modelSize * modelSize)
            canvas.get(0, 0, pixels)

            FloatArray(modelSize * modelSize).also { out ->
                for (i in pixels.indices) {
                    out[i] = (pixels[i].toInt() and 0xFF) / 255f
                }
            }
        } finally {
            try { src.release() } catch (_: Throwable) {}
            try { resized.release() } catch (_: Throwable) {}
            try { canvas.release() } catch (_: Throwable) {}
        }
    }

    private fun spatialCoverage(points: FloatArray, width: Float, height: Float): Float {
        if (points.size < 4) return 0f

        var minX = width
        var maxX = 0f
        var minY = height
        var maxY = 0f

        var i = 0
        while (i + 1 < points.size) {
            val x = points[i]
            val y = points[i + 1]
            minX = minOf(minX, x)
            maxX = maxOf(maxX, x)
            minY = minOf(minY, y)
            maxY = maxOf(maxY, y)
            i += 2
        }

        return (((maxX - minX).coerceAtLeast(0f) * (maxY - minY).coerceAtLeast(0f)) / (width * height)).coerceIn(0f, 1f)
    }

    fun close() {
        try { runner?.close() } catch (_: Throwable) {}
    }

    val modelInputSize: Int
        get() = modelSize

    private val modelSize = XFeatQnnContextBinaryRunner.modelSizeForAsset(assetName)

    companion object {
        const val W8A16_ASSET_384 = XFeatQnnContextBinaryRunner.W8A16_ASSET_384
        const val W8A16_ASSET_480 = XFeatQnnContextBinaryRunner.W8A16_ASSET_480
        const val W8A16_ASSET_512 = XFeatQnnContextBinaryRunner.W8A16_ASSET_512
        const val W8A16_ASSET_640 = XFeatQnnContextBinaryRunner.W8A16_ASSET_640
        const val W8A16_ASSET = W8A16_ASSET_480
        const val INT8_ASSET = "xfeat_256_int8_sm8550.bin.SM8550.bin"

        const val MIN_GEOMETRY_POINTS = 4
        const val DEFAULT_IDENTITY_THRESHOLD = 0.60f

        private const val TOP_K = 1024
        private const val DETECTION_THRESHOLD = 0.05f
        private const val TARGET_INLIERS_FOR_FULL_SCORE = 12f
        private const val TAG = "XFeatIdentity"
    }
}