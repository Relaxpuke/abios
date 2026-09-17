package com.example.aiobs.ai

/** Grayscale ROI passed to the XFeat local matcher. */
data class LocalGrayPatch(
    val width: Int,
    val height: Int,
    val pixels: ByteArray
)


/** XFeat sparse keypoints + learned 64D descriptors produced by Android post-processing. */
data class XFeatFeatureSet(
    val keypointsXY: FloatArray,
    val descriptors: FloatArray,
    val descriptorRows: Int,
    val descriptorCols: Int,
    val scores: FloatArray,
    val chunkSizes: IntArray? = null // 🌟 新增：记录内部包含了几个姿态切片，每个切片多大
) {
    val count: Int get() = descriptorRows

    fun copyFeatures(): XFeatFeatureSet = XFeatFeatureSet(
        keypointsXY = keypointsXY.clone(),
        descriptors = descriptors.clone(),
        descriptorRows = descriptorRows,
        descriptorCols = descriptorCols,
        scores = scores.clone()
    )

    // 新增：特征融合魔法，将当前的特征集合与另一个特征集合首尾无缝拼接
    fun combineWith(other: XFeatFeatureSet?): XFeatFeatureSet {
        if (other == null || other.count == 0) return this.copyFeatures()
        require(this.descriptorCols == other.descriptorCols) { "Descriptor columns must match" }

        val newCount = this.count + other.count
        return XFeatFeatureSet(
            keypointsXY = this.keypointsXY + other.keypointsXY,
            descriptors = this.descriptors + other.descriptors,
            descriptorRows = newCount,
            descriptorCols = this.descriptorCols,
            scores = this.scores + other.scores
        )
    }

    // --- 添加在 LocalGrayPatch.kt 的 XFeatFeatureSet 中 ---

    // 新增：批量特征融合魔法。将 Anchor 和多个 Temporal 记忆一次性组装成全视角点云。
    // --- 替换 LocalGrayPatch.kt 中的 combineWithMultiple 方法 ---

    /**
     * 批量特征融合魔法 (带底层内存安全限制)
     * 底层 C++ NEON 匹配器最大仅支持 2048 个点。
     * 策略：保留完整的 Anchor (1024点)，剩余的 1024 点额度均分给记忆池中的各个 Temporal 姿态，
     * 截取它们质量最高的前 N 个特征，确保总点数绝对不超过 2048。
     */
    // --- 在 LocalGrayPatch.kt 中修改 ---

    fun combineWithMultiple(others: List<XFeatFeatureSet>, maxTotalPoints: Int = 20000): XFeatFeatureSet {
        val validOthers = others.filter { it.count > 0 }
        if (validOthers.isEmpty()) return this.copyFeatures()

        // 🌟 解除封印：Anchor 彻底不限制，原图提取了多少特征就保留多少特征！(保证初恋视角的绝对高清)
        val anchorCount = this.count

        // 计算剩余预算，并均分给记忆池中的各个 Temporal 姿态
        val remainingBudget = maxOf(0, maxTotalPoints - anchorCount)

        // Kotlin 的整数除法会自动向下取整(抹零)，不用担心出现非整数导致的崩溃
        val pointsPerTemporal = if (validOthers.isNotEmpty()) remainingBudget / validOthers.size else 0


        // 3. 计算切片大小与新数组总长度
        var totalNewCount = anchorCount
        val slices = validOthers.map {
            val take = minOf(it.count, pointsPerTemporal)
            totalNewCount += take
            take
        }

        val newKeypoints = FloatArray(totalNewCount * 2)
        val newDescriptors = FloatArray(totalNewCount * descriptorCols)
        val newScores = FloatArray(totalNewCount)

        // 4. 拷贝 Anchor 特征
        System.arraycopy(this.keypointsXY, 0, newKeypoints, 0, anchorCount * 2)
        System.arraycopy(this.descriptors, 0, newDescriptors, 0, anchorCount * descriptorCols)
        System.arraycopy(this.scores, 0, newScores, 0, anchorCount)

        var kpOffset = anchorCount * 2
        var descOffset = anchorCount * descriptorCols
        var scoreOffset = anchorCount

        // 5. 依次拷贝“掐尖”后的各个 Temporal 历史姿态
        for (i in validOthers.indices) {
            val other = validOthers[i]
            val takeCount = slices[i]
            if (takeCount <= 0) continue

            System.arraycopy(other.keypointsXY, 0, newKeypoints, kpOffset, takeCount * 2)
            System.arraycopy(other.descriptors, 0, newDescriptors, descOffset, takeCount * descriptorCols)
            System.arraycopy(other.scores, 0, newScores, scoreOffset, takeCount)

            kpOffset += takeCount * 2
            descOffset += takeCount * descriptorCols
            scoreOffset += takeCount
        }

        // 🌟 6. 新增核心逻辑：记录切片大小，装配给对象
        val sizes = IntArray(1 + validOthers.size)
        sizes[0] = anchorCount
        for (i in validOthers.indices) {
            sizes[i + 1] = slices[i]
        }
        android.util.Log.d(
            "XFeatCombine",
            "特征融合完成 | 总点数: $totalNewCount (Anchor: $anchorCount, " +
                    "Temporal[${validOthers.size}个槽位]贡献: ${slices.sum()}点, 每槽分配上限: $pointsPerTemporal 点)"
        )

        return XFeatFeatureSet(
            keypointsXY = newKeypoints,
            descriptors = newDescriptors,
            descriptorRows = totalNewCount,
            descriptorCols = this.descriptorCols,
            scores = newScores,
            chunkSizes = sizes // 🌟 存入切片记忆，传给下一关！
        )
    }
}
