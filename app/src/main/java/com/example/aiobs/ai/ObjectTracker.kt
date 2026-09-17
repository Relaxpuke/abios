package com.example.aiobs.ai

import android.graphics.RectF
import android.util.Log

/**
 * Live target state holder for the three-stage pipeline:
 * NORMAL -> YOLO detections, LOCKED -> OSTrack box authority,
 * LOST -> YOLO proposals + XFeat identity recovery.
 */
class ObjectTracker {
    data class TrackedObject(
        val id: Int,
        var label: String,
        var cx: Float,
        var cy: Float,
        var w: Float,
        var h: Float,
        var state: TrackState = TrackState.ACTIVE,
        var confidence: Float = 0f,
        var currentMask: SegmentationMask? = null,
        var recoveryConfirmCount: Int = 0,
        var recoveryStage: String = "NORMAL",
        var xfeatScore: Float = 0f,
        var xfeatConfidence: Float = 0f,
        var xfeatInliers: Int = 0,
        var xfeatRatio: Float = 0f,
        var ostrackMissStreak: Int = 0
    )

    private data class InternalTrack(
        val target: TrackedObject,
        var anchorXFeat: XFeatFeatureSet? = null,
        // 🌟 DINOv2 精英记忆库：1 个初恋本源 + 4 个异变环境态
        var anchorDinov2: FloatArray? = null,
        val dinov2TemporalBank: java.util.ArrayDeque<FloatArray> = java.util.ArrayDeque(4)
    )

    private val tracks = mutableListOf<InternalTrack>()
    private var nextId = 1
    private var lockedId: Int? = null

    companion object {
        private const val TAG = "ObjectTracker"
        private const val OSTRACK_MISS_TO_LOST = 3
        private const val NORMAL_ASSOCIATION_MIN_IOU = 0.30f
    }

    // =========================================================================
    // 🌟 核心：DINOv2 多维环境记忆库 (Elite Bank) 🌟
    // =========================================================================

    @Synchronized
    fun anchorDinov2(targetId: Int): FloatArray? = findTrack(targetId)?.anchorDinov2?.clone()

    /**
     * 存入精英环境态：只记录环境/距离发生突变的 DINOv2 特征
     * 最大保留 4 个环境记忆，自动淘汰最老的。
     */
    @Synchronized
    fun addTemporalDinov2(targetId: Int, dinov2: FloatArray) {
        val track = findTrack(targetId) ?: return
        if (track.dinov2TemporalBank.size >= 4) {
            track.dinov2TemporalBank.removeFirst() // 挤掉最老的记忆
        }
        track.dinov2TemporalBank.addLast(dinov2.clone())
    }

    /**
     * 多槽位联合检索法庭：
     * 让候选人同时接受 [1 个初恋 Anchor] + [4 个环境态] 的联合审查，取最高分。
     */
    @Synchronized
    fun getBestDinov2Score(targetId: Int, candidateDinov2: FloatArray): Float {
        val track = findTrack(targetId) ?: return 0f
        var bestScore = 0f

        // 1. 和初恋 Anchor 比对 (权重最高，绝对真理)
        track.anchorDinov2?.let { anchor ->
            bestScore = maxOf(bestScore, cosineSimilarity(anchor, candidateDinov2))
        }

        // 2. 和环境精英库里的 4 个异变状态比对 (抗光线/距离干扰)
        track.dinov2TemporalBank.forEach { temporal ->
            bestScore = maxOf(bestScore, cosineSimilarity(temporal, candidateDinov2))
        }

        return bestScore
    }

    private fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        if (v1.size != v2.size) return 0f
        var dot = 0f; var n1 = 0f; var n2 = 0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
            n1 += v1[i] * v1[i]
            n2 += v2[i] * v2[i]
        }
        return if (n1 > 0f && n2 > 0f) dot / (kotlin.math.sqrt(n1) * kotlin.math.sqrt(n2)) else 0f
    }

    /**
     * 捕获初恋锚点：同时保存 XFeat 点云和 DINOv2 灵魂。
     * 注意：锁定瞬间会清空之前的 DINOv2 环境记忆库。
     */
    // 🌟 新增：检查是否已经有 DINOv2 本源记忆
    @Synchronized
    fun hasAnchorDinov2(targetId: Int): Boolean = findTrack(targetId)?.anchorDinov2 != null

    // 🌟 替换原来的 captureAnchor，解除捆绑！
    @Synchronized
    fun captureAnchor(
        targetId: Int,
        sourceMask: SegmentationMask? = null,
        xfeat: XFeatFeatureSet? = null,
        dinov2: FloatArray? = null
    ): Boolean {
        val track = findTrack(targetId) ?: return false
        track.target.currentMask = sourceMask?.copyMask()

        // 1. 如果有 XFeat，单独存 XFeat
        if (xfeat != null && xfeat.count >= XFeatLocalIdentityMatcher.MIN_GEOMETRY_POINTS) {
            track.anchorXFeat = xfeat.copyFeatures()
        }

        // 2. 如果有 DINOv2，单独存 DINOv2（互不干扰）
        if (dinov2 != null) {
            track.anchorDinov2 = dinov2.clone()
            track.dinov2TemporalBank.clear()
        }

        track.target.recoveryStage = "ANCHOR_READY_MULTI_MODAL"
        return true
    }

    // =========================================================================
    // 基础状态机与帧间流转逻辑
    // =========================================================================

    /**
     * Updates the NORMAL-stage YOLO view while preserving IDs for detections that
     * still correspond to an existing target. This is deliberately a lightweight
     * frame-to-frame association, not a second full tracker: LOCKED continues to
     * use OSTrack as the sole bbox authority.
     */
    @Synchronized
    fun update(detections: List<DetectionResult>): List<TrackedObject> {
        if (detections.isEmpty()) {
            tracks.clear()
            return emptyList()
        }

        if (tracks.isEmpty()) {
            detections.forEach { tracks += createTrack(it) }
            return snapshotForUi()
        }

        val previous = tracks.toList()
        val unmatchedTrackIndexes = previous.indices.toMutableSet()
        val unmatchedDetectionIndexes = detections.indices.toMutableSet()

        // Greedy highest-IoU matching. It keeps the NORMAL-stage ID stable without
        // introducing a heavyweight association/tracking implementation.
        val matches = mutableListOf<Pair<Int, Int>>()
        while (unmatchedTrackIndexes.isNotEmpty() && unmatchedDetectionIndexes.isNotEmpty()) {
            var bestTrack = -1
            var bestDetection = -1
            var bestIou = 0f

            for (trackIndex in unmatchedTrackIndexes) {
                val old = previous[trackIndex].target
                for (detectionIndex in unmatchedDetectionIndexes) {
                    val detection = detections[detectionIndex]
                    if (!old.label.equals(detection.label, ignoreCase = true)) continue

                    val iou = iou(old, detection)
                    if (iou > bestIou) {
                        bestIou = iou
                        bestTrack = trackIndex
                        bestDetection = detectionIndex
                    }
                }
            }

            if (bestTrack < 0 || bestDetection < 0 || bestIou < NORMAL_ASSOCIATION_MIN_IOU) break

            matches += bestTrack to bestDetection
            unmatchedTrackIndexes.remove(bestTrack)
            unmatchedDetectionIndexes.remove(bestDetection)
        }

        val nextTracks = mutableListOf<InternalTrack>()
        for ((trackIndex, detectionIndex) in matches) {
            val track = previous[trackIndex]
            applyDetection(track.target, detections[detectionIndex])
            track.target.state = TrackState.ACTIVE
            track.target.recoveryConfirmCount = 0
            track.target.recoveryStage = "NORMAL"
            track.target.ostrackMissStreak = 0
            nextTracks += track
        }

        // Detections that cannot be associated are new targets and receive a new ID.
        for (detectionIndex in unmatchedDetectionIndexes.sorted()) {
            nextTracks += createTrack(detections[detectionIndex])
        }

        tracks.clear()
        tracks.addAll(nextTracks.sortedBy { it.target.id })
        return snapshotForUi()
    }

    @Synchronized
    fun lockTarget(targetId: Int): Boolean {
        val track = findTrack(targetId) ?: return false
        lockedId = targetId
        track.target.state = TrackState.ACTIVE
        track.target.recoveryConfirmCount = 0
        track.target.recoveryStage = "LOCKED"
        Log.i(TAG, "LOCK id=$targetId")
        return true
    }

    @Synchronized
    fun unlockTarget() {
        lockedId = null
        tracks.forEach {
            it.target.recoveryConfirmCount = 0
            if (it.target.state == TrackState.LOST) it.target.state = TrackState.ACTIVE
        }
        Log.i(TAG, "UNLOCK")
    }

    fun lockedTargetId(): Int? = lockedId

    data class LockedUiSnapshot(
        val state: TrackState,
        val recoveryStage: String,
        val recoveryConfirmCount: Int
    )

    @Synchronized
    fun lockedTargetSnapshot(targetId: Int): LockedUiSnapshot? {
        val t = findTrack(targetId)?.target ?: return null
        return LockedUiSnapshot(t.state, t.recoveryStage, t.recoveryConfirmCount)
    }

    @Synchronized
    fun transformTargetBetweenRotations(targetId: Int, deltaClockwiseDegrees: Int): Boolean {
        val track = findTrack(targetId) ?: return false
        val t = track.target
        val left = (t.cx - t.w * 0.5f).coerceIn(0f, 1f)
        val top = (t.cy - t.h * 0.5f).coerceIn(0f, 1f)
        val right = (t.cx + t.w * 0.5f).coerceIn(0f, 1f)
        val bottom = (t.cy + t.h * 0.5f).coerceIn(0f, 1f)

        fun point(x: Float, y: Float): Pair<Float, Float> = when ((deltaClockwiseDegrees + 360) % 360) {
            90 -> (1f - y) to x
            180 -> (1f - x) to (1f - y)
            270 -> y to (1f - x)
            else -> x to y
        }

        val points = arrayOf(point(left, top), point(right, top), point(right, bottom), point(left, bottom))
        val minX = points.minOf { it.first }.coerceIn(0f, 1f)
        val minY = points.minOf { it.second }.coerceIn(0f, 1f)
        val maxX = points.maxOf { it.first }.coerceIn(0f, 1f)
        val maxY = points.maxOf { it.second }.coerceIn(0f, 1f)
        t.cx = (minX + maxX) * 0.5f
        t.cy = (minY + maxY) * 0.5f
        t.w = (maxX - minX).coerceAtLeast(0.001f)
        t.h = (maxY - minY).coerceAtLeast(0.001f)
        // Masks are in the previous frame coordinate space; force the next YOLO
        // refresh to provide a fresh mask while retaining identity anchors.
        t.currentMask = null
        t.ostrackMissStreak = 0
        return true
    }

    @Synchronized
    fun updateTrackedBox(targetId: Int, box: RectF): Boolean {
        val t = findTrack(targetId)?.target ?: return false
        if (t.state == TrackState.LOST) return false
        applyBox(t, box)
        t.ostrackMissStreak = 0
        t.state = TrackState.ACTIVE
        t.recoveryStage = "OSTRACK_ACTIVE"
        return true
    }

    @Synchronized
    fun noteOstrackMiss(targetId: Int): Boolean {
        val t = findTrack(targetId)?.target ?: return false
        if (t.state == TrackState.LOST) return true
        t.ostrackMissStreak++
        t.recoveryStage = "OSTRACK_MISS_${t.ostrackMissStreak}"
        if (t.ostrackMissStreak >= OSTRACK_MISS_TO_LOST) {
            t.state = TrackState.LOST
            Log.w(TAG, "LOST id=$targetId after OSTrack miss streak=${t.ostrackMissStreak}")
            return true
        }
        return false
    }

    @Synchronized
    fun markRelockPositionPending(targetId: Int): Boolean {
        val t = findTrack(targetId)?.target ?: return false
        t.state = TrackState.LOST
        t.recoveryStage = "RELOCK_POSITION_REFRESH_PENDING"
        return true
    }

    @Synchronized
    fun refreshRelockPosition(
        targetId: Int,
        detection: DetectionResult
    ): Boolean {
        val track = findTrack(targetId) ?: return false
        val t = track.target
        applyDetection(t, detection)
        t.ostrackMissStreak = 0
        t.recoveryConfirmCount = 0
        t.state = TrackState.ACTIVE
        t.recoveryStage = "RELOCK_POSITION_REFRESHED"
        return true
    }

    @Synchronized
    fun recoverWithCandidate(
        targetId: Int,
        detection: DetectionResult,
        identityScore: Float,
        xfeatResult: XFeatLocalIdentityMatcher.MatchResult? = null
    ): Boolean {
        val track = findTrack(targetId) ?: return false
        val t = track.target
        applyDetection(t, detection)
        t.recoveryConfirmCount++
        t.confidence = maxOf(t.confidence, identityScore.coerceIn(0f, 1f))
        if (xfeatResult != null) {
            t.xfeatScore = xfeatResult.score
            t.xfeatConfidence = xfeatResult.quality
            t.xfeatInliers = xfeatResult.inliers
            t.xfeatRatio = xfeatResult.inlierRatio
        }
        t.recoveryStage = "RELOCKED"
        t.state = TrackState.ACTIVE
        return true
    }

    @Synchronized
    fun candidatesForRecovery(detections: List<DetectionResult>): List<DetectionResult> {
        val id = lockedId ?: return emptyList()
        val label = findTrack(id)?.target?.label ?: return emptyList()
        return detections.filter { it.label.equals(label, ignoreCase = true) }
    }

    @Synchronized
    fun resetRecoveryConfirmCount(targetId: Int): Boolean {
        val t = findTrack(targetId)?.target ?: return false
        t.recoveryConfirmCount = 0
        return true
    }

    @Synchronized
    fun updateCurrentMask(targetId: Int, mask: SegmentationMask?) {
        findTrack(targetId)?.target?.currentMask = mask?.copyMask()
    }

    fun lockedTargetState(targetId: Int): TrackState? = findTrack(targetId)?.target?.state
    fun snapshot(): List<TrackedObject> = snapshotForUi()
    fun lifecycleSnapshot(targetId: Int): TrackedObject? = findTrack(targetId)?.target?.deepCopyForExternal()

    @Synchronized
    fun anchorXFeat(targetId: Int): XFeatFeatureSet? = findTrack(targetId)?.anchorXFeat?.copyFeatures()

    @Synchronized
    fun hasAnchorXFeat(targetId: Int): Boolean =
        findTrack(targetId)?.anchorXFeat?.count?.let { it >= XFeatLocalIdentityMatcher.MIN_GEOMETRY_POINTS } == true

    @Synchronized
    fun reset() {
        tracks.clear()
        nextId = 1
        lockedId = null
    }

    private fun createTrack(d: DetectionResult): InternalTrack {
        val t = TrackedObject(
            id = nextId++,
            label = d.label,
            cx = d.cx,
            cy = d.cy,
            w = d.w,
            h = d.h,
            confidence = d.confidence,
            currentMask = d.segmentationMask?.copyMask()
        )
        return InternalTrack(t)
    }

    private fun iou(target: TrackedObject, detection: DetectionResult): Float {
        val aLeft = target.cx - target.w * 0.5f
        val aTop = target.cy - target.h * 0.5f
        val aRight = target.cx + target.w * 0.5f
        val aBottom = target.cy + target.h * 0.5f

        val bLeft = detection.cx - detection.w * 0.5f
        val bTop = detection.cy - detection.h * 0.5f
        val bRight = detection.cx + detection.w * 0.5f
        val bBottom = detection.cy + detection.h * 0.5f

        val intersectionLeft = maxOf(aLeft, bLeft)
        val intersectionTop = maxOf(aTop, bTop)
        val intersectionRight = minOf(aRight, bRight)
        val intersectionBottom = minOf(aBottom, bBottom)

        val intersectionWidth = (intersectionRight - intersectionLeft).coerceAtLeast(0f)
        val intersectionHeight = (intersectionBottom - intersectionTop).coerceAtLeast(0f)
        val intersection = intersectionWidth * intersectionHeight
        if (intersection <= 0f) return 0f

        val areaA = (aRight - aLeft).coerceAtLeast(0f) * (aBottom - aTop).coerceAtLeast(0f)
        val areaB = (bRight - bLeft).coerceAtLeast(0f) * (bBottom - bTop).coerceAtLeast(0f)
        val union = areaA + areaB - intersection
        return if (union > 0f) intersection / union else 0f
    }

    private fun applyDetection(t: TrackedObject, d: DetectionResult) {
        t.cx = d.cx
        t.cy = d.cy
        t.w = d.w
        t.h = d.h
        t.confidence = d.confidence
        t.currentMask = d.segmentationMask?.copyMask()
    }

    private fun applyBox(t: TrackedObject, box: RectF) {
        t.cx = (box.left + box.right) * 0.5f
        t.cy = (box.top + box.bottom) * 0.5f
        t.w = box.width()
        t.h = box.height()
    }

    private fun findTrack(id: Int): InternalTrack? = tracks.firstOrNull { it.target.id == id }

    private fun snapshotForUi(): List<TrackedObject> = tracks.map { it.target.deepCopyForExternal() }

    private fun TrackedObject.deepCopyForExternal(): TrackedObject = copy(
        currentMask = currentMask?.copyMask()
    )
}