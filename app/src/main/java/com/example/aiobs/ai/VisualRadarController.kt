package com.example.aiobs.ai

import android.graphics.RectF
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.example.aiobs.performance.AiPerformanceProfile
import com.example.aiobs.performance.AiPerformanceProfileFactory
import com.example.aiobs.performance.AiPerformanceSnapshot
import com.example.aiobs.performance.PipelineProfiler
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max

/**
 * Simple three-state live architecture.
 *
 * NORMAL:
 *   every frame -> YOLO11s-seg detection only.
 *
 * LOCKED:
 *   every frame -> OSTrack-256 direct QNN HTP -> trackedBox.
 *   A low-frequency YOLO-Seg refresh provides the foreground mask used by the
 *   XFeat temporal-anchor path. Invalid/low-confidence OSTrack results are
 *   counted as misses and can transition to LOST.
 *
 * LOST (Ultra-fast Single-Frame Recovery):
 *   YOLO provides same-class proposals -> Cheap Color/Confidence Ranking
 *   -> Top 2 Candidates -> XFeat Identity Verification (Combined Anchor+Temporal)
 *   -> Single threshold PASS/FAIL.
 */
class VisualRadarController(
    private val frameSource: FrameSource,
    context: android.content.Context,
    private val runtimeConfig: com.example.aiobs.core.ModelRuntimeConfig.Snapshot =
        com.example.aiobs.core.ModelRuntimeConfig.Snapshot(),
    private val tuningConfig: com.example.aiobs.core.RuntimeTuningConfig =
        com.example.aiobs.core.RuntimeTuningConfig(),
    private val rotationProvider: () -> Int = { 0 },
    private val onTargetsUpdated: (List<TrackedObject>) -> Unit
) {
    interface FrameSource {
        fun start(
            onFrame: (AiFrame, Long) -> Unit,
            tryAcquireFrame: () -> Long? = { 1L },
            releaseFrame: (Long) -> Unit = {}
        ): Boolean

        fun onFrameConsumed(slotToken: Long)
        fun stop()
    }

    private val profile: AiPerformanceProfile = AiPerformanceProfileFactory.create()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val workerThread = AtomicReference<Thread?>(null)

    private val worker: ExecutorService =
        Executors.newSingleThreadExecutor { r ->
            Thread(r, "AIOBS-OSTRACK-AI").apply {
                isDaemon = true
                priority = Thread.NORM_PRIORITY
                workerThread.set(this)
            }
        }

    private val running = AtomicBoolean(false)
    private val destroyed = AtomicBoolean(false)
    private val generation = AtomicLong(0L)
    private val aiFrozen = AtomicBoolean(false)

    private val slotTokenGenerator = AtomicLong(0L)
    private val inferenceSlotOwnerLock = Any()
    private var inferenceSlotOwnerToken = 0L
    private var inferenceSlotOwnerSession = 0L
    private var inferenceSlotOwnerFrameSeq = 0L
    private var inferenceSlotOwnerStartNs = 0L

    private var diagLockedTemporalRefreshes = 0L
    private var diagLockedTemporalSuccess = 0L
    private var diagLockedTemporalNoMask = 0L
    private var diagLastStatsMs = 0L
    private val aiFrameSequence = AtomicLong(0L)
    private val lastAiDropLogNs = AtomicLong(0L)

    private val tracker = ObjectTracker()
    private val pipeline = ObjectDetectionPipeline(context, runtimeConfig, tuningConfig)
    private val profiler = PipelineProfiler()

    @Volatile private var lockedId: Int? = null

    // ==========================================
    // 极简恢复状态追踪 (Single-Frame Recovery)
    // ==========================================
    private var recoveryStartMs: Long = 0L
    private var isRecovering: Boolean = false
    private var latestTemporalColor: FloatArray? = null

    private var pendingRelockPosition: PendingRelockPosition? = null
    // 🌟 扩容：将多姿态记忆池从 7 提升到 49 (加上 Anchor 共 50 个视角)
    private val MAX_MEMORY_BANK_SIZE = 100
    private val temporalMemoryBank = java.util.ArrayDeque<XFeatFeatureSet>(MAX_MEMORY_BANK_SIZE)
    private var latestTemporalTemplateBox: RectF? = null
    // 👇 新增：记录当前锁定目标有史以来总共存入了多少个新姿态
    private var totalPosesAdded = 0

    private var lastLockedTemporalRefreshMs = 0L
    private var lockedTemporalRefreshNotBeforeMs = 0L

    private var frameCounter = 0L
    private var availableFrames = 0L
    private var processedFrames = 0L
    private var droppedFrames = 0L
    private var firstStatsNs = 0L

    private var lastInferenceMs = 0f
    private var lastEndToEndMs = 0f
    private var lastRawDetectionCount = 0
    private var lastAcceptedDetectionCount = 0

    @Volatile private var lastRecoveryUiMs = 0f
    @Volatile private var lastDinov2Ms = 0f
    @Volatile private var lastRecoveryUiCandidates = 0
    @Volatile private var lastRecoveryXFeatExtractMs = 0f
    @Volatile private var lastRecoveryXFeatMatchMs = 0f

    /** Per-frame timings. This prevents YOLO/OSTrack metrics from overwriting each other. */
    private data class FramePerformanceContext(
        val token: PipelineProfiler.FrameToken,
        var yoloTotalMs: Float? = null,
        var yoloInferMs: Float? = null,
        var ostrackMs: Float? = null
    )

    private var lastPreparedFrame: ObjectDetectionPipeline.PreparedFrame? = null

    // Last AI-oriented frame rotation. When the physical device rotates while the
    // Activity is locked, the next PreparedFrame carries a new rotation even though
    // Android does not dispatch a configuration change. The worker then transforms
    // the existing locked box into the new normalized coordinate space and refreshes
    // only the orientation-sensitive OSTrack/template state.
    private var lastProcessedRotationDegrees: Int? = null

    private val lockedTemporalFirstRefreshDelayMs: Long get() = tuningConfig.lockedYoloFirstRefreshDelayMs
    private val lockedTemporalRefreshIntervalMs: Long get() = tuningConfig.lockedYoloRefreshIntervalMs

    companion object {
        private const val TAG = "VisualRadar"
    }

    fun start() {
        check(!destroyed.get()) { "VisualRadarController is destroyed" }
        if (!running.compareAndSet(false, true)) return
        val session = generation.incrementAndGet()
        resetCounters()

        lastProcessedRotationDegrees = null

        if (!frameSource.start(
                onFrame = { frame, slotToken -> onFrameAvailable(session, frame, slotToken) },
                tryAcquireFrame = { tryAcquire(session) },
                releaseFrame = { slotToken -> release(session, slotToken) }
            )
        ) {
            running.set(false)
            Log.e(TAG, "Frame source failed to start")
        }
    }

    fun stop(clearLock: Boolean = false) {
        if (destroyed.get()) return
        running.set(false)
        generation.incrementAndGet()
        frameSource.stop()

        try {
            worker.execute {
                try {
                    if (clearLock) {
                        clearLockInternal(publish = false)
                    } else {
                        isRecovering = false
                        recoveryStartMs = 0L
                        pendingRelockPosition = null
                        clearLatestTemporalTemplate()
                        resetLockedTemporalRefreshState()
                    }
                    tracker.reset()
                    pipeline.resetOstrackTemplate()
                    recyclePreparedBitmap()
                } catch (t: Throwable) {
                    Log.e(TAG, "Deferred AI stop cleanup failed", t)
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to enqueue AI stop cleanup", t)
        }

        mainHandler.post { onTargetsUpdated(emptyList()) }
    }

    fun toggleLock(targetId: Int) {
        if (destroyed.get()) return
        try {
            worker.execute {
                if (lockedId == targetId) {
                    clearLockInternal()
                    return@execute
                }
                val target = tracker.lifecycleSnapshot(targetId) ?: return@execute
                if (!tracker.lockTarget(targetId)) return@execute

                lockedId = target.id
                isRecovering = false
                recoveryStartMs = 0L
                clearLatestTemporalTemplate()
                resetLockedTemporalRefreshState()
                lockedTemporalRefreshNotBeforeMs = SystemClock.elapsedRealtime() + lockedTemporalFirstRefreshDelayMs
                pipeline.resetOstrackTemplate()

                Log.i(TAG, "LOCK requested id=${target.id}; OSTrack active next frame")
                publishTargets()
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to enqueue lock request", t)
        }
    }

    fun clearLock() {
        if (destroyed.get()) return
        try {
            worker.execute { clearLockInternal() }
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to enqueue clear-lock request", t)
        }
    }

    private fun clearLockInternal(publish: Boolean = true) {
        lockedId = null
        isRecovering = false
        recoveryStartMs = 0L
        pendingRelockPosition = null
        clearLatestTemporalTemplate()
        resetLockedTemporalRefreshState()
        tracker.unlockTarget()
        pipeline.resetOstrackTemplate()
        if (publish) publishTargets()
    }

    fun getLockedTargetId(): Int? = lockedId

    fun toggleAiFrozen(): Boolean {
        val next = !aiFrozen.get()
        aiFrozen.set(next)
        return next
    }

    fun performanceSnapshot(): AiPerformanceSnapshot {
        val nowNs = SystemClock.elapsedRealtimeNanos()
        val sec = if (firstStatsNs > 0L) ((nowNs - firstStatsNs).coerceAtLeast(1L)) / 1_000_000_000f else 0f

        return AiPerformanceSnapshot(
            profileName = profile.profileName,
            socModel = profile.socModel,
            delegate = pipeline.delegateName(),
            targetFps = profile.targetInferenceFps,
            processedFps = if (sec > 0f) processedFrames / sec else 0f,
            captureFps = if (sec > 0f) availableFrames / sec else 0f,
            inferenceMs = profiler.snapshot().inferenceMs,
            endToEndMs = profiler.snapshot().endToEndMs,
            droppedFrames = droppedFrames,
            rawDetectionCount = lastRawDetectionCount,
            acceptedDetectionCount = lastAcceptedDetectionCount,
            pipeline = profiler.snapshot(),
            yoloTotalMs = profiler.snapshot().yoloTotalMs,
            yoloInferMs = profiler.snapshot().yoloInferMs,
            ostrackMs = profiler.snapshot().ostrackMs,
            dinov2Ms = lastDinov2Ms,
            xfeatExtractMs = lastRecoveryXFeatExtractMs,
            xfeatMatchMs = lastRecoveryXFeatMatchMs,
            recoveryMs = lastRecoveryUiMs,
            recoveryCandidates = lastRecoveryUiCandidates,
            yoloModel = "YOLO11n-Seg",
            yoloBackend = pipeline.yoloBackendName(),
            ostrackModel = "OSTrack-256",
            ostrackBackend = "QNN-HTP",
            xfeatModel = "XFeat-480 W8A16",
            xfeatBackend = pipeline.xfeatBackendName()
        )
    }

    data class UiStatusSnapshot(
        val state: TrackState?,
        val recoveryStage: String,
        val recoveryConfirmCount: Int,
        val recoveryActive: Boolean
    )

    fun uiStatusSnapshot(): UiStatusSnapshot {
        val id = lockedId ?: tracker.lockedTargetId()
        if (id == null) return UiStatusSnapshot(null, "NONE", 0, isRecovering)
        val t = tracker.lockedTargetSnapshot(id)
        return if (t == null) {
            UiStatusSnapshot(null, "NONE", 0, isRecovering)
        } else {
            UiStatusSnapshot(t.state, t.recoveryStage, t.recoveryConfirmCount, isRecovering)
        }
    }

    fun destroy() {
        if (!destroyed.compareAndSet(false, true)) return
        running.set(false)
        generation.incrementAndGet()
        frameSource.stop()
        lastProcessedRotationDegrees = null
        aiFrozen.set(false)

        try {
            worker.execute {
                try {
                    clearLockInternal(publish = false)
                    tracker.reset()
                } catch (t: Throwable) {
                    Log.e(TAG, "Tracker reset during destroy failed", t)
                }
                try {
                    recyclePreparedBitmap()
                } catch (t: Throwable) {
                    Log.e(TAG, "Prepared bitmap recycle during destroy failed", t)
                }
                try {
                    pipeline.destroy()
                } catch (t: Throwable) {
                    Log.e(TAG, "Pipeline destroy failed", t)
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to enqueue native teardown barrier", t)
        }

        worker.shutdown()
        val current = Thread.currentThread()
        if (workerThread.get() !== current) {
            try {
                if (!worker.awaitTermination(10, TimeUnit.SECONDS)) {
                    Log.e(TAG, "AI worker did not terminate within 10s")
                }
            } catch (ie: InterruptedException) {
                Thread.currentThread().interrupt()
                Log.e(TAG, "Interrupted while awaiting AI worker termination", ie)
            }
        }
        mainHandler.removeCallbacksAndMessages(null)
    }

    private fun onFrameAvailable(session: Long, frame: AiFrame, slotToken: Long) {
        if (!running.get() || generation.get() != session || aiFrozen.get()) {
            release(session, slotToken)
            frameSource.onFrameConsumed(slotToken)
            return
        }

        availableFrames++
        if (firstStatsNs == 0L) firstStatsNs = SystemClock.elapsedRealtimeNanos()

        val frameSeq = aiFrameSequence.incrementAndGet()
        attachOwnerFrameSeq(slotToken, frameSeq)
        val profilerToken = profiler.newToken(
            cameraAvailableNs = frame.availableStartNs,
            cameraHandoffNs = frame.availableEndNs
        )

        try {
            worker.execute {
                val workerStartNs = SystemClock.elapsedRealtimeNanos()
                try {
                    processOneFrame(
                        session = session,
                        frame = frame,
                        frameSeq = frameSeq,
                        slotToken = slotToken,
                        workerStartNs = workerStartNs,
                        profilerToken = profilerToken
                    )
                } finally {
                    // completion logged in processOneFrame
                }
            }
        } catch (t: Throwable) {
            release(session, slotToken)
            frameSource.onFrameConsumed(slotToken)
            Log.e(TAG, "worker submit failed seq=$frameSeq", t)
        }
    }

    private fun tryAcquire(session: Long): Long? {
        if (!running.get() || generation.get() != session || aiFrozen.get()) return null
        val nowNs = SystemClock.elapsedRealtimeNanos()
        var token = slotTokenGenerator.incrementAndGet()
        if (token == 0L) token = slotTokenGenerator.incrementAndGet()

        synchronized(inferenceSlotOwnerLock) {
            if (inferenceSlotOwnerToken == 0L) {
                inferenceSlotOwnerToken = token
                inferenceSlotOwnerSession = session
                inferenceSlotOwnerFrameSeq = 0L
                inferenceSlotOwnerStartNs = nowNs
                return token
            }
            droppedFrames++
            val lastNs = lastAiDropLogNs.get()
            if (nowNs - lastNs >= 1_000_000_000L && lastAiDropLogNs.compareAndSet(lastNs, nowNs)) {
                Log.i(TAG, "[AI_DROP] reason=AI_SLOT_BUSY session=$session totalDropped=$droppedFrames")
            }
            return null
        }
    }

    private fun attachOwnerFrameSeq(slotToken: Long, frameSeq: Long) {
        synchronized(inferenceSlotOwnerLock) {
            if (inferenceSlotOwnerToken == slotToken) {
                if (inferenceSlotOwnerFrameSeq == 0L) inferenceSlotOwnerFrameSeq = frameSeq
            }
        }
    }

    private fun release(session: Long, slotToken: Long) {
        synchronized(inferenceSlotOwnerLock) {
            if (inferenceSlotOwnerToken == slotToken && inferenceSlotOwnerSession == session) {
                inferenceSlotOwnerToken = 0L
                inferenceSlotOwnerSession = 0L
                inferenceSlotOwnerFrameSeq = 0L
                inferenceSlotOwnerStartNs = 0L
            }
        }
    }

    private fun processOneFrame(
        session: Long,
        frame: AiFrame,
        frameSeq: Long,
        slotToken: Long,
        workerStartNs: Long,
        profilerToken: PipelineProfiler.FrameToken
    ) {
        val framePerf = FramePerformanceContext(token = profilerToken)
        try {
            val prepared = pipeline.prepareFrame(frame, rotationProvider()) ?: return

            val previousRotation = lastProcessedRotationDegrees
            val currentRotation = prepared.rotationDegrees
            if (previousRotation != null && previousRotation != currentRotation) {
                val delta = ((currentRotation - previousRotation) % 360 + 360) % 360
                val lockedTargetId = lockedId
                if (lockedTargetId != null) {
                    tracker.transformTargetBetweenRotations(lockedTargetId, delta)
                    clearLatestTemporalTemplate()
                    resetLockedTemporalRefreshState()
                    pipeline.resetOstrackTemplate()
                    isRecovering = false
                    recoveryStartMs = 0L
                    pendingRelockPosition = null
                    Log.i(TAG, "AI frame rotation changed $previousRotation->$currentRotation; preserving lock id=$lockedTargetId and refreshing orientation-sensitive state")
                }
            }
            lastProcessedRotationDegrees = currentRotation

            recyclePreparedBitmap()
            lastPreparedFrame = prepared
            frameCounter++

            val currentLockedId = lockedId
            if (currentLockedId == null) {
                processNormalFrame(session, prepared, workerStartNs, framePerf)
                return
            }

            val current = tracker.lifecycleSnapshot(currentLockedId) ?: run {
                clearLockInternal(publish = false)
                publishAndFinish(session, tracker.snapshot(), workerStartNs, prepared = prepared, framePerf = framePerf)
                return
            }

            if (processPendingRelockPosition(session, prepared, currentLockedId, workerStartNs, framePerf)) return
            if (processRecoveryFrame(session, prepared, frameSeq, current, currentLockedId, workerStartNs, framePerf)) return

            processLockedFrame(session, prepared, current, currentLockedId, frameSeq, workerStartNs, framePerf)

        } catch (t: Throwable) {
            Log.e(TAG, "frame failed", t)
        } finally {
            val frameEndNs = SystemClock.elapsedRealtimeNanos()
            val frameMs = (frameEndNs - workerStartNs).coerceAtLeast(0L) / 1_000_000f
            BaselinePerfLogger.frame(
                seq = frameSeq,
                totalMs = frameMs,
                state = when {
                    isRecovering -> "RECOVERY"
                    lockedId != null -> "LOCKED"
                    else -> "NORMAL"
                }
            )
            recyclePreparedBitmap()
            release(session, slotToken)
            frameSource.onFrameConsumed(slotToken)
        }
    }

    private fun processNormalFrame(
        session: Long,
        prepared: ObjectDetectionPipeline.PreparedFrame,
        workerStartNs: Long,
        framePerf: FramePerformanceContext
    ) {
        val detections = pipeline.runYolo(prepared, YoloDetectionMode.BASIC)
        framePerf.yoloTotalMs = pipeline.currentPerceptionTotalMs()
        framePerf.yoloInferMs = pipeline.currentInferenceMs()
        lastInferenceMs = framePerf.yoloInferMs ?: 0f
        lastRawDetectionCount = detections.size
        lastAcceptedDetectionCount = detections.size
        val targets = tracker.update(detections)
        publishAndFinish(session, targets, workerStartNs, prepared = prepared, framePerf = framePerf)
    }

    /** Handles the one-frame YOLO position refresh after identity recovery. */
    private fun processPendingRelockPosition(
        session: Long,
        prepared: ObjectDetectionPipeline.PreparedFrame,
        currentLockedId: Int,
        workerStartNs: Long,
        framePerf: FramePerformanceContext
    ): Boolean {
        val pendingPosition = pendingRelockPosition ?: return false
        if (pendingPosition.targetId != currentLockedId) {
            pendingRelockPosition = null
            return false
        }

        val freshFrameSize = rotatedFrameSize(prepared.bitmap, prepared.rotationDegrees)
        val freshFrameWidth = freshFrameSize.first
        val freshFrameHeight = freshFrameSize.second
        val seedBox = RectF(
            pendingPosition.confirmedBoxNormalized.left * freshFrameWidth,
            pendingPosition.confirmedBoxNormalized.top * freshFrameHeight,
            pendingPosition.confirmedBoxNormalized.right * freshFrameWidth,
            pendingPosition.confirmedBoxNormalized.bottom * freshFrameHeight
        )

        Log.i(TAG, "[RELOCK_POSITION_REFRESH] id=$currentLockedId source=NEXT_FRESH_FRAME seedBox=$seedBox label=${pendingPosition.label}")

        val refreshed = pipeline.findSameClassYoloDetectionForMask(
            prepared = prepared,
            ostrackBox = seedBox,
            expectedLabel = pendingPosition.label
        )
        framePerf.yoloTotalMs = pipeline.currentPerceptionTotalMs()
        framePerf.yoloInferMs = pipeline.currentInferenceMs()

        if (refreshed == null) {
            Log.w(TAG, "[RELOCK_POSITION_REFRESH] id=$currentLockedId ready=false reason=NO_SAME_CLASS_DETECTION")
            lastInferenceMs = framePerf.yoloInferMs ?: 0f
            publishAndFinish(session, tracker.snapshot(), workerStartNs, prepared = prepared, framePerf = framePerf)
            return true
        }

        val refreshedOk = tracker.refreshRelockPosition(currentLockedId, refreshed.detection)
        if (!refreshedOk) {
            Log.w(TAG, "[RELOCK_POSITION_REFRESH] id=$currentLockedId ready=false reason=TRACKER_REFRESH_FAILED box=${refreshed.boxPx}")
            lastInferenceMs = framePerf.yoloInferMs ?: 0f
            publishAndFinish(session, tracker.snapshot(), workerStartNs, prepared = prepared, framePerf = framePerf)
            return true
        }

        // 改成这样：
        pendingRelockPosition = null
        resetLockedTemporalRefreshState()
        lockedTemporalRefreshNotBeforeMs = SystemClock.elapsedRealtime() + lockedTemporalFirstRefreshDelayMs
        pipeline.resetOstrackTemplate()

        Log.i(TAG, "[RELOCKED] id=$currentLockedId positionRefreshed=true OSTrack template reset")
        publishAndFinish(session, tracker.snapshot(), workerStartNs, prepared = prepared, framePerf = framePerf)
        return true
    }

    /** Runs Ultra-fast Single-Frame Recovery. */
    /** Runs Ultra-fast Single-Frame Recovery. */
    private fun processRecoveryFrame(
        session: Long,
        prepared: ObjectDetectionPipeline.PreparedFrame,
        frameSeq: Long,
        current: TrackedObject,
        currentLockedId: Int,
        workerStartNs: Long,
        framePerf: FramePerformanceContext
    ): Boolean {
        if (current.state != TrackState.LOST) return false
        val nowMs = SystemClock.elapsedRealtime()

        if (!isRecovering) {
            isRecovering = true
            recoveryStartMs = nowMs
            Log.w(TAG, "OSTRACK RECOVERY START id=$currentLockedId timeout=60000ms")
        }

        if (nowMs - recoveryStartMs >= 60_000L) {
            Log.w(TAG, "OSTRACK RECOVERY TIMEOUT id=$currentLockedId elapsed=${nowMs - recoveryStartMs}ms -> NORMAL")
            clearLockInternal(publish = false)
            val detections = pipeline.runYolo(prepared, YoloDetectionMode.BASIC)
            framePerf.yoloTotalMs = pipeline.currentPerceptionTotalMs()
            framePerf.yoloInferMs = pipeline.currentInferenceMs()
            lastInferenceMs = framePerf.yoloInferMs ?: 0f
            publishAndFinish(session, tracker.update(detections), workerStartNs, prepared = prepared, framePerf = framePerf)
            return true
        }

        val recoveryBitmap = prepared.canonicalBitmap()
        val recoveryFrameWidth = recoveryBitmap.width
        val recoveryFrameHeight = recoveryBitmap.height

        val recoveryFrameStartNs = SystemClock.elapsedRealtimeNanos()
        var recoveryXFeatExtractMs = 0f
        var recoveryXFeatMatchMs = 0f

        try {
            val detections = pipeline.runYolo(prepared, YoloDetectionMode.RECOVERY)
            framePerf.yoloTotalMs = pipeline.currentPerceptionTotalMs()
            framePerf.yoloInferMs = pipeline.currentInferenceMs()
            lastInferenceMs = framePerf.yoloInferMs ?: 0f
            lastRawDetectionCount = detections.size
            lastAcceptedDetectionCount = detections.size

            val candidates = tracker.candidatesForRecovery(detections)
            if (candidates.isEmpty()) {
                publishAndFinish(session, tracker.snapshot(), workerStartNs, prepared = prepared, framePerf = framePerf)
                return true
            }

            // ========================================================
            // 🌟 Stage 2: DINOv2 灵魂法庭审查
            // ========================================================
            var recoveryDinov2Ms = 0f
            val rankedCandidates = candidates
                .filter { isValidRecoveryBox(boxToPixels(it, recoveryFrameWidth, recoveryFrameHeight), recoveryFrameWidth, recoveryFrameHeight) }
                .mapNotNull { candidate ->
                    val boxPx = boxToPixels(candidate, recoveryFrameWidth, recoveryFrameHeight)

                    // 🚀 DINOv2 提取
                    val dStartNs = System.nanoTime()
                    val dinov2Vector = pipeline.extractDinov2(recoveryBitmap, boxPx)
                    recoveryDinov2Ms += (System.nanoTime() - dStartNs) / 1_000_000f

                    if (dinov2Vector == null) return@mapNotNull null

                    // 🌟 进阶法庭：让候选人接受 Anchor + 4个精英环境记忆 的联合审查
                    val bestCosine = tracker.getBestDinov2Score(currentLockedId, dinov2Vector)

                    Log.w(TAG, "👁️ DINOv2 多槽位审查: YOLO置信度=${fmt(candidate.confidence)}, 最高灵魂相似度=${fmt(bestCosine)} (放行门限=0.70)")

                    Pair(candidate, bestCosine)
                }
                .filter { it.second >= 0.70f } // 门限收紧到 0.70
                .sortedByDescending { it.second }
                .take(2)

            lastRecoveryUiCandidates = rankedCandidates.size
            if (recoveryDinov2Ms > 0f) {
                lastDinov2Ms = recoveryDinov2Ms
            }

            if (rankedCandidates.isEmpty()) {
                publishAndFinish(session, tracker.snapshot(), workerStartNs, prepared = prepared, framePerf = framePerf)
                return true
            }

            // ========================================================
            // 🌟 核心控制台：关闭 XFeat，让 DINOv2 独立扛大旗！
            // ========================================================
            val USE_XFEAT = runtimeConfig.xfeatEnabled// 👈 设为 false，彻底禁用 XFeat

            var bestDetection: DetectionResult? = null
            var bestIdentityScore = -1f
            var bestMatchResult: XFeatLocalIdentityMatcher.MatchResult? = null

            if (USE_XFEAT && tracker.hasAnchorXFeat(currentLockedId)) {
                // --- Stage 3 & 4: XFeat 微观几何终审 ---
                val anchorXFeat = tracker.anchorXFeat(currentLockedId)!!
                val combinedReference = anchorXFeat.combineWithMultiple(temporalMemoryBank.toList())

                for ((candidate, dinov2Score) in rankedCandidates) {
                    val boxPx = boxToPixels(candidate, recoveryFrameWidth, recoveryFrameHeight)

                    val extractStart = System.nanoTime()
                    val candidateXFeat = pipeline.extractXFeat(recoveryBitmap, boxPx, candidate.segmentationMask)
                    recoveryXFeatExtractMs += (System.nanoTime() - extractStart) / 1_000_000f

                    if (candidateXFeat == null) continue

                    val matchStart = System.nanoTime()
                    val matchResult = pipeline.matchXFeat(combinedReference, candidateXFeat, 0)
                    recoveryXFeatMatchMs += (System.nanoTime() - matchStart) / 1_000_000f

                    val identityScore = matchResult?.score ?: 0f
                    if (identityScore > bestIdentityScore) {
                        bestIdentityScore = identityScore
                        bestDetection = candidate
                        bestMatchResult = matchResult
                    }
                }
            } else {
                // --- 🚀 PURE DINOv2 模式：直接保送 DINOv2 状元！ ---
                val (topCandidate, topDinov2Score) = rankedCandidates.first()
                bestDetection = topCandidate
                bestIdentityScore = topDinov2Score
                bestMatchResult = null // 无 XFeat 匹配结果

                Log.i(TAG, "🤖 [PURE DINOv2 MODE] XFeat 被拦截，DINOv2 独立确认真身! 最终得分=${fmt(bestIdentityScore)}")
            }

            // ========================================================
            // 🌟 最终宣判
            // ========================================================
            // 判定及格线：启用 XFeat 则用 XFeat 的门限；纯 DINOv2 模式则用 DINOv2 的 0.70 门限
            val passThreshold = if (USE_XFEAT) tuningConfig.xfeatIdentityThreshold else 0.70f

            var recovered = false
            if (bestIdentityScore >= passThreshold && bestDetection != null) {
                recovered = tracker.recoverWithCandidate(currentLockedId, bestDetection, bestIdentityScore, bestMatchResult)
                if (recovered) {
                    tracker.markRelockPositionPending(currentLockedId)
                    pendingRelockPosition = PendingRelockPosition(
                        targetId = currentLockedId,
                        confirmedBoxNormalized = RectF(
                            bestDetection.cx - bestDetection.w * 0.5f,
                            bestDetection.cy - bestDetection.h * 0.5f,
                            bestDetection.cx + bestDetection.w * 0.5f,
                            bestDetection.cy + bestDetection.h * 0.5f
                        ),
                        label = bestDetection.label
                    )
                    Log.i(TAG, "⚡ ULTRA-FAST RELOCK SUCCESS id=$currentLockedId IdentityScore=${fmt(bestIdentityScore)} >= ${fmt(passThreshold)}")
                    isRecovering = false
                }
            }

            lastRecoveryXFeatExtractMs = recoveryXFeatExtractMs
            lastRecoveryXFeatMatchMs = recoveryXFeatMatchMs
            lastRecoveryUiMs = (SystemClock.elapsedRealtimeNanos() - recoveryFrameStartNs).coerceAtLeast(0L) / 1_000_000f
            publishAndFinish(session, tracker.snapshot(), workerStartNs, prepared = prepared, framePerf = framePerf)
            return true

        } finally {
            // PreparedFrame owns the shared canonical bitmap.
        }
    }

    private fun processLockedFrame(
        session: Long,
        prepared: ObjectDetectionPipeline.PreparedFrame,
        current: TrackedObject,
        currentLockedId: Int,
        frameSeq: Long,
        workerStartNs: Long,
        framePerf: FramePerformanceContext
    ) {
        val trackedBox = RectF(
            current.cx - current.w * 0.5f,
            current.cy - current.h * 0.5f,
            current.cx + current.w * 0.5f,
            current.cy + current.h * 0.5f
        )
        val ostrackResult = pipeline.runOstrack(prepared = prepared, currentBox = trackedBox)
        framePerf.ostrackMs = pipeline.currentOstrackMs()
        lastInferenceMs = framePerf.ostrackMs ?: 0f

        if (ostrackResult != null) {
            tracker.updateTrackedBox(targetId = currentLockedId, box = ostrackResult.box)
            lastRawDetectionCount = 0
            lastAcceptedDetectionCount = 1
        } else {
            val nowLost = tracker.noteOstrackMiss(currentLockedId)
            lastRawDetectionCount = 0
            lastAcceptedDetectionCount = 0
            Log.w(TAG, "OSTRACK MISS id=$currentLockedId -> lost=$nowLost")
        }

        val latest = tracker.lifecycleSnapshot(currentLockedId)
        if (latest == null) {
            publishAndFinish(session, tracker.snapshot(), workerStartNs, prepared = prepared, framePerf = framePerf)
            return
        }

        refreshLockedTemporalMemory(prepared, currentLockedId, latest, ostrackResult, framePerf)
        publishAndFinish(session, tracker.snapshot(), workerStartNs, prepared = prepared, framePerf = framePerf)
    }

    private fun refreshLockedTemporalMemory(
        prepared: ObjectDetectionPipeline.PreparedFrame,
        currentLockedId: Int,
        latest: TrackedObject,
        ostrackResult: OstrackQnnTracker.Result?,
        framePerf: FramePerformanceContext
    ) {
        if (latest.state != TrackState.ACTIVE || ostrackResult == null) return

        val lockedNowMs = SystemClock.elapsedRealtime()

        // 获取当前应该使用的刷新间隔时间 (1秒 / 10秒 / 60秒)
        val currentIntervalMs = getCurrentRefreshIntervalMs()

        if (lockedNowMs >= lockedTemporalRefreshNotBeforeMs &&
            (lastLockedTemporalRefreshMs == 0L || lockedNowMs - lastLockedTemporalRefreshMs >= currentIntervalMs)
        ) {
            lastLockedTemporalRefreshMs = lockedNowMs
            diagLockedTemporalRefreshes++

            val (trackedFrameWidth, trackedFrameHeight) = rotatedFrameSize(prepared.bitmap, prepared.rotationDegrees)
            val currentBox = RectF(
                ostrackResult.box.left * trackedFrameWidth,
                ostrackResult.box.top * trackedFrameHeight,
                ostrackResult.box.right * trackedFrameWidth,
                ostrackResult.box.bottom * trackedFrameHeight
            )

            val nearest = pipeline.findSameClassYoloDetectionForMask(
                prepared = prepared, ostrackBox = currentBox, expectedLabel = latest.label
            )
            framePerf.yoloTotalMs = pipeline.currentPerceptionTotalMs()
            framePerf.yoloInferMs = pipeline.currentInferenceMs()

            if (nearest != null) {
                val lockedSampleBitmap = prepared.canonicalBitmap()

                try {
                    val mask = nearest.detection.segmentationMask
                    val box = RectF(nearest.boxPx)
                    if (mask != null && mask.area > 0) {
                        tracker.updateCurrentMask(currentLockedId, mask)

                        // ====================================================
                        // 🌟 1. 独立提取 DINOv2 (永远执行，不受 XFeat 开关影响)
                        // ====================================================
                        val dinov2StartNs = System.nanoTime()
                        val temporalDinov2 = pipeline.extractDinov2(lockedSampleBitmap, box)
                        lastDinov2Ms = (System.nanoTime() - dinov2StartNs) / 1_000_000f

                        // ====================================================
                        // 🌟 2. 独立提取 XFeat (受 UI 开关控制)
                        // ====================================================
                        val temporalXFeat = if (runtimeConfig.xfeatEnabled) {
                            pipeline.extractXFeat(lockedSampleBitmap, box, mask)
                        } else null

                        val validXFeat = temporalXFeat != null && temporalXFeat.count >= XFeatLocalIdentityMatcher.MIN_GEOMETRY_POINTS
                        val trustedXFeat = if (validXFeat) temporalXFeat?.copyFeatures() else null

                        // ====================================================
                        // 🌟 3. 初始化判断：以 DINOv2 为基准！
                        // ====================================================
                        if (!tracker.hasAnchorDinov2(currentLockedId)) {
                            // 初始化 Anchor
                            tracker.captureAnchor(
                                targetId = currentLockedId,
                                sourceMask = mask,
                                xfeat = trustedXFeat,
                                dinov2 = temporalDinov2
                            )
                            if (trustedXFeat != null) {
                                temporalMemoryBank.addLast(trustedXFeat)
                            }
                            totalPosesAdded = 1
                            diagLockedTemporalSuccess++
                            Log.i(TAG, "[LOCKED_TEMPORAL_REFRESH] INIT Anchor Ready (XFeat Enabled: ${runtimeConfig.xfeatEnabled})")
                        } else {
                            // ====================================================
                            // 🌟 4. DINOv2 灵魂大闸防漂移 & 精英入库
                            // ====================================================
                            val anchorDinov2 = tracker.anchorDinov2(currentLockedId)
                            if (temporalDinov2 != null && anchorDinov2 != null) {
                                var dot = 0f; var n1 = 0f; var n2 = 0f
                                for (i in temporalDinov2.indices) {
                                    dot += temporalDinov2[i] * anchorDinov2[i]
                                    n1 += temporalDinov2[i] * temporalDinov2[i]
                                    n2 += anchorDinov2[i] * anchorDinov2[i]
                                }
                                val authenticityScore = if (n1 > 0f && n2 > 0f) dot / (kotlin.math.sqrt(n1) * kotlin.math.sqrt(n2)) else 0f

                                // 🚨 发现 OSTrack 发生灾难性漂移，立刻切断入库并强制找回！
                                if (authenticityScore < 0.70f) {
                                    Log.w(TAG, "🚨 [POISON PREVENTION] 拦截污染! 灵魂相似度=${fmt(authenticityScore)} < 0.70, 拒绝假目标入库, 强制触发重锁!")
                                    tracker.markRelockPositionPending(currentLockedId)
                                    return // 阻断执行，绝不更新颜色直方图
                                }

                                // 智能环境收录
                                if (authenticityScore in 0.70f..0.92f) {
                                    tracker.addTemporalDinov2(currentLockedId, temporalDinov2)
                                    Log.i(TAG, "🧿 [DINOV2_ELITE_BANK] 目标外观异变, 存入精英库! 环境相似度=${fmt(authenticityScore)}")
                                }
                            }

                            // ====================================================
                            // 🌟 5. XFeat 局部新颖度查重 (仅在 XFeat 开启且有效时执行)
                            // ====================================================
                            if (runtimeConfig.xfeatEnabled && trustedXFeat != null) {
                                val anchorXFeat = tracker.anchorXFeat(currentLockedId)
                                var isNovelPose = true
                                var redundancyScore = 0f

                                if (anchorXFeat != null) {
                                    // 构造全息参考系：Anchor + 库里历史姿态
                                    val globalMemory = anchorXFeat.combineWithMultiple(temporalMemoryBank.toList())
                                    val noveltyMatch = pipeline.matchXFeat(globalMemory, trustedXFeat, 0)
                                    redundancyScore = noveltyMatch?.score ?: 0f

                                    // 如果和记忆库里任何一个姿态相似度 > 0.65，说明是老姿态
                                    if (redundancyScore > 0.65f) {
                                        isNovelPose = false
                                    }
                                }

                                // 确认是新姿态才入库
                                if (isNovelPose) {
                                    if (temporalMemoryBank.size >= MAX_MEMORY_BANK_SIZE) {
                                        temporalMemoryBank.removeFirst()
                                    }
                                    temporalMemoryBank.addLast(trustedXFeat)
                                    totalPosesAdded++
                                    diagLockedTemporalSuccess++
                                    Log.i(TAG, "[LOCKED_TEMPORAL_REFRESH] 🟢 POSE_ADDED RedundancyScore=${fmt(redundancyScore)} (New Angle!) BankSize=${temporalMemoryBank.size} NextInterval=${currentIntervalMs/1000}s")
                                } else {
                                    Log.i(TAG, "[LOCKED_TEMPORAL_REFRESH] 🔴 POSE_REJECTED RedundancyScore=${fmt(redundancyScore)} >= 0.65 (Ignored)")
                                }
                            } else if (!runtimeConfig.xfeatEnabled) {
                                // 纯 DINOv2 模式下，只要没触发漂移警报，就算刷新成功
                                diagLockedTemporalSuccess++
                            }
                        }

                        // 无论特征是否入库，几何体型和颜色直方图永远保持最新！(用于 Cheap Ranking)
                        latestTemporalTemplateBox = RectF(box)
                        latestTemporalColor = extractSpatialHistogram(lockedSampleBitmap, box, mask)

                    } else {
                        diagLockedTemporalNoMask++
                    }
                } finally {
                    // PreparedFrame owns the shared canonical bitmap.
                }
            } else {
                diagLockedTemporalNoMask++
            }
        }

        val statsNowMs = SystemClock.elapsedRealtime()
        if (statsNowMs - diagLastStatsMs >= 5_000L) {
            diagLastStatsMs = statsNowMs
            Log.i(TAG, "[STATS] id=$currentLockedId temporalRefresh=$diagLockedTemporalRefreshes temporalSuccess=$diagLockedTemporalSuccess temporalNoMask=$diagLockedTemporalNoMask")
        }
    }

    private data class PendingRelockPosition(
        val targetId: Int,
        val confirmedBoxNormalized: RectF,
        val label: String
    )

    private fun isValidRecoveryBox(box: RectF, width: Int, height: Int): Boolean {
        if (width <= 0 || height <= 0 || box.width() < 2f || box.height() < 2f) return false
        return box.right > 0f && box.bottom > 0f && box.left < width.toFloat() && box.top < height.toFloat()
    }

    private fun resetLockedTemporalRefreshState() {
        lastLockedTemporalRefreshMs = 0L
        lockedTemporalRefreshNotBeforeMs = 0L
    }

    private fun clearLatestTemporalTemplate() {
        temporalMemoryBank.clear() // 清空记忆池
        latestTemporalTemplateBox = null
        latestTemporalColor = null
        totalPosesAdded = 0 // 👇 🌟 新增：重置计步器
    }

    /**
     * 🌟 四段式智能变速箱 (Dynamic Refresh Interval)
     * 换挡逻辑：1s -> 10s -> 45s -> 90s
     */
    private fun getCurrentRefreshIntervalMs(): Long {
        // 【1档】极速建库期：1 秒 / 次
        // 目标刚被锁定，贪婪地吃满所有 50 个槽位，确保 360 度视角全覆盖
        if (temporalMemoryBank.size < MAX_MEMORY_BANK_SIZE) {
            return 1_000L
        }

        // 【2档】稳健洗牌期：10 秒 / 次
        // 槽位已满。此时把最初录入的第一批记忆（因为刚锁定，可能包含不稳定的视角）轮换清洗一遍
        // 覆盖范围：第 50 ~ 99 个存入的姿态
        if (totalPosesAdded < MAX_MEMORY_BANK_SIZE * 2) {
            return 10_000L
        }

        // 【3档】中频过渡期：45 秒 / 次
        // 记忆库已经很稳固，进入中长期跟踪，再进行第二轮深度优胜劣汰
        // 覆盖范围：第 100 ~ 149 个存入的姿态
        if (totalPosesAdded < MAX_MEMORY_BANK_SIZE * 3) {
            return 45_000L
        }

        // 【4档】极限潜航期：90 秒 / 次
        // 经历了两轮完整的“大洗血”，库里存的绝对是最高清、最有代表性的极品切片
        // 此时为了最大化节省电量和算力，进入 1.5 分钟才看一眼的极限待机模式
        return 90_000L
    }

    private fun publishAndFinish(
        session: Long,
        targets: List<TrackedObject>,
        workerStartNs: Long,
        prepared: ObjectDetectionPipeline.PreparedFrame,
        framePerf: FramePerformanceContext
    ) {
        publishTargetsIfCurrent(session, targets)
        processedFrames++

        val endNs = SystemClock.elapsedRealtimeNanos()
        val frameInferenceMs = framePerf.yoloInferMs ?: framePerf.ostrackMs ?: 0f
        lastInferenceMs = frameInferenceMs
        lastEndToEndMs = if (framePerf.token.cameraAvailableNs > 0L) {
            (endNs - framePerf.token.cameraAvailableNs).coerceAtLeast(0L) / 1_000_000f
        } else {
            (endNs - workerStartNs).coerceAtLeast(0L) / 1_000_000f
        }

        profiler.finish(
            profiler.buildTimings(
                token = framePerf.token,
                workerStartNs = workerStartNs,
                prepareStartNs = prepared.prepareStartNs,
                prepareEndNs = prepared.prepareEndNs,
                // TRACK will be refined separately; E2E already uses the real frame lifecycle.
                trackerStartNs = endNs,
                imageBuildMs = 0f,
                inferenceMs = frameInferenceMs,
                parseMs = pipeline.currentParseMs(),
                yoloTotalMs = framePerf.yoloTotalMs,
                yoloInferMs = framePerf.yoloInferMs,
                ostrackMs = framePerf.ostrackMs
            ),
            trackerEndNs = endNs
        )
    }

    private fun publishTargetsIfCurrent(session: Long, targets: List<TrackedObject>) {
        if (!running.get() || generation.get() != session) return
        mainHandler.post {
            if (running.get() && generation.get() == session) {
                onTargetsUpdated(targets)
            }
        }
    }

    private fun publishTargets() {
        mainHandler.post { onTargetsUpdated(tracker.snapshot()) }
    }

    private fun resetCounters() {
        frameCounter = 0L
        availableFrames = 0L
        processedFrames = 0L
        droppedFrames = 0L
        firstStatsNs = 0L
        lastInferenceMs = 0f
        lastEndToEndMs = 0f
        lastRawDetectionCount = 0
        lastAcceptedDetectionCount = 0
        lastRecoveryUiMs = 0f
        lastRecoveryUiCandidates = 0
        lastRecoveryXFeatExtractMs = 0f
        lastRecoveryXFeatMatchMs = 0f
        diagLockedTemporalRefreshes = 0L
        diagLockedTemporalSuccess = 0L
        diagLockedTemporalNoMask = 0L
        diagLastStatsMs = SystemClock.elapsedRealtime()

        isRecovering = false
        recoveryStartMs = 0L
        profiler.reset()
    }

    private fun recyclePreparedBitmap() {
        lastPreparedFrame?.release()
        lastPreparedFrame = null
    }

    private fun rotatedFrameSize(bitmap: android.graphics.Bitmap, rotationDegrees: Int): Pair<Int, Int> {
        return if (rotationDegrees % 180 == 0) bitmap.width to bitmap.height else bitmap.height to bitmap.width
    }

    private fun toRectPixels(t: TrackedObject, width: Int, height: Int): RectF =
        RectF(t.cx * width - (t.w * width) * 0.5f, t.cy * height - (t.h * height) * 0.5f, t.cx * width + (t.w * width) * 0.5f, t.cy * height + (t.h * height) * 0.5f)

    private fun boxToPixels(d: DetectionResult, width: Int, height: Int): RectF =
        RectF((d.cx - d.w / 2f) * width, (d.cy - d.h / 2f) * height, (d.cx + d.w / 2f) * width, (d.cy + d.h / 2f) * height)

    private fun fmt(v: Float): String = "%.3f".format(v)

    /**
     * 极速 3D-RGB 空间色彩直方图 (128 维：上半身64 + 下半身64)
     * 完美识别黑白灰与肤色，专治“同款制服”陷阱！
     */
    /**
     * 极速 3D-RGB 空间色彩直方图 (192 维：头部64 + 上半身64 + 下半身64)
     * 纵向三段式金字塔切分 (15% 头部/发色, 35% 上身, 50% 下身)
     * 彻底消除水平旋转不一致问题，精准识别头饰与发色！
     */
    private fun extractSpatialHistogram(
        bitmap: android.graphics.Bitmap,
        boxPx: RectF,
        mask: SegmentationMask?
    ): FloatArray {
        val left = boxPx.left.toInt().coerceIn(0, bitmap.width - 1)
        val top = boxPx.top.toInt().coerceIn(0, bitmap.height - 1)
        val right = boxPx.right.toInt().coerceIn(left + 1, bitmap.width)
        val bottom = boxPx.bottom.toInt().coerceIn(top + 1, bitmap.height)

        val width = right - left
        val height = bottom - top
        if (width <= 0 || height <= 0) return FloatArray(192)

        // 1. 批量读取像素，避免频繁 JNI 调用
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, left, top, width, height)

        val hist = FloatArray(192) // 0~63: Head, 64~127: Upper, 128~191: Lower
        var countHead = 0
        var countUpper = 0
        var countLower = 0

        // 2. 切分界限：15% 头部，35% 上身（累计 50%），其余 50% 下身
        val headSplitY = (height * 0.15f).toInt()
        val upperSplitY = (height * 0.50f).toInt()

        // 3. 动态步长
        val stepX = maxOf(1, width / 12)
        val stepY = maxOf(1, height / 12)

        for (y in 0 until height step stepY) {
            val rowOffset = y * width

            // 判断当前 Y 所属的区间偏移量
            val segmentOffset = when {
                y < headSplitY -> 0
                y < upperSplitY -> 64
                else -> 128
            }

            for (x in 0 until width step stepX) {
                // Segmentation Mask 校验
                if (mask != null && mask.width > 0) {
                    val absX = left + x
                    val absY = top + y
                    val mx = ((absX - boxPx.left) / boxPx.width() * mask.width).toInt().coerceIn(0, mask.width - 1)
                    val my = ((absY - boxPx.top) / boxPx.height() * mask.height).toInt().coerceIn(0, mask.height - 1)
                    if (!mask.contains(mx, my)) continue
                }

                val pixel = pixels[rowOffset + x]

                // RGB 位运算量化 (0~3)
                val rBin = ((pixel shr 16) and 0xFF) shr 6
                val gBin = ((pixel shr 8) and 0xFF) shr 6
                val bBin = (pixel and 0xFF) shr 6

                val binIndex = (rBin shl 4) or (gBin shl 2) or bBin

                // 累加对应区间的 bin
                hist[segmentOffset + binIndex] += 1f

                when (segmentOffset) {
                    0 -> countHead++
                    64 -> countUpper++
                    128 -> countLower++
                }
            }
        }

        // 4. L1 独立归一化（保证每段特征权重均衡）
        if (countHead > 0) {
            val invHead = 1.0f / countHead
            for (i in 0 until 64) hist[i] *= invHead
        }
        if (countUpper > 0) {
            val invUpper = 1.0f / countUpper
            for (i in 64 until 128) hist[i] *= invUpper
        }
        if (countLower > 0) {
            val invLower = 1.0f / countLower
            for (i in 128 until 192) hist[i] *= invLower
        }

        return hist
    }

    /**
     * 智能外观评分漏斗：基于直方图交叉 (Histogram Intersection) + 几何形体
     */
    private fun calculateCheapScoreHistogram(
        yoloConfidence: Float,
        candHist: FloatArray,
        targetHist: FloatArray?,
        candBox: RectF,
        targetBox: RectF?
    ): Float {
        if (targetHist == null || targetBox == null) return yoloConfidence

        var headIntersection = 0f
        var upperIntersection = 0f
        var lowerIntersection = 0f

        // 1. 遍历 192 维：单次循环同时计算 3 段交集，最大化利用 CPU 缓存
        for (i in 0 until 64) {
            headIntersection += minOf(candHist[i], targetHist[i])
            upperIntersection += minOf(candHist[i + 64], targetHist[i + 64])
            lowerIntersection += minOf(candHist[i + 128], targetHist[i + 128])
        }

        // 三段相似度取等权平均 (均值在 [0.0, 1.0])
        val colorSimilarity = (headIntersection + upperIntersection + lowerIntersection) / 3.0f

        // 2. 几何相似度保持不变
        val candArea = candBox.width() * candBox.height()
        val targetArea = targetBox.width() * targetBox.height()
        val areaSim = minOf(candArea, targetArea) / maxOf(candArea, targetArea).coerceAtLeast(1f)
        val candAspect = candBox.width() / candBox.height().coerceAtLeast(1f)
        val targetAspect = targetBox.width() / targetBox.height().coerceAtLeast(1f)
        val aspectSim = minOf(candAspect, targetAspect) / maxOf(candAspect, targetAspect).coerceAtLeast(0.01f)
        val geometrySimilarity = (areaSim + aspectSim) * 0.5f

        // 3. 终极分数融合
        return yoloConfidence * 0.02f + colorSimilarity * 0.80f + geometrySimilarity * 0.18f
    }
}