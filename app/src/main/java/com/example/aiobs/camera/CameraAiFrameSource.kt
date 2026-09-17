package com.example.aiobs.camera

import android.graphics.ImageFormat
import android.media.Image
import android.os.SystemClock
import android.util.Log
import com.example.aiobs.ai.AiFrame
import com.example.aiobs.ai.VisualRadarController
import com.example.aiobs.ai.Yuv420FrameSnapshot
import com.example.aiobs.core.AppState
import com.pedro.encoder.input.video.Camera2ApiManager
import com.pedro.library.srt.SrtCamera2
import java.nio.ByteBuffer
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/**
 * Camera AI frame source.
 *
 * ================================================================
 * THREE-LAYER CAMERA ARCHITECTURE
 * ================================================================
 *
 * Camera2 Preview
 *      │
 *      ├── AI ImageReader
 *      │       │
 *      │       └── permanent attachment
 *      │
 *      └── Video Encoder / SRT
 *
 * IMPORTANT:
 *
 * 1. ImageReader 只 attach 一次。
 * 2. start() 只开启 AI callback。
 * 3. stop() 只暂停 AI callback。
 * 4. stop() 绝不 removeImageListener()。
 * 5. release() 才真正 removeImageListener()。
 *
 * Camera2 Image:
 *
 *     Image
 *       ↓
 *     copy Y/U/V
 *       ↓
 *     Image.close()
 *       ↓
 *     Yuv420FrameSnapshot
 */
class CameraAiFrameSource(
    private val context: android.content.Context,
    private val camera: SrtCamera2,
    private val state: AppState
) : VisualRadarController.FrameSource {

    private val running =
        AtomicBoolean(false)

    private val prepared =
        AtomicBoolean(false)

    @Volatile
    private var callback:
            ((AiFrame, Long) -> Unit)? = null

    @Volatile
    private var tryAcquireFrame: (() -> Long?)? = null

    @Volatile
    private var releaseFrame: ((Long) -> Unit)? = null

    private val frameCounter =
        AtomicLong(0L)

    // Pure diagnostics only. These counters do not participate in frame admission
    // or any camera/AI control flow.
    private val cameraReceivedCount =
        AtomicLong(0L)

    private val cameraDroppedCount =
        AtomicLong(0L)

    private val aiSubmittedCount =
        AtomicLong(0L)

    private val aiCompletedCount =
        AtomicLong(0L)

    data class StatsSnapshot(
        val receivedFps: Float,
        val droppedFps: Float,
        val submittedFps: Float,
        val completedFps: Float
    )

    @Volatile
    private var lastStatsSnapshot = StatsSnapshot(0f, 0f, 0f, 0f)

    @Volatile
    private var statsWindowStartNs =
        SystemClock.elapsedRealtimeNanos()

    private val lastStatsReceived =
        AtomicLong(0L)

    private val lastStatsDropped =
        AtomicLong(0L)

    private val lastStatsSubmitted =
        AtomicLong(0L)

    private val lastStatsCompleted =
        AtomicLong(0L)

    private var imageWidth =
        0

    private var imageHeight =
        0

    /**
     * V43D: reuse Y/U/V snapshot storage. Camera Image cannot cross threads,
     * so we still copy the planes before Image.close(); however the plane
     * arrays themselves are pooled to remove per-frame allocations/GC.
     * The controller owns a single in-flight inference slot, therefore at
     * most one lease is normally active. Two slots keep the implementation
     * robust against callback/release timing races.
     */
    private val yuvBufferPool =
        ArrayDeque<YuvBufferLease>(YUV_POOL_SIZE)

    private val activeYuvLease =
        AtomicReference<YuvBufferLease?>(null)

    private val yuvPoolLock =
        Any()

    private val listener =
        object : Camera2ApiManager.ImageCallback {

            override fun onImageAvailable(
                image: Image
            ) {
                // True frame-lifecycle start: as soon as ImageReader invokes us.
                val callbackStartNs = SystemClock.elapsedRealtimeNanos()

                cameraReceivedCount.incrementAndGet()

                val number =
                    frameCounter.incrementAndGet()

                /*
                 * ImageReader 永久存在。
                 *
                 * 即使 YOLO OFF，
                 * RootEncoder 仍然会 acquire image。
                 *
                 * 所以这里必须立即 close。
                 */
                if (!running.get()) {

                    safeClose(
                        image
                    )

                    return
                }

                val target =
                    callback

                if (target == null) {

                    safeClose(
                        image
                    )

                    return
                }

                /*
                 * Admission MUST happen before any YUV byte copy.
                 *
                 * VisualRadarController owns a single in-flight AI slot.
                 * If that slot is busy, this image is stale for inference and
                 * must be dropped immediately without copying Y/U/V. This keeps
                 * the camera path truly latest-frame / zero-queue.
                 */
                val acquire =
                    tryAcquireFrame

                val slotToken =
                    acquire?.invoke()

                if (slotToken == null) {
                    cameraDroppedCount.incrementAndGet()
                    safeClose(image)
                    maybeLogCameraStats()
                    return
                }

                Log.i(
                    TAG,
                    "[AI_FRAME_ADMITTED] cameraN=$number token=$slotToken " +
                            "thread=${Thread.currentThread().name}"
                )

                val release =
                    releaseFrame

                val startNs = callbackStartNs

                val snapshotCopyStartNs =
                    SystemClock.elapsedRealtimeNanos()

                val snapshot =
                    try {

                        copyImage(
                            image
                        )

                    } catch (t: Throwable) {

                        Log.e(
                            TAG,
                            "YUV snapshot copy failed " +
                                    "n=$number",
                            t
                        )

                        null

                    } finally {

                        /*
                         * 极其重要：
                         *
                         * 不允许 Image 跨线程。
                         */
                        safeClose(
                            image
                        )
                    }

                if (snapshot == null) {
                    Log.w(
                        TAG,
                        "[AI_FRAME_ABORT] token=$slotToken cameraN=$number reason=SNAPSHOT_NULL"
                    )
                    release?.invoke(slotToken)
                    return
                }

                val snapshotCopyMs =
                    (
                            SystemClock.elapsedRealtimeNanos() -
                                    snapshotCopyStartNs
                            ) / 1_000_000f

                maybeLogCameraStats()

                if (
                    number == 1L ||
                    number % DEBUG_FRAME_LOG_INTERVAL == 0L
                ) {

                    Log.i(
                        TAG,
                        "YUV FRAME RECEIVED " +
                                "n=$number " +
                                "size=${snapshot.width}x${snapshot.height} " +
                                "copy=${"%.2f".format(snapshotCopyMs)}ms " +
                                "Y=${snapshot.yData.size}B " +
                                "U=${snapshot.uData.size}B " +
                                "V=${snapshot.vData.size}B " +
                                "pool=REUSE " +
                                "running=${running.get()}"
                    )
                }

                try {

                    val callbackNs = SystemClock.elapsedRealtimeNanos()
                    Log.i(
                        TAG,
                        "[AI_FRAME_HANDOFF] token=$slotToken cameraN=$number " +
                                "copyMs=${"%.2f".format(snapshotCopyMs)} thread=${Thread.currentThread().name}"
                    )
                    target.invoke(
                        AiFrame.CameraImage(
                            frame =
                                snapshot,
                            availableStartNs =
                                startNs,
                            availableEndNs =
                                callbackNs
                        ),
                        slotToken
                    )

                    aiSubmittedCount.incrementAndGet()
                    Log.i(
                        TAG,
                        "[AI_FRAME_SUBMITTED] token=$slotToken cameraN=$number " +
                                "thread=${Thread.currentThread().name}"
                    )
                    maybeLogCameraStats()

                } catch (t: Throwable) {

                    /*
                     * Admission succeeded, but the snapshot could not be
                     * handed to VisualRadarController. Return the slot here;
                     * otherwise the next camera frame would be rejected forever.
                     */
                    Log.e(
                        TAG,
                        "[AI_FRAME_ABORT] token=$slotToken cameraN=$number reason=CALLBACK_EXCEPTION",
                        t
                    )
                    release?.invoke(slotToken)


                    maybeLogCameraStats()
                }
            }
        }

    // =================================================================
    // Permanent preparation
    // =================================================================

    /**
     * 只允许在 Camera Preview 启动之前调用。
     *
     * 一次 attach。
     */
    fun prepare(): Boolean {

        if (prepared.get()) {

            Log.d(
                TAG,
                "prepare(): already attached"
            )

            return true
        }

        /*
         * 如果 Preview 已经启动，
         * 此时再 addImageListener() 会改变 Camera2 session，
         * 正是我们现在要避免的。
         */
        if (camera.isOnPreview) {

            Log.e(
                TAG,
                "prepare() refused: Camera Preview is already running"
            )

            return false
        }

        return try {

            var width =
                camera.getStreamWidth()

            var height =
                camera.getStreamHeight()

            if (
                width <= 0 ||
                height <= 0
            ) {

                width =
                    1920

                height =
                    1080
            }

            /*
             * 使用 Preview / Encoder 的逻辑尺寸。
             *
             * RootEncoder 会根据 Camera2 实际支持能力
             * 选择最终 ImageReader 尺寸。
             */
            imageWidth =
                width

            imageHeight =
                height

            camera.addImageListener(
                width,
                height,
                ImageFormat.YUV_420_888,
                MAX_IMAGES,
                false,
                listener
            )

            prepared.set(
                true
            )

            Log.i(
                TAG,
                "AI ImageReader PERMANENTLY ATTACHED " +
                        "size=${width}x${height} " +
                        "format=YUV_420_888 " +
                        "maxImages=$MAX_IMAGES"
            )

            true

        } catch (t: Throwable) {

            prepared.set(
                false
            )

            Log.e(
                TAG,
                "Permanent AI ImageReader attach failed",
                t
            )

            false
        }
    }

    // =================================================================
    // FrameSource
    // =================================================================

    override fun start(
        onFrame: (AiFrame, Long) -> Unit,
        tryAcquireFrame: () -> Long?,
        releaseFrame: (Long) -> Unit
    ): Boolean {

        if (!prepared.get()) {

            Log.e(
                TAG,
                "start() refused: ImageReader was not prepared"
            )

            return false
        }

        callback =
            onFrame

        this.tryAcquireFrame =
            tryAcquireFrame

        this.releaseFrame =
            { token ->
                try {
                    releaseFrame(token)
                } finally {
                    releaseActiveYuvLease()
                    aiCompletedCount.incrementAndGet()
                    maybeLogCameraStats()
                    Log.d(
                        TAG,
                        "[AI_YUV_LEASE_RELEASE] token=$token reason=EARLY_RELEASE_CALLBACK"
                    )
                }
            }

        statsWindowStartNs =
            SystemClock.elapsedRealtimeNanos()

        lastStatsReceived.set(
            cameraReceivedCount.get()
        )
        lastStatsDropped.set(
            cameraDroppedCount.get()
        )
        lastStatsSubmitted.set(
            aiSubmittedCount.get()
        )
        lastStatsCompleted.set(
            aiCompletedCount.get()
        )

        running.set(
            true
        )

        Log.i(
            TAG,
            "CAMERA_DIAGNOSTIC_STARTED"
        )

        Log.i(
            TAG,
            "AI callback ENABLED; " +
                    "Camera2/ImageReader lifecycle unchanged"
        )

        return true
    }

    /**
     * V49: successful worker completion path.
     * The controller already released the inference slot separately; this hook
     * only returns the YUV backing arrays to the camera pool.
     */
    override fun onFrameConsumed(slotToken: Long) {
        releaseActiveYuvLease()
        maybeLogCameraStats()
        Log.d(
            TAG,
            "[AI_YUV_LEASE_RELEASE] token=$slotToken reason=FRAME_CONSUMED"
        )
    }

    override fun stop() {

        /*
         * 关键：
         *
         * 不 removeImageListener()
         * 不改变 Camera2 CaptureSession
         * 不关闭 ImageReader
         *
         * 只暂停 callback。
         */
        running.set(
            false
        )

        callback =
            null

        tryAcquireFrame =
            null

        releaseActiveYuvLease()

        releaseFrame =
            null

        Log.i(
            TAG,
            "AI callback DISABLED; " +
                    "ImageReader remains permanently attached"
        )
    }

    /**
     * 真正销毁 Camera AI source。
     *
     * 只在 Activity 最终 destroy 时调用。
     */
    fun release() {

        running.set(
            false
        )

        callback =
            null

        tryAcquireFrame =
            null

        releaseActiveYuvLease()

        releaseFrame =
            null

        if (
            prepared.get()
        ) {

            try {

                camera.removeImageListener()

                Log.i(
                    TAG,
                    "AI ImageReader RELEASED"

                )

            } catch (t: Throwable) {

                Log.w(
                    TAG,
                    "removeImageListener failed during release",
                    t
                )
            }
        }

        prepared.set(
            false
        )
    }

    // =================================================================
    // YUV snapshot
    // =================================================================

    private fun copyImage(
        image: Image
    ): Yuv420FrameSnapshot {

        val planes =
            image.planes

        require(
            planes.size >= 3
        ) {
            "YUV_420_888 requires 3 planes"
        }

        val y = planes[0]
        val u = planes[1]
        val v = planes[2]

        val lease = acquireYuvLease(
            y.buffer.remaining(),
            u.buffer.remaining(),
            v.buffer.remaining()
        )

        try {
            copyBufferInto(
                y.buffer,
                lease.yData
            )
            copyBufferInto(
                u.buffer,
                lease.uData
            )
            copyBufferInto(
                v.buffer,
                lease.vData
            )

            if (!activeYuvLease.compareAndSet(null, lease)) {
                releaseYuvLease(lease)
                Log.e(TAG, "[AI_YUV_LEASE_COLLISION] activeLeaseAlreadyPresent=true")
                error("YUV buffer lease collision")
            }

            Log.d(TAG, "[AI_YUV_LEASE_ACQUIRED] y=${lease.yData.size} u=${lease.uData.size} v=${lease.vData.size}")

            return Yuv420FrameSnapshot(
                width = image.width,
                height = image.height,
                yData = lease.yData,
                uData = lease.uData,
                vData = lease.vData,
                yRowStride = y.rowStride,
                yPixelStride = y.pixelStride,
                uRowStride = u.rowStride,
                uPixelStride = u.pixelStride,
                vRowStride = v.rowStride,
                vPixelStride = v.pixelStride
            )
        } catch (t: Throwable) {
            releaseYuvLease(lease)
            throw t
        }
    }

    private fun acquireYuvLease(
        ySize: Int,
        uSize: Int,
        vSize: Int
    ): YuvBufferLease {
        synchronized(yuvPoolLock) {
            val lease =
                if (yuvBufferPool.isNotEmpty()) {
                    yuvBufferPool.removeFirst()
                } else {
                    YuvBufferLease(
                        ByteArray(ySize),
                        ByteArray(uSize),
                        ByteArray(vSize)
                    )
                }

            lease.ensureCapacity(
                ySize,
                uSize,
                vSize
            )

            return lease
        }
    }

    private fun releaseActiveYuvLease() {
        activeYuvLease.getAndSet(null)?.let {
            releaseYuvLease(it)
        }
    }

    private fun releaseYuvLease(
        lease: YuvBufferLease
    ) {
        synchronized(yuvPoolLock) {
            if (yuvBufferPool.size < YUV_POOL_SIZE) {
                yuvBufferPool.addLast(lease)
            }
        }
    }

    private fun copyBufferInto(
        source: ByteBuffer,
        destination: ByteArray
    ) {
        val duplicate = source.duplicate()
        duplicate.get(
            destination,
            0,
            duplicate.remaining()
        )
    }

    /**
     * Diagnostic-only rate reporting.
     *
     * Every report covers the real callback stream since the previous report.
     * This does not participate in admission, buffering, or frame processing.
     */
    fun statsSnapshot(): StatsSnapshot = lastStatsSnapshot

    private fun maybeLogCameraStats() {

        val nowNs =
            SystemClock.elapsedRealtimeNanos()

        val startNs =
            statsWindowStartNs

        val elapsedNs =
            nowNs - startNs

        // Report approximately once per second.
        if (elapsedNs < 1_000_000_000L) {
            return
        }

        synchronized(this) {

            val nowNs2 =
                SystemClock.elapsedRealtimeNanos()

            val elapsedNs2 =
                nowNs2 - statsWindowStartNs

            if (elapsedNs2 < 1_000_000_000L) {
                return
            }

            val received =
                cameraReceivedCount.get()

            val dropped =
                cameraDroppedCount.get()

            val submitted =
                aiSubmittedCount.get()

            val completed =
                aiCompletedCount.get()

            val windowReceived =
                received - lastStatsReceived.getAndSet(received)

            val windowDropped =
                dropped - lastStatsDropped.getAndSet(dropped)

            val windowSubmitted =
                submitted - lastStatsSubmitted.getAndSet(submitted)

            val windowCompleted =
                completed - lastStatsCompleted.getAndSet(completed)

            val seconds =
                elapsedNs2 / 1_000_000_000.0

            lastStatsSnapshot = StatsSnapshot(
                receivedFps = (windowReceived / seconds).toFloat(),
                droppedFps = (windowDropped / seconds).toFloat(),
                submittedFps = (windowSubmitted / seconds).toFloat(),
                completedFps = (windowCompleted / seconds).toFloat()
            )

            statsWindowStartNs =
                nowNs2

            Log.i(
                TAG,
                "CAMERA_STATS " +
                        "CAMERA_RECEIVED=$received " +
                        "CAMERA_DROPPED=$dropped " +
                        "AI_SUBMITTED=$submitted " +
                        "AI_COMPLETED=$completed " +
                        "windowReceived=${"%.2f".format(windowReceived / seconds)}fps " +
                        "windowDropped=${"%.2f".format(windowDropped / seconds)}fps " +
                        "windowSubmitted=${"%.2f".format(windowSubmitted / seconds)}fps " +
                        "windowCompleted=${"%.2f".format(windowCompleted / seconds)}fps"
            )
        }
    }

    private fun safeClose(
        image: Image
    ) {

        try {
            image.close()
        } catch (_: Throwable) {
        }
    }

    private class YuvBufferLease(
        var yData: ByteArray,
        var uData: ByteArray,
        var vData: ByteArray
    ) {
        fun ensureCapacity(
            ySize: Int,
            uSize: Int,
            vSize: Int
        ) {
            if (yData.size < ySize) yData = ByteArray(ySize)
            if (uData.size < uSize) uData = ByteArray(uSize)
            if (vData.size < vSize) vData = ByteArray(vSize)
        }
    }

    companion object {

        private const val TAG =
            "CameraAiFrameSourceV46"

        private const val MAX_IMAGES =
            2

        private const val YUV_POOL_SIZE =
            2

        private const val DEBUG_FRAME_LOG_INTERVAL =
            30L
    }
}