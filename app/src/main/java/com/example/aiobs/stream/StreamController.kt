package com.example.aiobs.stream

import android.os.SystemClock
import android.util.Log
import com.example.aiobs.core.AppState
import com.pedro.common.ConnectChecker
import com.pedro.common.VideoCodec
import com.pedro.library.srt.SrtCamera2
import com.pedro.library.view.OpenGlView

class StreamController(
    private val surfaceView: OpenGlView,
    private val state: AppState,
    private val callback: Listener,
    /** AI frame rotation provider. Kept separate from the UI/preview rotation in locked mode. */
    private val rotationProvider: () -> Int = { if (state.portrait) 90 else 0 },
    /** Rotation applied to the visible camera preview/encoder surface. */
    private val previewRotationProvider: () -> Int = rotationProvider
) : ConnectChecker {

    interface Listener {
        fun onStreamStateChanged(streaming: Boolean, reconnecting: Boolean)
        fun onBitrateChanged(bitrate: Long)
        fun onStreamError(reason: String)
    }

    private val camera = SrtCamera2(surfaceView, this)

    // 🌟 新增：暴露实时推流帧率给外部
    var currentStreamFps: Int = 0
        private set

    // 🌟 新增：在这里监听底层推流库的帧率变化
    init {
        camera.setFpsListener { fps ->
            currentStreamFps = fps
        }
    }

    private var reconnectGeneration = 0L
    private var reconnectRunnable: Runnable? = null

    var totalStreamedBytes: Long = 0L
        private set
    var currentRealBitrate: Long = 0L
        private set
    private var lastBitrateTimestampMs = 0L

    fun camera(): SrtCamera2 = camera
    fun isStreaming(): Boolean = camera.isStreaming


    fun preparePreview(): Boolean {
        // In locked orientation the UI/preview stays fixed while AI follows the
        // phone's physical posture. Therefore Camera preview rotation and AI frame
        // rotation are intentionally separate here.
        val rotation = ((previewRotationProvider() % 360) + 360) % 360
        return try {
            camera.setVideoCodec(VideoCodec.H265)
            val ok = camera.prepareVideo(
                state.videoWidth, state.videoHeight, state.videoFps,
                state.videoBitrate, 2, rotation
            )
            if (!ok) Log.e(TAG, "prepareVideo failed inside RootEncoder")
            ok
        } catch (t: Throwable) {
            Log.e(TAG, "prepareVideo exception", t)
            false
        }
    }

    fun start(url: String): Boolean {
        cancelReconnect()
        totalStreamedBytes = 0L
        currentRealBitrate = 0L
        lastBitrateTimestampMs = 0L

        if (!preparePreview()) {
            callback.onStreamError("视频编码器准备失败")
            return false
        }

        if (!camera.prepareAudio(128 * 1024, 44100, true)) {
            callback.onStreamError("音频准备失败")
            return false
        }

        return try {
            camera.startStream(url)
            callback.onStreamStateChanged(true, false)
            true
        } catch (e: Exception) {
            Log.e(TAG, "startStream failed", e)
            callback.onStreamError(e.message ?: "推流启动失败")
            false
        }
    }

    fun stop(userInitiated: Boolean = true) {
        if (userInitiated) {
            state.userStoppingStream = true
            cancelReconnect()
        }
        if (camera.isStreaming) camera.stopStream()
        lastBitrateTimestampMs = 0L
        callback.onStreamStateChanged(false, false)
    }

    fun scheduleReconnect(urlProvider: () -> String, delayMs: Long = 3000L) {
        if (state.userStoppingStream) return
        cancelReconnect()
        val generation = reconnectGeneration
        callback.onStreamStateChanged(false, true)

        val runnable = Runnable {
            if (generation != reconnectGeneration || state.userStoppingStream || camera.isStreaming) return@Runnable
            start(urlProvider())
        }
        reconnectRunnable = runnable
        surfaceView.postDelayed(runnable, delayMs)
    }

    private fun cancelReconnect() {
        reconnectGeneration++
        reconnectRunnable?.let { surfaceView.removeCallbacks(it) }
        reconnectRunnable = null
    }

    fun shouldAutoReconnect(reason: String): Boolean = true

    override fun onConnectionSuccess() {
        reconnectGeneration++
        callback.onStreamStateChanged(true, false)
    }

    override fun onConnectionFailed(reason: String) {
        currentRealBitrate = 0L
        callback.onBitrateChanged(0L)
        callback.onStreamError(reason)
    }

    override fun onConnectionStarted(url: String) = Unit
    override fun onDisconnect() {
        currentRealBitrate = 0L
        callback.onBitrateChanged(0L)
        callback.onStreamError("SRT 已断开")
    }
    override fun onAuthError() = Unit
    override fun onAuthSuccess() = Unit

    override fun onNewBitrate(bitrate: Long) {
        val now = SystemClock.elapsedRealtime()
        if (lastBitrateTimestampMs > 0L && currentRealBitrate > 0L) {
            val elapsedMs = (now - lastBitrateTimestampMs).coerceAtLeast(0L)
            totalStreamedBytes += (currentRealBitrate * elapsedMs / 8_000L)
        }
        lastBitrateTimestampMs = now
        currentRealBitrate = bitrate
        callback.onBitrateChanged(bitrate)
    }

    companion object {
        private const val TAG = "StreamController"
    }
}