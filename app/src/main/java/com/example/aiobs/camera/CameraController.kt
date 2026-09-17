package com.example.aiobs.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Log
import android.view.SurfaceHolder
import com.example.aiobs.ai.VisualRadarController
import com.example.aiobs.core.AppState
import com.pedro.encoder.utils.gl.AspectRatioMode
import com.pedro.library.srt.SrtCamera2
import com.pedro.library.view.OpenGlView

/**
 * Camera2/SrtCamera2 控制层。
 * Preview 生命周期保持原有路径；Camera2 能力读取与 CaptureRequest 参数下发解耦。
 */
class CameraController(
    private val context: Context,
    private val camera: SrtCamera2,
    private val surfaceView: OpenGlView,
    private val state: AppState
) {
    private val aiFrameSource = CameraAiFrameSource(context, camera, state)
    @Volatile private var surfaceAvailable = false

    @Volatile private var cachedCameraCapabilities: CameraCapabilities? = null
    @Volatile private var lastAdvanced = CameraSettingsManager.Advanced()

    fun startPreview() {
        if (!surfaceAvailable) return
        if (camera.isOnPreview) return
        try {
            surfaceView.setAspectRatioMode(AspectRatioMode.Adjust)
            camera.startPreview(state.videoWidth, state.videoHeight)
            Log.i(TAG, "Camera preview START requested ${state.videoWidth}x${state.videoHeight}")
        } catch (t: Throwable) {
            Log.e(TAG, "Camera preview START failed", t)
        }
    }

    fun stopPreview() {
        if (!camera.isOnPreview) return
        try {
            camera.stopPreview()
            Log.i(TAG, "Camera preview STOP requested")
        } catch (t: Throwable) {
            Log.e(TAG, "Camera preview STOP failed", t)
        }
    }

    fun rebuildPreview(
        prepare: () -> Boolean,
        beforeRestart: (() -> Unit)? = null,
        afterRestart: (() -> Unit)? = null,
        delayMs: Long = 250L
    ) {
        beforeRestart?.invoke()
        runCatching { stopPreview() }
        surfaceView.postDelayed({
            if (camera.isOnPreview) return@postDelayed
            try {
                if (!prepare()) return@postDelayed
                startPreview()
                applySettings()
                afterRestart?.invoke()
            } catch (t: Throwable) {
                Log.e(TAG, "rebuildPreview failed", t)
            }
        }, delayMs)
    }

    fun prepareAiFrameSource(): Boolean = aiFrameSource.prepare()
    fun aiFrameSource(): VisualRadarController.FrameSource = aiFrameSource
    fun releaseAiFrameSource() = aiFrameSource.release()
    fun cameraStatsSnapshot(): CameraAiFrameSource.StatsSnapshot = aiFrameSource.statsSnapshot()

    fun applySettingsDelayed(delayMs: Long = 500L) {
        surfaceView.postDelayed({ applySettings() }, delayMs)
    }

    /** 兼容 MainActivity 原有无参数调用。 */
    fun applySettings() {
        if (!camera.isOnPreview && !camera.isStreaming) return
        applySettings(lastAdvanced)
    }

    /** 新 UI 使用的完整摄影参数入口。 */
    fun applySettings(advanced: CameraSettingsManager.Advanced) {
        if (!camera.isOnPreview && !camera.isStreaming) return
        lastAdvanced = advanced
        try {
            val caps = getCameraCapabilities()
            CameraSettingsManager.applySettings(
                srtCamera2 = camera,
                capabilities = caps,
                advanced = advanced,
                zoomLevel = state.zoomLevel,
                targetFps = state.videoFps
            )
        } catch (t: Throwable) {
            Log.e(TAG, "applySettings failed", t)
        }
    }

    fun getSensorOrientation(): Int {
        return try {
            val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cid = getCurrentCameraId() ?: return 90
            manager.getCameraCharacteristics(cid).get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        } catch (_: Throwable) {
            90
        }
    }

    fun getCameraCapabilities(forceRefresh: Boolean = false): CameraCapabilities? {
        if (!forceRefresh) cachedCameraCapabilities?.let { return it }
        val caps = try {
            val directId = getCurrentCameraId()
            if (!directId.isNullOrBlank()) {
                CameraCapabilitiesReader.read(context, directId)
            } else null
        } catch (_: Throwable) { null }

        val resolved = caps ?: try {
            val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val wanted = if (state.frontCamera) {
                CameraCharacteristics.LENS_FACING_FRONT
            } else {
                CameraCharacteristics.LENS_FACING_BACK
            }
            val id = manager.cameraIdList.firstOrNull { cameraId ->
                runCatching {
                    manager.getCameraCharacteristics(cameraId)
                        .get(CameraCharacteristics.LENS_FACING) == wanted
                }.getOrDefault(false)
            } ?: manager.cameraIdList.firstOrNull()
            id?.let { CameraCapabilitiesReader.read(context, it) }
        } catch (t: Throwable) {
            Log.e(TAG, "Camera capability discovery failed", t)
            null
        }

        if (resolved != null) cachedCameraCapabilities = resolved
        return resolved
    }


    private fun getCurrentCameraId(): String? {
        var clazz: Class<*>? = camera.javaClass
        while (clazz != null) {
            try {
                val field = clazz.getDeclaredField("cameraId")
                field.isAccessible = true
                (field.get(camera) as? String)?.let { return it }
            } catch (_: Throwable) { }
            clazz = clazz.superclass
        }
        return null
    }

    fun attachSurfaceCallbacks(onReady: () -> Unit, onDestroyed: () -> Unit) {
        surfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            private var isReady = false
            override fun surfaceCreated(holder: SurfaceHolder) {
                surfaceAvailable = true
                isReady = false
            }
            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                surfaceAvailable = true
                if (!isReady) {
                    isReady = true
                    onReady()
                }
            }
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                surfaceAvailable = false
                isReady = false
                stopPreview()
                onDestroyed()
            }
        })
    }

    fun isPreviewing(): Boolean = camera.isOnPreview

    companion object {
        private const val TAG = "CameraControllerV22"
    }
}
