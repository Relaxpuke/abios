package com.example.aiobs.camera

import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.util.Log
import com.pedro.library.srt.SrtCamera2
import java.lang.reflect.Field
import kotlin.math.max
import kotlin.math.min

/**
 * 所有对 Pedro 内部 Camera2 CaptureRequest.Builder 的操作集中在这里。
 *
 * UI 只描述“想要什么”，本层负责：
 * 1. 按设备能力裁剪参数；
 * 2. 手动对焦只关闭 AF，不强制关闭 AE；
 * 3. 只有真正选择 ISO/曝光等手动传感器参数时才进入 Manual Sensor；
 * 4. FPS 使用设备声明的合法 Range。
 */
object CameraSettingsManager {

    /** MainActivity 当前 UI 所需的摄影参数快照。 */
    data class Advanced(
        val afMode: Int = CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO,
        val focusDistance: Float? = null,
        val iso: Int? = null,
        val exposureTimeNs: Long? = null,
        val exposureCompensation: Int? = null,
        val awbMode: Int? = null,
        val videoStabilizationMode: Int? = null,
        val flashMode: Int? = null
    )

    internal const val manualFocusAfMode: Int = CameraMetadata.CONTROL_AF_MODE_OFF

    /** 无 UI 快照时使用安全的自动摄影模式。 */
    fun applySettings(
        srtCamera2: SrtCamera2,
        isProMode: Boolean,
        zoomLevel: Float,
        targetFps: Int
    ) {
        val advanced = if (isProMode) {
            Advanced(
                afMode = CameraMetadata.CONTROL_AF_MODE_OFF,
                focusDistance = null,
                iso = 800,
                exposureTimeNs = 1_000_000_000L / max(1, targetFps * 2)
            )
        } else {
            Advanced()
        }
        applySettings(srtCamera2, null, advanced, zoomLevel, targetFps)
    }

    /** 新 UI 路径：按 Advanced 参数应用。 */
    fun applySettings(
        srtCamera2: SrtCamera2,
        capabilities: CameraCapabilities?,
        advanced: Advanced,
        zoomLevel: Float,
        targetFps: Int
    ) {
        try {
            val cameraManager = findCameraManager(srtCamera2)
                ?: throw IllegalStateException("找不到 CameraManager")
            val builder = findCaptureRequestBuilder(cameraManager)
                ?: throw IllegalStateException("找不到 CaptureRequest.Builder")

            val safeFps = targetFps.coerceIn(1, 240)
            val fpsRange = capabilities?.fpsRanges
                ?.filter { safeFps in it.lower..it.upper }
                ?.minByOrNull { (it.upper - it.lower) + kotlin.math.abs(it.upper - safeFps) }
                ?: capabilities?.fpsRanges?.minByOrNull {
                    if (safeFps < it.lower) it.lower - safeFps
                    else if (safeFps > it.upper) safeFps - it.upper
                    else 0
                }

            val manualFocus = advanced.afMode == CameraMetadata.CONTROL_AF_MODE_OFF
            val manualSensor = advanced.iso != null || advanced.exposureTimeNs != null

            // 对焦和曝光完全拆开：手动对焦不等于全手动曝光。
            builder.safeSet(
                CaptureRequest.CONTROL_AF_MODE,
                if (manualFocus) CameraMetadata.CONTROL_AF_MODE_OFF
                else normalizeAfMode(advanced.afMode, capabilities)
            )

            if (manualFocus) {
                capabilities?.minFocusDistance?.takeIf { it > 0f }?.let { maxFocus ->
                    val d = (advanced.focusDistance ?: 0f).coerceIn(0f, maxFocus)
                    builder.safeSet(CaptureRequest.LENS_FOCUS_DISTANCE, d)
                }
            } else {
                runCatching { builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, null) }
            }

            if (manualSensor) {
                // AE_OFF 仅在用户真正调整 ISO/曝光时启用。
                builder.safeSet(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)

                capabilities?.isoRange?.let { r ->
                    advanced.iso?.let { value ->
                        builder.safeSet(CaptureRequest.SENSOR_SENSITIVITY, value.coerceIn(r.lower, r.upper))
                    }
                }

                capabilities?.exposureTimeRangeNs?.let { r ->
                    advanced.exposureTimeNs?.let { value ->
                        val frameNs = 1_000_000_000L / max(1, safeFps)
                        val exposure = value.coerceIn(r.lower, r.upper).coerceAtMost(frameNs)
                        builder.safeSet(CaptureRequest.SENSOR_EXPOSURE_TIME, exposure)
                        builder.safeSet(CaptureRequest.SENSOR_FRAME_DURATION, frameNs)
                    }
                }
            } else {
                builder.safeSet(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
                builder.safeSet(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, advanced.exposureCompensation ?: 0)
            }

            advanced.awbMode?.let { builder.safeSet(CaptureRequest.CONTROL_AWB_MODE, it) }
            advanced.videoStabilizationMode?.let {
                builder.safeSet(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, it)
            }

            // flashMode: 0/1/2 仅在设备声明支持时尝试下发；失败由 safeSet 截获。
            when (advanced.flashMode) {
                0 -> builder.safeSet(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF)
                1 -> builder.safeSet(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_SINGLE)
                2 -> builder.safeSet(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_TORCH)
            }

            fpsRange?.let { builder.safeSet(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it) }
            if (!manualSensor) {
                builder.safeSet(
                    CaptureRequest.SENSOR_FRAME_DURATION,
                    1_000_000_000L / max(1, safeFps)
                )
            }

            invokeSetZoom(cameraManager, zoomLevel.coerceAtLeast(1.0f))

            Log.d(
                TAG,
                "applySettings af=${advanced.afMode} manualFocus=$manualFocus " +
                    "manualSensor=$manualSensor fps=$safeFps fpsRange=$fpsRange " +
                    "iso=${advanced.iso} focus=${advanced.focusDistance} " +
                    "exposureNs=${advanced.exposureTimeNs} zoom=$zoomLevel"
            )
        } catch (e: Exception) {
            Log.e(TAG, "Camera settings update failed", e)
        }
    }

    private fun normalizeAfMode(mode: Int, capabilities: CameraCapabilities?): Int {
        if (capabilities == null || capabilities.supportsAfMode(mode)) return mode
        return when {
            capabilities.supportsAfMode(CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO) -> CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO
            capabilities.supportsAfMode(CameraMetadata.CONTROL_AF_MODE_AUTO) -> CameraMetadata.CONTROL_AF_MODE_AUTO
            capabilities.afModes.isNotEmpty() -> capabilities.afModes.first()
            else -> CameraMetadata.CONTROL_AF_MODE_OFF
        }
    }

    private fun findCameraManager(srtCamera2: SrtCamera2): Any? {
        var clazz: Class<*>? = srtCamera2.javaClass
        while (clazz != null) {
            findField(clazz, "cameraManager")?.let { field ->
                try {
                    field.isAccessible = true
                    return field.get(srtCamera2)
                } catch (_: Throwable) { }
            }
            clazz = clazz.superclass
        }
        return null
    }

    private fun findCaptureRequestBuilder(cameraManager: Any): CaptureRequest.Builder? {
        var clazz: Class<*>? = cameraManager.javaClass
        while (clazz != null) {
            for (field in clazz.declaredFields) {
                try {
                    field.isAccessible = true
                    val value = field.get(cameraManager)
                    if (value is CaptureRequest.Builder) return value
                } catch (_: Throwable) { }
            }
            clazz = clazz.superclass
        }
        return null
    }

    private fun invokeSetZoom(cameraManager: Any, zoom: Float) {
        val method: java.lang.reflect.Method? = cameraManager.javaClass.declaredMethods.firstOrNull {
            it.name == "setZoom" && it.parameterTypes.size == 1
        }
        if (method != null) {
            try {
                method.isAccessible = true
                method.invoke(cameraManager, zoom)
            } catch (t: Throwable) {
                Log.w(TAG, "CameraManager.setZoom invocation failed", t)
            }
        }
    }

    private fun findField(clazz: Class<*>, name: String): Field? = try {
        clazz.getDeclaredField(name)
    } catch (_: NoSuchFieldException) {
        clazz.superclass?.let { findField(it, name) }
    }

    private fun <T> CaptureRequest.Builder.safeSet(
        key: CaptureRequest.Key<T>,
        value: T
    ) {
        try {
            set(key, value)
        } catch (e: Exception) {
            Log.w(TAG, "Unsupported capture key: ${key.name}", e)
        }
    }

    private const val TAG = "AIOBS_CAM"
}
