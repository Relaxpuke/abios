package com.example.aiobs.camera

import android.content.Context
import android.hardware.camera2.CameraMetadata
import com.example.aiobs.core.AppState

/**
 * Camera 参数持久化。
 *
 * 只保存“用户确认/当前使用”的摄影参数，不保存 Camera2 对象或运行时句柄。
 * 启动时先读取，再由当前 CameraCapabilities 做合法化。
 */
object CameraSettingsStore {
    private const val PREFS = "v52_camera_settings_v2"

    private const val KEY_WIDTH = "width"
    private const val KEY_HEIGHT = "height"
    private const val KEY_FPS = "fps"
    private const val KEY_BITRATE = "bitrate"
    private const val KEY_ZOOM = "zoom"
    private const val KEY_FRONT = "front_camera"
    private const val KEY_AF_MODE = "af_mode"
    private const val KEY_FOCUS = "focus_distance"
    private const val KEY_ISO = "iso"
    private const val KEY_EXPOSURE = "exposure_ns"
    private const val KEY_EV = "ev"
    private const val KEY_AWB = "awb_mode"
    private const val KEY_VIDEO_STAB = "video_stab"
    private const val KEY_FLASH = "flash_mode"
    private const val KEY_ORIENTATION_LOCKED = "orientation_locked"
    private const val KEY_PORTRAIT = "portrait"

    fun loadOrientationLocked(context: Context, defaultValue: Boolean): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_ORIENTATION_LOCKED, defaultValue)

    /** Loads the persisted camera state into the current AppState and returns advanced settings. */
    fun load(context: Context, state: AppState): CameraSettingsManager.Advanced {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

        state.videoWidth = p.getInt(KEY_WIDTH, state.videoWidth)
        state.videoHeight = p.getInt(KEY_HEIGHT, state.videoHeight)
        state.videoFps = p.getInt(KEY_FPS, state.videoFps)
        state.videoBitrate = p.getInt(KEY_BITRATE, state.videoBitrate)
        state.zoomLevel = p.getFloat(KEY_ZOOM, state.zoomLevel)
        state.frontCamera = p.getBoolean(KEY_FRONT, state.frontCamera)
        state.portrait = p.getBoolean(KEY_PORTRAIT, state.portrait)

        return CameraSettingsManager.Advanced(
            afMode = p.getInt(KEY_AF_MODE, CameraSettingsManager.Advanced().afMode),
            focusDistance = if (p.contains(KEY_FOCUS)) p.getFloat(KEY_FOCUS, 0f) else null,
            iso = if (p.contains(KEY_ISO)) p.getInt(KEY_ISO, 0) else null,
            exposureTimeNs = if (p.contains(KEY_EXPOSURE)) p.getLong(KEY_EXPOSURE, 0L) else null,
            exposureCompensation = if (p.contains(KEY_EV)) p.getInt(KEY_EV, 0) else null,
            awbMode = if (p.contains(KEY_AWB)) p.getInt(KEY_AWB, 0) else null,
            videoStabilizationMode = if (p.contains(KEY_VIDEO_STAB)) p.getInt(KEY_VIDEO_STAB, 0) else null,
            flashMode = if (p.contains(KEY_FLASH)) p.getInt(KEY_FLASH, 0) else null
        )
    }

    fun save(
        context: Context,
        state: AppState,
        advanced: CameraSettingsManager.Advanced,
        orientationLocked: Boolean
    ) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putInt(KEY_WIDTH, state.videoWidth)
            .putInt(KEY_HEIGHT, state.videoHeight)
            .putInt(KEY_FPS, state.videoFps)
            .putInt(KEY_BITRATE, state.videoBitrate)
            .putFloat(KEY_ZOOM, state.zoomLevel)
            .putBoolean(KEY_FRONT, state.frontCamera)
            .putInt(KEY_AF_MODE, advanced.afMode)
            .apply {
                if (advanced.focusDistance != null) putFloat(KEY_FOCUS, advanced.focusDistance)
                else remove(KEY_FOCUS)
                if (advanced.iso != null) putInt(KEY_ISO, advanced.iso)
                else remove(KEY_ISO)
                if (advanced.exposureTimeNs != null) putLong(KEY_EXPOSURE, advanced.exposureTimeNs)
                else remove(KEY_EXPOSURE)
                if (advanced.exposureCompensation != null) putInt(KEY_EV, advanced.exposureCompensation)
                else remove(KEY_EV)
                if (advanced.awbMode != null) putInt(KEY_AWB, advanced.awbMode)
                else remove(KEY_AWB)
                if (advanced.videoStabilizationMode != null) putInt(KEY_VIDEO_STAB, advanced.videoStabilizationMode)
                else remove(KEY_VIDEO_STAB)
                if (advanced.flashMode != null) putInt(KEY_FLASH, advanced.flashMode)
                else remove(KEY_FLASH)
            }
            .putBoolean(KEY_ORIENTATION_LOCKED, orientationLocked)
            .putBoolean(KEY_PORTRAIT, state.portrait)
            .apply()
    }

    fun normalizeAdvanced(
        advanced: CameraSettingsManager.Advanced,
        caps: CameraCapabilities
    ): CameraSettingsManager.Advanced {
        val afMode = if (caps.supportsAfMode(advanced.afMode)) {
            advanced.afMode
        } else when {
            caps.supportsAfMode(CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO) ->
                CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO
            caps.supportsAfMode(CameraMetadata.CONTROL_AF_MODE_AUTO) ->
                CameraMetadata.CONTROL_AF_MODE_AUTO
            caps.afModes.isNotEmpty() -> caps.afModes.first()
            else -> CameraMetadata.CONTROL_AF_MODE_OFF
        }

        val focus = advanced.focusDistance?.let { d ->
            if (caps.minFocusDistance > 0f) d.coerceIn(0f, caps.minFocusDistance) else null
        }

        val iso = advanced.iso?.let { value ->
            caps.isoRange?.let { value.coerceIn(it.lower, it.upper) }
        }

        val exposure = advanced.exposureTimeNs?.let { value ->
            caps.exposureTimeRangeNs?.let { r ->
                value.coerceIn(r.lower, r.upper)
            }
        }

        val ev = advanced.exposureCompensation?.let { value ->
            caps.exposureCompensationRange?.let { value.coerceIn(it.lower, it.upper) }
        }

        val awb = advanced.awbMode?.takeIf { caps.awbModes.contains(it) }
        val stab = advanced.videoStabilizationMode?.takeIf { caps.videoStabilizationModes.contains(it) }
        val flash = advanced.flashMode?.let { value ->
            if (caps.flashAvailable && value in 0..2) value else null
        }

        return advanced.copy(
            afMode = afMode,
            focusDistance = focus,
            iso = iso,
            exposureTimeNs = exposure,
            exposureCompensation = ev,
            awbMode = awb,
            videoStabilizationMode = stab,
            flashMode = flash
        )
    }
}
