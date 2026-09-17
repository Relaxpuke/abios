package com.example.aiobs.camera

import android.content.Context
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.params.StreamConfigurationMap
import android.util.Range
import android.util.Size
import java.util.Locale

/** Runtime camera capabilities read from Camera2. Unsupported items are omitted by the UI. */
data class CameraCapabilities(
    val cameraId: String,
    val lensFacing: Int,
    val sensorOrientation: Int,
    val outputSizes: List<Size>,
    val fpsRanges: List<Range<Int>>,
    val maxDigitalZoom: Float,
    val afModes: List<Int>,
    val minFocusDistance: Float,
    val isoRange: Range<Int>?,
    val exposureTimeRangeNs: Range<Long>?,
    val exposureCompensationRange: Range<Int>?,
    val aeModes: List<Int>,
    val awbModes: List<Int>,
    val videoStabilizationModes: List<Int>,
    val opticalStabilizationModes: List<Int>,
    val flashAvailable: Boolean
) {
    fun supportsAfMode(mode: Int): Boolean = afModes.contains(mode)
    fun supportsManualFocus(): Boolean = minFocusDistance > 0f && supportsAfMode(CameraMetadata.CONTROL_AF_MODE_OFF)
}

object CameraCapabilitiesReader {
    fun read(context: Context, cameraId: String): CameraCapabilities? {
        return try {
            val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val c = manager.getCameraCharacteristics(cameraId)
            val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val sizes = collectOutputSizes(map)
            val fps = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                ?.toList()
                ?.sortedWith(compareBy<Range<Int>> { it.upper }.thenBy { it.lower })
                ?: emptyList()
            val af = c.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)?.toList() ?: emptyList()
            val ae = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES)?.toList() ?: emptyList()
            val awb = c.get(CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES)?.toList() ?: emptyList()
            val videoStab = c.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)?.toList() ?: emptyList()
            val opticalStab = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)?.toList() ?: emptyList()

            CameraCapabilities(
                cameraId = cameraId,
                lensFacing = c.get(CameraCharacteristics.LENS_FACING) ?: CameraCharacteristics.LENS_FACING_BACK,
                sensorOrientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90,
                outputSizes = sizes,
                fpsRanges = fps,
                maxDigitalZoom = c.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f,
                afModes = af,
                minFocusDistance = c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f,
                isoRange = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE),
                exposureTimeRangeNs = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE),
                exposureCompensationRange = c.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE),
                aeModes = ae,
                awbModes = awb,
                videoStabilizationModes = videoStab,
                opticalStabilizationModes = opticalStab,
                flashAvailable = c.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
            )
        } catch (_: Throwable) {
            null
        }
    }


    private fun collectOutputSizes(map: StreamConfigurationMap?): List<Size> {
        if (map == null) return emptyList()
        val candidates = linkedSetOf<Size>()
        try { map.getOutputSizes(SurfaceTexture::class.java)?.forEach { candidates += it } } catch (_: Throwable) {}
        try { map.getOutputSizes(android.graphics.ImageFormat.YUV_420_888)?.forEach { candidates += it } } catch (_: Throwable) {}
        return candidates
            .filter { it.width > 0 && it.height > 0 }
            .sortedWith(compareByDescending<Size> { it.width.toLong() * it.height.toLong() }.thenByDescending { it.width })
    }


    fun formatSize(size: Size): String = "${size.width} × ${size.height}"

    fun formatFps(range: Range<Int>): String {
        return if (range.lower == range.upper) "${range.upper} FPS" else "${range.lower}–${range.upper} FPS"
    }

    fun lensLabel(facing: Int): String = when (facing) {
        CameraCharacteristics.LENS_FACING_FRONT -> "前置"
        CameraCharacteristics.LENS_FACING_BACK -> "后置"
        CameraCharacteristics.LENS_FACING_EXTERNAL -> "外接"
        else -> String.format(Locale.US, "Camera(%d)", facing)
    }
}
