package com.example.aiobs.camera

import android.graphics.Bitmap
import android.util.Log

/**
 * Native YUV420 converter for the live pipeline.
 *
 * The public contract intentionally remains Bitmap-based so the rest of the
 * perception / tracking / recovery stack does not change.
 *
 * Native implementation uses ARM64 NEON for the fixed-point YUV->ARGB math
 * and writes directly into the reusable Bitmap pixels.
 */
object NativeYuv420Converter {

    private const val TAG = "AIOBS-YUV-Native"

    @Volatile
    private var available = false

    init {
        available = try {
            System.loadLibrary("aiobs_yuv")
            true
        } catch (t: Throwable) {
            Log.w(TAG, "Native YUV converter unavailable; Kotlin fallback will be used", t)
            false
        }
    }

    fun isAvailable(): Boolean = available

    fun convert(
        bitmap: Bitmap,
        yData: ByteArray,
        uData: ByteArray,
        vData: ByteArray,
        yRowStride: Int,
        yPixelStride: Int,
        uRowStride: Int,
        uPixelStride: Int,
        vRowStride: Int,
        vPixelStride: Int,
        targetWidth: Int,
        targetHeight: Int,
        srcXMap: IntArray,
        uvXMap: IntArray,
        yRowMap: IntArray,
        uvYMap: IntArray
    ): Boolean {
        if (!available) return false

        return try {
            nativeConvert(
                bitmap,
                yData,
                uData,
                vData,
                yRowStride,
                yPixelStride,
                uRowStride,
                uPixelStride,
                vRowStride,
                vPixelStride,
                targetWidth,
                targetHeight,
                srcXMap,
                uvXMap,
                yRowMap,
                uvYMap
            )
        } catch (t: Throwable) {
            available = false
            Log.e(TAG, "Native YUV conversion failed; disabling native path", t)
            false
        }
    }

    private external fun nativeConvert(
        bitmap: Bitmap,
        yData: ByteArray,
        uData: ByteArray,
        vData: ByteArray,
        yRowStride: Int,
        yPixelStride: Int,
        uRowStride: Int,
        uPixelStride: Int,
        vRowStride: Int,
        vPixelStride: Int,
        targetWidth: Int,
        targetHeight: Int,
        srcXMap: IntArray,
        uvXMap: IntArray,
        yRowMap: IntArray,
        uvYMap: IntArray
    ): Boolean
}
