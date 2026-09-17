package com.example.aiobs.ai

import android.graphics.Bitmap
import android.util.Log

/**
 * Native Bitmap-side preprocessing helper for XFeat gray-patch extraction.
 *
 * Camera YUV -> Bitmap remains unchanged in this stage.
 */
object NativeBitmapPreprocessor {
    private const val TAG = "NativeBitmapPreproc"

    @Volatile
    private var loaded = false

    init {
        loaded = try {
            System.loadLibrary("aiobs_bitmap_preproc")
            true
        } catch (t: Throwable) {
            Log.w(TAG, "Native bitmap preprocessor unavailable; Kotlin fallback will be used", t)
            false
        }
    }


    fun extractGrayPatch(
        bitmap: Bitmap,
        rectLeft: Int,
        rectTop: Int,
        rectWidth: Int,
        rectHeight: Int,
        mask: SegmentationMask?,
        rawLeft: Float,
        rawTop: Float,
        rawRight: Float,
        rawBottom: Float,
        output: ByteArray
    ): Boolean {
        if (!loaded) return false
        val hasMask = mask != null && mask.width > 0 && mask.height > 0 &&
                mask.pixels.isNotEmpty() && mask.area > 0
        return try {
            nativeExtractGrayPatch(
                bitmap,
                rectLeft,
                rectTop,
                rectWidth,
                rectHeight,
                if (hasMask) mask!!.pixels else ByteArray(0),
                if (hasMask) mask.width else 0,
                if (hasMask) mask.height else 0,
                rawLeft,
                rawTop,
                rawRight,
                rawBottom,
                output
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Native gray patch extraction failed", t)
            false
        }
    }

    @JvmStatic
    private external fun nativeExtractGrayPatch(
        bitmap: Bitmap,
        rectLeft: Int,
        rectTop: Int,
        rectWidth: Int,
        rectHeight: Int,
        maskPixels: ByteArray,
        maskWidth: Int,
        maskHeight: Int,
        rawLeft: Float,
        rawTop: Float,
        rawRight: Float,
        rawBottom: Float,
        output: ByteArray
    ): Boolean
}
