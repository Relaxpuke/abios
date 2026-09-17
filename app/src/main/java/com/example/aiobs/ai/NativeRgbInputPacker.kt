package com.example.aiobs.ai

import android.graphics.Bitmap
import android.util.Log

/**
 * Shared RGB FP32 NCHW input packer.
 *
 * Reads an ARGB_8888/RGBA_8888 Bitmap directly from native memory and writes
 * ImageNet-normalized RGB NCHW values into a caller-owned FloatArray.
 *
 * This is intentionally shared by models that use the same preprocessing
 * contract (RGB + ImageNet mean/std). Model-specific crop/resize policies stay
 * outside this helper.
 */
object NativeRgbInputPacker {
    private const val TAG = "NativeRgbInputPacker"

    @Volatile
    private var loaded = false

    init {
        loaded = try {
            System.loadLibrary("aiobs_input_neon")
            true
        } catch (t: Throwable) {
            Log.w(TAG, "Native RGB packer unavailable", t)
            false
        }
    }

    fun isAvailable(): Boolean = loaded

    fun packBitmapImageNetNchwFloat32(
        bitmap: Bitmap,
        output: FloatArray,
        width: Int,
        height: Int
    ): Boolean {
        if (!loaded) return false
        if (width <= 0 || height <= 0) return false
        if (bitmap.width != width || bitmap.height != height) return false
        if (output.size < 3 * width * height) return false
        return try {
            nativePackBitmapImageNetNchwFloat32(bitmap, output, width, height)
        } catch (t: Throwable) {
            Log.e(TAG, "Native ImageNet RGB pack failed", t)
            false
        }
    }

    @JvmStatic
    private external fun nativePackBitmapImageNetNchwFloat32(
        bitmap: Bitmap,
        output: FloatArray,
        width: Int,
        height: Int
    ): Boolean
}
