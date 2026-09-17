package com.example.aiobs.ai

import android.graphics.Bitmap
import android.util.Log
import java.nio.ByteBuffer

/**
 * ARM64 native INT8/UINT8 & FLOAT32 NCHW input packer.
 */
object NativeYoloInputPacker {
    private const val TAG = "NativeYoloInputPacker"

    @Volatile
    private var loaded = false

    init {
        try {
            System.loadLibrary("aiobs_input_neon")
            loaded = true
            Log.i(TAG, "ARM64 NEON input packer loaded")
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to load aiobs_input_neon", t)
        }
    }

    fun isAvailable(): Boolean = loaded

    // 原有量化通道
    fun packNchwQuantized(
        inputPixels: IntArray,
        outputBuffer: ByteBuffer,
        quantizedLut: ByteArray,
        width: Int,
        height: Int
    ): Boolean {
        if (!loaded) return false
        return try {
            nativePackNchwQuantized(
                inputPixels,
                outputBuffer,
                quantizedLut,
                width,
                height
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Native quantized input pack failed", t)
            false
        }
    }

    // 新增 FP32 浮点通道
    fun packBitmapNchwFloat32(
        bitmap: Bitmap,
        outputBuffer: ByteBuffer,
        width: Int,
        height: Int
    ): Boolean {
        if (!loaded) return false
        return try {
            nativePackBitmapNchwFloat32(bitmap, outputBuffer, width, height)
        } catch (t: Throwable) {
            Log.e(TAG, "Native bitmap float32 input pack failed", t)
            false
        }
    }

    fun packBitmapNchwQuantized(
        bitmap: Bitmap,
        outputBuffer: ByteBuffer,
        quantizedLut: ByteArray,
        width: Int,
        height: Int
    ): Boolean {
        if (!loaded) return false
        return try {
            nativePackBitmapNchwQuantized(bitmap, outputBuffer, quantizedLut, width, height)
        } catch (t: Throwable) {
            Log.e(TAG, "Native bitmap quantized input pack failed", t)
            false
        }
    }

    // Existing IntArray-based fallback/compatibility path.
    fun packNchwFloat32(
        inputPixels: IntArray,
        outputBuffer: ByteBuffer,
        width: Int,
        height: Int
    ): Boolean {
        if (!loaded) return false
        return try {
            nativePackNchwFloat32(
                inputPixels,
                outputBuffer,
                width,
                height
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Native float32 input pack failed", t)
            false
        }
    }

    @JvmStatic
    private external fun nativePackBitmapNchwFloat32(
        bitmap: Bitmap,
        outputBuffer: ByteBuffer,
        width: Int,
        height: Int
    ): Boolean

    @JvmStatic
    private external fun nativePackBitmapNchwQuantized(
        bitmap: Bitmap,
        outputBuffer: ByteBuffer,
        quantizedLut: ByteArray,
        width: Int,
        height: Int
    ): Boolean

    @JvmStatic
    private external fun nativePackNchwQuantized(
        inputPixels: IntArray,
        outputBuffer: ByteBuffer,
        quantizedLut: ByteArray,
        width: Int,
        height: Int
    ): Boolean

    @JvmStatic
    private external fun nativePackNchwFloat32(
        inputPixels: IntArray,
        outputBuffer: ByteBuffer,
        width: Int,
        height: Int
    ): Boolean
}