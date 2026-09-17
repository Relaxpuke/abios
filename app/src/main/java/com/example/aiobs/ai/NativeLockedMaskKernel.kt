package com.example.aiobs.ai

import android.util.Log
import java.nio.ByteBuffer

/**
 * Native ARM64 NEON mask decoder for both Quantized and Float32 YOLO masks.
 */
object NativeLockedMaskKernel {
    private const val TAG = "NativeLockedMaskKernel"

    @Volatile
    private var loaded = false

    init {
        try {
            System.loadLibrary("aiobs_mask_neon")
            loaded = true
            Log.i(TAG, "ARM64 NEON mask kernel loaded")
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to load aiobs_mask_neon", t)
        }
    }

    fun isAvailable(): Boolean = loaded

    // 原有量化通道
    fun decodeQuantizedMask(
        detBuffer: ByteBuffer,
        detScale: Float,
        detZeroPoint: Int,
        protoBuffer: ByteBuffer,
        protoScale: Float,
        protoZeroPoint: Int,
        output0Type: Int,
        output1Type: Int,
        anchorIndex: Int,
        boxLeft: Float,
        boxTop: Float,
        boxRight: Float,
        boxBottom: Float,
        modelSize: Int,
        protoSize: Int,
        protoChannels: Int,
        outputSize: Int,
        maskThreshold: Float,
        outputMask: ByteArray
    ): Int {
        if (!loaded) return -1
        return try {
            nativeDecodeQuantizedMask(
                detBuffer, detScale, detZeroPoint, protoBuffer, protoScale, protoZeroPoint,
                output0Type, output1Type, anchorIndex, boxLeft, boxTop, boxRight, boxBottom,
                modelSize, protoSize, protoChannels, outputSize, maskThreshold, outputMask
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Native mask decode failed", t)
            -1
        }
    }

    // 新增 FP32 浮点通道
    fun decodeFloatMask(
        detBuffer: ByteBuffer,
        protoBuffer: ByteBuffer,
        anchorIndex: Int,
        boxLeft: Float,
        boxTop: Float,
        boxRight: Float,
        boxBottom: Float,
        modelSize: Int,
        protoSize: Int,
        protoChannels: Int,
        outputSize: Int,
        maskThreshold: Float,
        outputMask: ByteArray
    ): Int {
        if (!loaded) return -1
        return try {
            nativeDecodeFloatMask(
                detBuffer, protoBuffer, anchorIndex, boxLeft, boxTop, boxRight, boxBottom,
                modelSize, protoSize, protoChannels, outputSize, maskThreshold, outputMask
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Native float mask decode failed", t)
            -1
        }
    }

    @JvmStatic
    private external fun nativeDecodeQuantizedMask(
        detBuffer: ByteBuffer, detScale: Float, detZeroPoint: Int,
        protoBuffer: ByteBuffer, protoScale: Float, protoZeroPoint: Int,
        output0Type: Int, output1Type: Int, anchorIndex: Int,
        boxLeft: Float, boxTop: Float, boxRight: Float, boxBottom: Float,
        modelSize: Int, protoSize: Int, protoChannels: Int, outputSize: Int,
        maskThreshold: Float, outputMask: ByteArray
    ): Int

    @JvmStatic
    private external fun nativeDecodeFloatMask(
        detBuffer: ByteBuffer, protoBuffer: ByteBuffer, anchorIndex: Int,
        boxLeft: Float, boxTop: Float, boxRight: Float, boxBottom: Float,
        modelSize: Int, protoSize: Int, protoChannels: Int, outputSize: Int,
        maskThreshold: Float, outputMask: ByteArray
    ): Int
}