package com.example.aiobs.ai

import android.content.Context
import android.util.Log

class Dinov2QnnContextBinaryRunner(
    context: Context,
    val assetName: String
) : AutoCloseable {

    private val handle: Long
    val inputWidth: Int
    val inputHeight: Int
    val embeddingDim: Int

    @Volatile
    var lastInferenceMs: Float = 0f
        private set

    init {
        Log.i(TAG, "DINOv2 QNN Runner init BEGIN asset=$assetName")
        System.loadLibrary("aiobs_dinov2_qnn_context")

        handle = nativeCreate(
            context.applicationContext.assets,
            assetName,
            context.applicationContext.applicationInfo.nativeLibraryDir
        )
        if (handle == 0L) error("Failed to load DINOv2 QNN context binary: $assetName")

        inputWidth = nativeGetInputWidth(handle)
        inputHeight = nativeGetInputHeight(handle)
        embeddingDim = nativeGetEmbeddingDim(handle)

        Log.i(TAG, "DINOv2 Ready! 动态嗅探结果: ${inputWidth}x${inputHeight}, 特征维度=$embeddingDim")
    }

    fun isReady(): Boolean = handle != 0L

    fun infer(normalizedNchwInput: FloatArray): FloatArray {
        require(normalizedNchwInput.size == 3 * inputWidth * inputHeight) {
            "DINOv2 input size mismatch"
        }

        val startNs = System.nanoTime()
        val result = nativeRun(handle, normalizedNchwInput) ?: error("DINOv2 graph execute failed")
        lastInferenceMs = (System.nanoTime() - startNs) / 1_000_000.0f
        return result
    }

    override fun close() {
        if (handle != 0L) {
            nativeClose(handle)
        }
    }

    companion object {
        private const val TAG = "Dinov2QnnCtx"

        @JvmStatic private external fun nativeCreate(assetManager: android.content.res.AssetManager, assetName: String, nativeLibDir: String): Long
        @JvmStatic private external fun nativeGetInputWidth(handle: Long): Int
        @JvmStatic private external fun nativeGetInputHeight(handle: Long): Int
        @JvmStatic private external fun nativeGetEmbeddingDim(handle: Long): Int
        @JvmStatic private external fun nativeRun(handle: Long, input: FloatArray): FloatArray?
        @JvmStatic private external fun nativeClose(handle: Long)
    }
}