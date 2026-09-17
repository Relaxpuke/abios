package com.example.aiobs.ai

import android.content.Context
import android.util.Log

/**
 * Direct QNN Context Binary runner for XFeat W8A16 / SM8550.
 *
 * Supported production Context Binaries:
 *   384x384 -> 48x48 feature grid
 *   480x480 -> 60x60 feature grid
 *   512x512 -> 64x64 feature grid
 *   640x640 -> 80x80 feature grid
 *
 * The native layer validates the real tensor contract from the Context Binary
 * metadata and returns decoded FP32 tensors in logical NCHW order.
 */
class XFeatQnnContextBinaryRunner(
    context: Context,
    val assetName: String = W8A16_ASSET_384
) : AutoCloseable {
    data class Result(
        val feats: FloatArray,
        val keypointLogits: FloatArray,
        val reliability: FloatArray,
        val elapsedMs: Double
    )

    private val appContext = context.applicationContext
    private var handle: Long = 0L

    val modelSize: Int
        get() = modelSizeForAsset(assetName)

    val cellSize: Int
        get() = modelSize / 8

    val inputElements: Int
        get() = modelSize * modelSize

    val featsElements: Int
        get() = 64 * cellSize * cellSize

    val logitsElements: Int
        get() = 65 * cellSize * cellSize

    val reliabilityElements: Int
        get() = cellSize * cellSize

    init {
        Log.i(TAG, "XFeat W8A16 QNN runner init BEGIN asset=$assetName model=$modelSize")
        System.loadLibrary(NATIVE_LIBRARY)
        handle = nativeCreate(
            appContext.assets,
            assetName,
            appContext.applicationInfo.nativeLibraryDir
        )
        Log.i(TAG, "XFeat W8A16 QNN runner init END asset=$assetName model=$modelSize handle=$handle")
    }

    fun isReady(): Boolean = handle != 0L

    fun describe(): String {
        checkReady()
        return nativeDescribe(handle)
    }

    fun infer(input: FloatArray): Result {
        checkReady()
        require(input.size == inputElements) {
            "XFeat-$modelSize input must contain $inputElements FP32 elements"
        }

        val packed = nativeRun(handle, input)
            ?: error("XFeat nativeRun returned null")

        val expected = featsElements + logitsElements + reliabilityElements + 1
        require(packed.size == expected) {
            "Unexpected native result size=${packed.size}, expected=$expected"
        }

        var offset = 0
        val feats = FloatArray(featsElements).also {
            packed.copyInto(it, 0, offset, offset + featsElements)
            offset += featsElements
        }
        val logits = FloatArray(logitsElements).also {
            packed.copyInto(it, 0, offset, offset + logitsElements)
            offset += logitsElements
        }
        val reliability = FloatArray(reliabilityElements).also {
            packed.copyInto(it, 0, offset, offset + reliabilityElements)
            offset += reliabilityElements
        }

        return Result(
            feats = feats,
            keypointLogits = logits,
            reliability = reliability,
            elapsedMs = packed[offset].toDouble()
        )
    }

    fun benchmark(warmups: Int = 5, iterations: Int = 20): String {
        checkReady()
        require(warmups >= 0)
        require(iterations > 0)
        return nativeBenchmark(handle, warmups, iterations)
    }

    override fun close() {
        if (handle != 0L) {
            nativeClose(handle)
            handle = 0L
        }
    }

    private fun checkReady() {
        check(handle != 0L) {
            "XFeat W8A16 QNN Context Binary runner is not ready"
        }
    }

    companion object {
        // Current production default: 384 W8A16 Context Binary.
        // Supported production Context Binaries:
        // 384x384 -> 48x48 feature grid
        // 480x480 -> 60x60 feature grid
        // 512x512 -> 64x64 feature grid
        // 640x640 -> 80x80 feature grid

        const val W8A16_ASSET_384 = "xfeat_384_w8a16_sm8550.bin.SM8550.bin"
        const val W8A16_ASSET_480 = "xfeat_480_w8a16_sm8550.bin.SM8550.bin"
        const val W8A16_ASSET_512 = "xfeat_512_w8a16_sm8550.bin.SM8550.bin"
        const val W8A16_ASSET_640 = "xfeat_640_w8a16_sm8550.bin.SM8550.bin"

        // Backward-compatible alias used by existing production code.
        const val W8A16_ASSET = W8A16_ASSET_480

        const val INT8_ASSET = "xfeat256_int8.serialized.bin"

        const val DEFAULT_MODEL_SIZE = 384
        const val DEFAULT_GRID_SIZE = DEFAULT_MODEL_SIZE / 8
        const val INPUT_ELEMENTS =
            DEFAULT_MODEL_SIZE * DEFAULT_MODEL_SIZE
        const val FEATS_ELEMENTS =
            64 * DEFAULT_GRID_SIZE * DEFAULT_GRID_SIZE
        const val LOGITS_ELEMENTS =
            65 * DEFAULT_GRID_SIZE * DEFAULT_GRID_SIZE
        const val RELIABILITY_ELEMENTS =
            DEFAULT_GRID_SIZE * DEFAULT_GRID_SIZE

        fun modelSizeForAsset(assetName: String): Int = when (assetName) {
            W8A16_ASSET_384 -> 384
            W8A16_ASSET_480 -> 480
            W8A16_ASSET_512 -> 512
            W8A16_ASSET_640 -> 640
            else -> {
                Regex("xfeat_(\\d+)_w8a16_sm8550")
                    .find(assetName)
                    ?.groupValues
                    ?.getOrNull(1)
                    ?.toIntOrNull()
                    ?.takeIf { it > 0 && it % 8 == 0 }
                    ?: DEFAULT_MODEL_SIZE
            }
        }

        private const val NATIVE_LIBRARY = "aiobs_xfeat_qnn_context"
        private const val TAG = "XFeatQnnCtxBin"

        @JvmStatic
        private external fun nativeCreate(
            assetManager: android.content.res.AssetManager,
            assetName: String,
            nativeLibDir: String
        ): Long

        @JvmStatic
        private external fun nativeDescribe(handle: Long): String

        @JvmStatic
        private external fun nativeRun(
            handle: Long,
            input: FloatArray
        ): FloatArray?

        @JvmStatic
        private external fun nativeBenchmark(
            handle: Long,
            warmups: Int,
            iterations: Int
        ): String

        @JvmStatic
        private external fun nativeClose(handle: Long)
    }
}
