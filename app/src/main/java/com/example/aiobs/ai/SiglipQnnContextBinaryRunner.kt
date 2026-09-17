package com.example.aiobs.ai

import android.content.Context
import android.util.Log

/**
 * Standalone SigLIP2 QNN Context Binary validator.
 *
 * Validation-only. No TFLite, camera, tracker, recovery, or production
 * identity path is touched.
 *
 * The native implementation reads the actual QNN tensor metadata from the
 * supplied serialized Context Binary and benchmarks its graph directly.
 *
 * Supported candidate assets:
 *   - siglip2_vit_base_256_fp16.serialized.SM8550.bin
 *   - siglip2_vit_base_256_int8.serialized.SM8550.bin
 *   - siglip2_vit_base_256_w8a16.serialized.SM8550.bin
 */
class SiglipQnnContextBinaryRunner(
    context: Context,
    val assetName: String = DEFAULT_ASSET
) : AutoCloseable {

    data class BenchmarkResult(
        val assetName: String,
        val report: String
    )

    private val appContext =
        context.applicationContext

    private var handle: Long = 0L

    init {
        Log.i(
            TAG,
            "SigLIP runner init BEGIN asset=$assetName"
        )

        System.loadLibrary(NATIVE_LIBRARY)

        Log.i(
            TAG,
            "SigLIP native library loaded: $NATIVE_LIBRARY"
        )

        handle =
            nativeCreate(
                appContext.assets,
                assetName,
                appContext.applicationInfo.nativeLibraryDir
            )

        Log.i(
            TAG,
            "SigLIP runner init END asset=$assetName handle=$handle"
        )
    }

    fun isReady(): Boolean {
        val ready = handle != 0L

        Log.i(
            TAG,
            "SigLIP isReady=$ready asset=$assetName handle=$handle"
        )

        return ready
    }

    fun describe(): String {
        Log.i(
            TAG,
            "SigLIP describe ENTER asset=$assetName handle=$handle"
        )

        checkReady()

        val result =
            nativeDescribe(handle)

        Log.i(
            TAG,
            "SigLIP describe RETURN asset=$assetName length=${result.length}"
        )

        return result
    }

    /**
     * Native timing is measured around QNN graphExecute().
     * Kotlin/UI bookkeeping is excluded from the reported graph latency.
     */
    fun infer(input: FloatArray): FloatArray {
        checkReady()
        require(input.size == 256 * 256 * 3) {
            "SigLIP input must contain 256*256*3 FP32 elements"
        }

        val output = nativeRunImage(handle, input)

        require(output.size == 768) {
            "SigLIP output must contain 768 FP32 elements"
        }

        return output
    }


    /**
     * Native timing is measured only around QNN graphExecute() when the native
     * image inference API logs its execution time. Kotlin image decode and
     * preprocessing are intentionally outside that measurement.
     */
    fun benchmark(
        warmups: Int = DEFAULT_WARMUPS,
        iterations: Int = DEFAULT_ITERATIONS
    ): BenchmarkResult {

        Log.i(
            TAG,
            "SigLIP benchmark ENTER asset=$assetName " +
                    "handle=$handle " +
                    "warmups=$warmups " +
                    "iterations=$iterations"
        )

        checkReady()

        require(warmups >= 0) {
            "warmups must be >= 0"
        }

        require(iterations > 0) {
            "iterations must be > 0"
        }

        Log.i(
            TAG,
            "SigLIP benchmark BEFORE nativeBenchmark asset=$assetName"
        )

        val report =
            nativeBenchmark(
                handle,
                warmups,
                iterations
            )

        Log.i(
            TAG,
            "SigLIP benchmark AFTER nativeBenchmark " +
                    "asset=$assetName reportLength=${report.length}"
        )

        Log.i(
            TAG,
            "SigLIP benchmark REPORT asset=$assetName:\n$report"
        )

        return BenchmarkResult(
            assetName = assetName,
            report = report
        )
    }

    override fun close() {

        val current =
            handle

        Log.i(
            TAG,
            "SigLIP close ENTER asset=$assetName handle=$current"
        )

        if (current != 0L) {

            try {

                nativeClose(
                    current
                )

            } catch (t: Throwable) {

                Log.w(
                    TAG,
                    "Failed to close SigLIP QNN runner asset=$assetName",
                    t
                )

            } finally {

                handle = 0L
            }
        }

        Log.i(
            TAG,
            "SigLIP close END asset=$assetName"
        )
    }

    private fun checkReady() {

        if (handle == 0L) {

            Log.e(
                TAG,
                "SigLIP checkReady FAILED asset=$assetName handle=0"
            )

            error(
                "SigLIP QNN Context Binary runner is not initialized: $assetName"
            )
        }
    }

    companion object {

        private const val TAG =
            "SiglipQnnCtxBin"

        private const val NATIVE_LIBRARY =
            "aiobs_siglip_qnn_context"

        const val FP16_ASSET =
            "siglip2_vit_base_256_fp16.serialized.SM8550.bin"

        const val INT8_ASSET =
            "siglip2_vit_base_256_int8.serialized.SM8550.bin"

        const val W8A16_ASSET =
            "siglip2_vit_base_256_w8a16.serialized.SM8550.bin"

        const val DEFAULT_ASSET =
            FP16_ASSET

        private const val DEFAULT_WARMUPS =
            5

        private const val DEFAULT_ITERATIONS =
            20

        @JvmStatic
        private external fun nativeCreate(
            assetManager: android.content.res.AssetManager,
            assetName: String,
            nativeLibDir: String
        ): Long

        @JvmStatic
        private external fun nativeDescribe(
            handle: Long
        ): String

        @JvmStatic
        private external fun nativeBenchmark(
            handle: Long,
            warmups: Int,
            iterations: Int
        ): String

        @JvmStatic
        private external fun nativeRunImage(
            handle: Long,
            input: FloatArray
        ): FloatArray

        @JvmStatic
        private external fun nativeSetVerboseLogging(
            handle: Long,
            enabled: Boolean
        )

        @JvmStatic
        private external fun nativeClose(
            handle: Long
        )
    }
}
