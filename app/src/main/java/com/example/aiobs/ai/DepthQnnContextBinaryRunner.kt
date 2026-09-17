package com.example.aiobs.ai

import android.content.Context
import android.util.Log

/**
 * Standalone Qualcomm QNN Context Binary runner for the S23 / SM8550 build.
 *
 * This validation runner is independent from live tracking and
 * TfliteAcceleration.
 *
 * It exists only to prove that the AI Hub generated Context Binary can be
 * loaded and executed directly through QNN / HTP in the Android process
 * before replacing the live TFLite path.
 */
class DepthQnnContextBinaryRunner(
    context: Context,
    private val assetName: String = MODEL_ASSET
) : AutoCloseable {

    data class BenchmarkResult(
        val report: String
    )

    private val appContext =
        context.applicationContext

    private var handle: Long = 0L

    init {
        Log.i(
            TAG,
            "Runner init BEGIN asset=$assetName"
        )

        System.loadLibrary(
            NATIVE_LIBRARY
        )

        Log.i(
            TAG,
            "Native library loaded: $NATIVE_LIBRARY"
        )

        handle =
            nativeCreate(
                appContext.assets,
                assetName,
                appContext.applicationInfo.nativeLibraryDir
            )

        Log.i(
            TAG,
            "Runner init nativeCreate RETURN handle=$handle"
        )
    }

    fun isReady(): Boolean {
        val ready =
            handle != 0L

        Log.i(
            TAG,
            "isReady=$ready handle=$handle"
        )

        return ready
    }

    fun describe(): String {

        Log.i(
            TAG,
            "describe ENTER handle=$handle"
        )

        checkReady()

        val result =
            nativeDescribe(handle)

        Log.i(
            TAG,
            "describe RETURN length=${result.length}"
        )

        return result
    }

    /**
     * Runs a deterministic HTP-only benchmark.
     *
     * Timing reported by the native layer surrounds QNN graph execution,
     * excluding Kotlin / Java bookkeeping.
     */
    fun benchmark(
        warmups: Int = 2,
        iterations: Int = 5
    ): BenchmarkResult {

        Log.i(
            TAG,
            "benchmark ENTER handle=$handle warmups=$warmups iterations=$iterations"
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
            "benchmark BEFORE nativeBenchmark handle=$handle"
        )

        val report =
            nativeBenchmark(
                handle,
                warmups,
                iterations
            )

        Log.i(
            TAG,
            "benchmark AFTER nativeBenchmark reportLength=${report.length}"
        )

        Log.i(
            TAG,
            "benchmark REPORT:\n$report"
        )

        return BenchmarkResult(
            report = report
        )
    }

    /**
     * Executes one caller-supplied NCHW float32 tensor:
     *
     * [1, 3, 518, 518]
     */
    fun run(
        inputNchwFloat32: FloatArray
    ): FloatArray {

        Log.i(
            TAG,
            "run ENTER handle=${handle} inputSize=${inputNchwFloat32.size}"
        )

        checkReady()

        require(
            inputNchwFloat32.size ==
                    INPUT_ELEMENTS
        ) {
            "Expected $INPUT_ELEMENTS float values, got ${inputNchwFloat32.size}"
        }

        val result =
            nativeRun(
                handle,
                inputNchwFloat32
            )

        Log.i(
            TAG,
            "run RETURN outputSize=${result.size}"
        )

        return result
    }

    override fun close() {

        val current =
            handle

        Log.i(
            TAG,
            "close ENTER handle=$current"
        )

        if (current != 0L) {

            try {

                nativeClose(
                    current
                )

            } catch (t: Throwable) {

                Log.w(
                    TAG,
                    "Failed to close QNN context binary runner",
                    t
                )

            } finally {

                handle = 0L
            }
        }

        Log.i(
            TAG,
            "close END"
        )
    }

    private fun checkReady() {

        if (handle == 0L) {

            Log.e(
                TAG,
                "checkReady FAILED: handle == 0"
            )

            error(
                "Depth QNN Context Binary runner is not initialized"
            )
        }
    }

    private companion object {

        private const val TAG =
            "DepthQnnCtxBin"

        private const val NATIVE_LIBRARY =
            "aiobs_depth_qnn_context"

        private const val MODEL_ASSET =
            "depth_anything_v2_s23_w8a16.bin"

        private const val INPUT_ELEMENTS =
            3 * 518 * 518

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
        private external fun nativeRun(
            handle: Long,
            inputNchwFloat32: FloatArray
        ): FloatArray

        @JvmStatic
        private external fun nativeClose(
            handle: Long
        )
    }
}