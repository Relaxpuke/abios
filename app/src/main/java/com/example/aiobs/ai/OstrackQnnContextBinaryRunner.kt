package com.example.aiobs.ai

import android.content.Context
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Direct QNN HTP runner for the fixed-shape OSTrack-256 Dense 3-output context.
 *
 * The context binary is generated offline for SM8550 and is executed directly
 * through libQnnHtp.so. This is intentionally independent from TfliteAcceleration
 * and the LiteRT/QNN delegate path.
 */
class OstrackQnnContextBinaryRunner private constructor(
    context: Context,
    private val assetName: String
) : AutoCloseable {

    private val closed = AtomicBoolean(false)
    private var handle: Long = 0L

    init {
        val appContext = context.applicationContext
        handle = nativeCreate(
            appContext.assets,
            assetName,
            appContext.applicationInfo.nativeLibraryDir
        )
        if (handle == 0L) {
            throw IllegalStateException(
                "Failed to create OSTrack QNN context runner: $assetName"
            )
        }

        Log.i(TAG, nativeDescribe(handle))
    }

    fun isReady(): Boolean = !closed.get() && handle != 0L

    /**
     * Input tensors are logical FP32 NCHW arrays. Native code converts them to
     * the exact FP16/layout advertised by the SM8550 context and returns:
     *
     *   [ score_map(256), size_map(512), offset_map(512) ]
     *
     * Native code maps graph outputs by tensor name rather than relying on
     * context output order, because QAIRT may enumerate them as offset/score/size.
     */
    @Synchronized
    fun infer(template: FloatArray, search: FloatArray): FloatArray {
        check(isReady()) { "OSTrack QNN runner is closed" }
        require(template.size == TEMPLATE_ELEMENTS) {
            "template size=${template.size}, expected=$TEMPLATE_ELEMENTS"
        }
        require(search.size == SEARCH_ELEMENTS) {
            "search size=${search.size}, expected=$SEARCH_ELEMENTS"
        }
        return nativeRun(handle, template, search)
            ?: throw IllegalStateException("OSTrack QNN graph execution failed")
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            val h = handle
            handle = 0L
            if (h != 0L) nativeClose(h)
        }
    }

    companion object {
        private const val TAG = "OstrackQnnCtx"

        const val DEFAULT_ASSET =
            "ostrack_256_fp16_sm8550_3out.serialized.SM8550.bin"

        const val TEMPLATE_SIZE = 128
        const val SEARCH_SIZE = 256
        const val CHANNELS = 3
        const val TEMPLATE_ELEMENTS = CHANNELS * TEMPLATE_SIZE * TEMPLATE_SIZE
        const val SEARCH_ELEMENTS = CHANNELS * SEARCH_SIZE * SEARCH_SIZE
        const val OUTPUT_ELEMENTS = 256 + 512 + 512

        private var libraryLoaded = false

        @Synchronized
        private fun ensureNativeLoaded() {
            if (libraryLoaded) return
            System.loadLibrary("aiobs_ostrack_qnn_context")
            libraryLoaded = true
        }

        fun tryCreate(
            context: Context,
            assetName: String = DEFAULT_ASSET
        ): OstrackQnnContextBinaryRunner? {
            return try {
                context.assets.open(assetName).use { }
                ensureNativeLoaded()
                OstrackQnnContextBinaryRunner(context, assetName)
            } catch (t: Throwable) {
                Log.e(TAG, "OSTrack QNN runner unavailable asset=$assetName", t)
                null
            }
        }

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
            template: FloatArray,
            search: FloatArray
        ): FloatArray?

        @JvmStatic
        private external fun nativeClose(handle: Long)
    }
}
