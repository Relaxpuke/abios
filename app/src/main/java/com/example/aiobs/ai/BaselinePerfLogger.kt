package com.example.aiobs.ai

import android.util.Log
import java.util.Locale

/** Single-purpose performance logger used only for the pre-optimization baseline. */
object BaselinePerfLogger {
    private const val TAG = "BaselinePerf"

    @JvmStatic
    fun model(
        name: String,
        prepMs: Float,
        preprocessMs: Float,
        inferMs: Float,
        postprocessMs: Float,
        totalMs: Float
    ) {
        Log.i(
            TAG,
            "[BASELINE][$name] prep=${fmt(prepMs)}ms " +
                "preprocess=${fmt(preprocessMs)}ms " +
                "infer=${fmt(inferMs)}ms " +
                "postprocess=${fmt(postprocessMs)}ms " +
                "total=${fmt(totalMs)}ms"
        )
    }

    @JvmStatic
    fun framePrep(totalMs: Float) {
        Log.i(TAG, "[BASELINE][FramePrep] yuvToBitmap=${fmt(totalMs)}ms total=${fmt(totalMs)}ms")
    }

    @JvmStatic
    fun frame(seq: Long, totalMs: Float, state: String) {
        Log.i(
            TAG,
            "[BASELINE][FRAME] seq=$seq total=${fmt(totalMs)}ms state=$state"
        )
    }

    @JvmStatic
    fun xfeatMatch(totalMs: Float) {
        Log.i(TAG, "[BASELINE][XFeatMatch] total=${fmt(totalMs)}ms")
    }

    private fun fmt(value: Float): String = String.format(Locale.US, "%.3f", value)
}
