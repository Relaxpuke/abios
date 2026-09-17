package com.example.aiobs.ai

import android.content.Context
import android.util.Log

/**
 * Global Qualcomm HTP performance controller.
 *
 * The manager owns one process-wide HTP performance vote shared by the live
 * AI pipeline and explicitly validated QNN models.
 */
object HtpPerformanceManager {

    private const val TAG = "HtpPerformance"

    private const val NATIVE_LIBRARY =
        "aiobs_htp_performance"

    @Volatile
    private var loaded = false

    @Volatile
    private var active = false

    @Synchronized
    private fun ensureNativeLoaded() {

        if (loaded) {
            return
        }

        System.loadLibrary(
            NATIVE_LIBRARY
        )

        loaded = true

        Log.i(
            TAG,
            "Native library loaded: $NATIVE_LIBRARY"
        )
    }

    /**
     * Acquire the global HTP performance vote.
     *
     * Safe to call repeatedly.
     *
     * Multiple calls while the vote is already active do not create another
     * power configuration.
     */
    @Synchronized
    fun acquire(
        context: Context
    ): Boolean {

        ensureNativeLoaded()

        if (active) {

            Log.i(
                TAG,
                "HTP performance already ACTIVE"
            )

            return true
        }

        val appContext =
            context.applicationContext

        val nativeLibDir =
            appContext.applicationInfo.nativeLibraryDir

        Log.i(
            TAG,
            "HTP performance acquire BEGIN " +
                    "nativeLibDir=$nativeLibDir"
        )

        val ok =
            try {
                nativeAcquire(
                    nativeLibDir
                )
            } catch (t: Throwable) {

                Log.e(
                    TAG,
                    "HTP performance acquire failed",
                    t
                )

                false
            }

        active = ok

        Log.i(
            TAG,
            "HTP performance acquire END active=$active"
        )

        return ok
    }

    /**
     * Apply the currently selected application performance mode.
     *
     * IMPORTANT:
     * The native implementation already has a validated HTP performance profile.
     * We keep that native profile unchanged:
     *   - DEFAULT: release the global vote and return control to the system.
     *   - Any non-default mode: request the existing validated HTP performance vote.
     *
     * The native mode integer is still forwarded so the JNI/native boundary remains
     * explicit and future per-mode native tuning can be added without changing
     * the Activity/TFLite call sites.
     */
    @Synchronized
    fun setPerformanceMode(
        context: Context,
        mode: TfliteAcceleration.QnnPerformanceMode
    ): Boolean {
        ensureNativeLoaded()

        val appContext = context.applicationContext

        if (mode == TfliteAcceleration.QnnPerformanceMode.DEFAULT) {
            release()
            Log.i(TAG, "HTP performance mode -> DEFAULT (system controlled)")
            return true
        }

        val nativeLibDir =
            appContext.applicationInfo.nativeLibraryDir

        val modeValue = mode.ordinal

        val ok = try {
            nativeSetPerformanceMode(
                nativeLibDir,
                modeValue
            )
        } catch (t: Throwable) {
            Log.e(
                TAG,
                "HTP performance mode apply failed mode=$mode",
                t
            )
            false
        }

        active = ok || active

        Log.i(
            TAG,
            "HTP performance mode -> $mode " +
                    "nativeMode=$modeValue active=$active"
        )

        return ok
    }

    /**
     * Release the global HTP performance vote.
     *
     * After this call HTP returns to the normal system-controlled state.
     */
    @Synchronized
    fun release() {

        if (!loaded) {
            return
        }

        if (!active) {

            Log.i(
                TAG,
                "HTP performance already RELEASED"
            )

            return
        }

        Log.i(
            TAG,
            "HTP performance release BEGIN"
        )

        try {

            nativeRelease()

        } catch (t: Throwable) {

            Log.e(
                TAG,
                "HTP performance release failed",
                t
            )
        } finally {

            active = false
        }

        Log.i(
            TAG,
            "HTP performance release END"
        )
    }


    private external fun nativeAcquire(
        nativeLibDir: String
    ): Boolean

    private external fun nativeRelease()

    private external fun nativeSetPerformanceMode(
        nativeLibDir: String,
        mode: Int
    ): Boolean

}