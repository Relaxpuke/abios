package com.example.aiobs

import android.app.Application
import android.util.Log
import com.example.aiobs.ai.HtpPerformanceManager

/**
 * AIOBS global application.
 *
 * HTP performance vote is acquired once when the application process starts,
 * before MainActivity / SigLIP / XFeat / YOLO / Depth initialize.
 *
 * The vote is process-wide and is intentionally not owned by any individual
 * model runner.
 */
class AIOBSApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        Log.i(TAG, "AIOBSApplication.onCreate BEGIN")

        try {
            HtpPerformanceManager.acquire(this)

            Log.i(
                TAG,
                "HTP performance manager acquire completed"
            )
        } catch (t: Throwable) {
            Log.e(
                TAG,
                "HTP performance manager acquire failed",
                t
            )
        }

        Log.i(TAG, "AIOBSApplication.onCreate END")
    }

    override fun onTerminate() {
        // onTerminate() is normally not called on physical Android devices.
        //
        // Do not rely on it for releasing the performance vote.
        // HtpPerformanceManager should remain process-owned.
        super.onTerminate()
    }

    companion object {
        private const val TAG = "AIOBSApplication"
    }
}