package com.example.aiobs.camera

import android.graphics.Bitmap
import android.os.SystemClock
import com.example.aiobs.ai.Yuv420FrameSnapshot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * YUV_420_888 snapshot -> reusable ARGB_8888 Bitmap.
 *
 * Performance notes:
 *
 * 1. Source/output dimension mappings are precomputed and reused while the
 *    Camera2 stream size stays unchanged.
 * 2. The output bitmap and output pixel array are reused across frames.
 * 3. Bitmap.setPixels() is called once for the whole image instead of once
 *    per row.
 * 4. The inner pixel loop avoids division and repeated bounds checks.
 *
 * The conversion remains a CPU YUV->ARGB path so model behavior does not
 * change. The optimization is strictly about reducing per-frame CPU work and
 * allocation pressure.
 */
class Yuv420ToBitmapConverter(
    private val maxLongSide: Int = 640
) {

    private var configuredSourceWidth = 0
    private var configuredSourceHeight = 0
    private var configuredTargetWidth = 0
    private var configuredTargetHeight = 0

    private var srcXMap = IntArray(0)
    private var uvXMap = IntArray(0)
    private var yRowMap = IntArray(0)
    private var uvYMap = IntArray(0)

    private var outputBitmap: Bitmap? = null
    private var outputPixels = IntArray(0)

    @Volatile
    private var lastConvertMs = 0f

    @Volatile
    private var lastNativeConvertMs = 0f

    private var probeFrameCounter = 0L

    fun convert(
        frame: Yuv420FrameSnapshot
    ): Bitmap {

        val startNs =
            SystemClock.elapsedRealtimeNanos()

        val sourceWidth =
            frame.width.coerceAtLeast(1)

        val sourceHeight =
            frame.height.coerceAtLeast(1)

        ensureConfiguration(
            sourceWidth,
            sourceHeight
        )

        val targetWidth =
            configuredTargetWidth

        val targetHeight =
            configuredTargetHeight

        val bitmap =
            ensureOutputBitmap(
                targetWidth,
                targetHeight
            )

        val pixels =
            outputPixels

        if (NativeYuv420Converter.isAvailable()) {
            val nativeStartNs =
                SystemClock.elapsedRealtimeNanos()

            val nativeOk =
                NativeYuv420Converter.convert(
                    bitmap = bitmap,
                    yData = frame.yData,
                    uData = frame.uData,
                    vData = frame.vData,
                    yRowStride = frame.yRowStride,
                    yPixelStride = frame.yPixelStride,
                    uRowStride = frame.uRowStride,
                    uPixelStride = frame.uPixelStride,
                    vRowStride = frame.vRowStride,
                    vPixelStride = frame.vPixelStride,
                    targetWidth = targetWidth,
                    targetHeight = targetHeight,
                    srcXMap = srcXMap,
                    uvXMap = uvXMap,
                    yRowMap = yRowMap,
                    uvYMap = uvYMap
                )

            if (nativeOk) {
                val endNs =
                    SystemClock.elapsedRealtimeNanos()

                lastNativeConvertMs =
                    (endNs - nativeStartNs) / 1_000_000f

                lastConvertMs =
                    (endNs - startNs) / 1_000_000f

                val probeFrame =
                    ++probeFrameCounter

                if (probeFrame == 1L || probeFrame % PROBE_LOG_INTERVAL == 0L) {
                    android.util.Log.i(
                        TAG,
                        "YUV_NATIVE " +
                                "NATIVE=${"%.3f".format(lastNativeConvertMs)}ms " +
                                "TOTAL=${"%.3f".format(lastConvertMs)}ms " +
                                "SIZE=${targetWidth}x${targetHeight}"
                    )
                }

                return bitmap
            }
        }

        var pixelIndex = 0

        var outY = 0

        while (outY < targetHeight) {

            val srcY =
                yRowMap[outY]

            val uvY =
                uvYMap[outY]

            val yRowOffset =
                srcY * frame.yRowStride

            val uRowOffset =
                uvY * frame.uRowStride

            val vRowOffset =
                uvY * frame.vRowStride

            var outX = 0

            while (outX < targetWidth) {

                val srcX =
                    srcXMap[outX]

                val uvX =
                    uvXMap[outX]

                val yIndex =
                    yRowOffset +
                            srcX * frame.yPixelStride

                val uIndex =
                    uRowOffset +
                            uvX * frame.uPixelStride

                val vIndex =
                    vRowOffset +
                            uvX * frame.vPixelStride

                val y =
                    frame.yData[yIndex].toInt() and 0xFF

                val u =
                    frame.uData[uIndex].toInt() and 0xFF

                val v =
                    frame.vData[vIndex].toInt() and 0xFF

                pixels[pixelIndex++] =
                    yuvToArgb(
                        y,
                        u,
                        v
                    )

                outX++
            }

            outY++
        }

        /*
         * One bulk upload instead of targetHeight setPixels() calls.
         */
        bitmap.setPixels(
            pixels,
            0,
            targetWidth,
            0,
            0,
            targetWidth,
            targetHeight
        )

        lastConvertMs =
            (
                SystemClock.elapsedRealtimeNanos() -
                        startNs
                ) / 1_000_000f

        return bitmap
    }

    /**
     * Time of the most recent YUV->Bitmap conversion.
     */
    private fun ensureConfiguration(
        sourceWidth: Int,
        sourceHeight: Int
    ) {

        if (
            sourceWidth == configuredSourceWidth &&
            sourceHeight == configuredSourceHeight
        ) {
            return
        }

        val scale =
            min(
                1f,
                maxLongSide.toFloat() /
                        max(
                            sourceWidth,
                            sourceHeight
                        ).toFloat()
            )

        val targetWidth =
            max(
                2,
                (
                        sourceWidth * scale
                        ).roundToInt()
            )

        val targetHeight =
            max(
                2,
                (
                        sourceHeight * scale
                        ).roundToInt()
            )

        configuredSourceWidth =
            sourceWidth

        configuredSourceHeight =
            sourceHeight

        configuredTargetWidth =
            targetWidth

        configuredTargetHeight =
            targetHeight

        srcXMap =
            IntArray(
                targetWidth
            )

        uvXMap =
            IntArray(
                targetWidth
            )

        yRowMap =
            IntArray(
                targetHeight
            )

        uvYMap =
            IntArray(
                targetHeight
            )

        val xDen =
            max(
                1,
                targetWidth - 1
            )

        var outX = 0

        while (outX < targetWidth) {

            val srcX =
                (
                        outX *
                                (sourceWidth - 1)
                        ) / xDen

            srcXMap[outX] =
                srcX

            uvXMap[outX] =
                srcX shr 1

            outX++
        }

        val yDen =
            max(
                1,
                targetHeight - 1
            )

        var outY = 0

        while (outY < targetHeight) {

            val srcY =
                (
                        outY *
                                (sourceHeight - 1)
                        ) / yDen

            yRowMap[outY] =
                srcY

            uvYMap[outY] =
                srcY shr 1

            outY++
        }

        /*
         * Output storage is resized only when the stream dimensions change.
         */
        if (
            outputPixels.size !=
            targetWidth * targetHeight
        ) {
            outputPixels =
                IntArray(
                    targetWidth * targetHeight
                )
        }

        val existing =
            outputBitmap

        if (
            existing == null ||
            existing.isRecycled ||
            existing.width != targetWidth ||
            existing.height != targetHeight
        ) {

            try {
                existing?.recycle()
            } catch (_: Throwable) {
            }

            outputBitmap =
                Bitmap.createBitmap(
                    targetWidth,
                    targetHeight,
                    Bitmap.Config.ARGB_8888
                )
        }
    }

    private fun ensureOutputBitmap(
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {

        val current =
            outputBitmap

        if (
            current != null &&
            !current.isRecycled &&
            current.width == targetWidth &&
            current.height == targetHeight
        ) {
            return current
        }

        outputBitmap =
            Bitmap.createBitmap(
                targetWidth,
                targetHeight,
                Bitmap.Config.ARGB_8888
            )

        return outputBitmap!!
    }

    private companion object {
        const val TAG = "Yuv420ToBitmapConverterV30"
        const val PROBE_LOG_INTERVAL = 30L
    }

    private fun yuvToArgb(
        y: Int,
        u: Int,
        v: Int
    ): Int {

        val c =
            max(
                0,
                y - 16
            )

        val d =
            u - 128

        val e =
            v - 128

        var r =
            (
                    298 * c +
                            409 * e +
                            128
                    ) shr 8

        var g =
            (
                    298 * c -
                            100 * d -
                            208 * e +
                            128
                    ) shr 8

        var b =
            (
                    298 * c +
                            516 * d +
                            128
                    ) shr 8

        r =
            min(
                255,
                max(0, r)
            )

        g =
            min(
                255,
                max(0, g)
            )

        b =
            min(
                255,
                max(0, b)
            )

        return (0xFF shl 24) or
                (r shl 16) or
                (g shl 8) or
                b
    }
}
