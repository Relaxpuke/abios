package com.example.aiobs.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Android-side OSTrack-256 wrapper.
 *
 * Locked frame flow:
 *
 *   current tracked box
 *       ->
 *   template/search square crop
 *       ->
 *   ImageNet normalization
 *       ->
 *   direct QNN W8A16 OSTrack context
 *       ->
 *   center-head decode
 *       ->
 *   search-crop -> image-space mapping
 *       ->
 *   normalized tracked box
 *
 * Network fixed shape:
 *
 *   template [1,3,128,128]
 *   search   [1,3,256,256]
 *
 * Native QNN runner accepts logical FP32 NCHW and converts it to the
 * actual tensor datatype/layout required by the Context Binary.
 *
 * Native output contract:
 *
 *   [0 .. 255]      score map   [16,16]
 *   [256 .. 767]    size map    [2,16,16]
 *   [768 .. 1279]   offset map  [2,16,16]
 *
 * All three are returned by native code in logical NCHW order.
 */
class OstrackQnnTracker private constructor(
    context: Context,
    private val minValidScore: Float,
    private val assetName: String
) : AutoCloseable {

    data class Result(
        val box: RectF,
        val score: Float,
        val inferenceMs: Float,
        val searchFactor: Float
    )

    private val appContext = context.applicationContext

    private val runner =
        OstrackQnnContextBinaryRunner.tryCreate(appContext, assetName)
            ?: throw IllegalStateException(
                "OSTrack QNN context binary unavailable"
            )

    private val templateBitmap = Bitmap.createBitmap(
        TEMPLATE_SIZE,
        TEMPLATE_SIZE,
        Bitmap.Config.ARGB_8888
    )

    private val searchBitmap = Bitmap.createBitmap(
        SEARCH_SIZE,
        SEARCH_SIZE,
        Bitmap.Config.ARGB_8888
    )

    private val templateCanvas = Canvas(templateBitmap)
    private val searchCanvas = Canvas(searchBitmap)

    private val paint =
        Paint(Paint.FILTER_BITMAP_FLAG)

    private val templateInput =
        FloatArray(
            OstrackQnnContextBinaryRunner.TEMPLATE_ELEMENTS
        )

    private val searchInput =
        FloatArray(
            OstrackQnnContextBinaryRunner.SEARCH_ELEMENTS
        )

    private var templateReady = false
    private var destroyed = false

    private var trackCallCount = 0L
    private var templateCropDebug: SquareCrop? = null

    @Volatile
    private var lastInferenceMsValue = 0f
    @Volatile private var lastPreprocessMsValue = 0f
    @Volatile private var lastPostprocessMsValue = 0f

    /*
     * Last search crop used by the current inference.
     *
     * Coordinates are in original bitmap pixels.
     * crop.left/top may be outside the image when padding is required.
     */
    private var currentCrop =
        SquareCrop(
            left = 0f,
            top = 0f,
            size = 1f
        )

    init {
        Log.i(
            TAG,
            "OSTrack QNN READY " +
                    "asset=$assetName " +
                    "template=[1,3,128,128] " +
                    "search=[1,3,256,256] " +
                    "backend=QNN/HTP SM8550"
        )
    }

    fun isReady(): Boolean =
        !destroyed && runner.isReady()

    /**
     * Run one tracking step.
     *
     * template/search are prepared as normalized FP32 NCHW inputs.
     */
    @Synchronized
    fun track(
        bitmap: Bitmap,
        currentBox: RectF
    ): Result? {
        if (!isReady() || bitmap.width < 8 || bitmap.height < 8) return null

        trackCallCount++

        val totalStart = SystemClock.elapsedRealtimeNanos()
        val preprocessStart = SystemClock.elapsedRealtimeNanos()

        val searchFactor = SEARCH_FACTOR_BASE

        try {

            /*
             * Template is intentionally kept fixed until
             * resetTemplate() is called.
             */
            if (!templateReady) {
                val templateCrop = prepareTemplate(bitmap, currentBox)
                templateCropDebug = templateCrop
                templateReady = true

                Log.i(
                    TAG,
                    "INPUT TEMPLATE " +
                            "frame=$trackCallCount " +
                            "bitmap=${bitmap.width}x${bitmap.height} " +
                            "box=$currentBox " +
                            "boxPx=${normalizedToPixels(currentBox, bitmap.width, bitmap.height)} " +
                            "crop=$templateCrop " +
                            "templateStats=${inputStats(templateInput)}"
                )
            }

            prepareSearch(bitmap, currentBox, searchFactor)
            lastPreprocessMsValue = elapsedMs(preprocessStart)

            if (trackCallCount == 1L || trackCallCount % 10L == 0L) {
                Log.i(
                    TAG,
                    "INPUT SEARCH " +
                            "frame=$trackCallCount " +
                            "bitmap=${bitmap.width}x${bitmap.height} " +
                            "box=$currentBox " +
                            "boxPx=${normalizedToPixels(currentBox, bitmap.width, bitmap.height)} " +
                            "crop=$currentCrop " +
                            "factor=$searchFactor " +
                            "searchStats=${inputStats(searchInput)}"
                )
            }

            val inferenceStart =
                SystemClock.elapsedRealtimeNanos()

            val raw =
                runner.infer(
                    templateInput,
                    searchInput
                )

            lastInferenceMsValue = elapsedMs(inferenceStart)
            val postprocessStart = SystemClock.elapsedRealtimeNanos()
            
            if (raw.size != OUTPUT_ELEMENTS) {

                lastPostprocessMsValue = elapsedMs(postprocessStart)
                Log.e(
                    TAG,
                    "Unexpected OSTrack output size=" +
                            "${raw.size}, " +
                            "expected=$OUTPUT_ELEMENTS"
                )

                return null
            }

            /*
             * Decode everything from one common peak.
             *
             * This avoids having one function determine score
             * and another independently determine the location.
             */
            val decoded =
                decodeCenterHead(raw)

            if (decoded == null) {
                lastPostprocessMsValue = elapsedMs(postprocessStart)
                return null
            }

            // OSTrack itself is the LOCKED-stage validity judge.
            // A weak response is treated as a tracker miss so the existing
            // ObjectTracker miss counter can decide when to transition to LOST.
            val thresholdPass = decoded.score >= minValidScore

            Log.i(
                TAG,
                "[OSTRACK_THRESHOLD_DEBUG] " +
                        "frame=$trackCallCount " +
                        "score=${"%.6f".format(decoded.score)} " +
                        "threshold=${"%.3f".format(minValidScore)} " +
                        "pass=$thresholdPass " +
                        "decodedBox=${decoded.box}"
            )

            if (!thresholdPass) {
                Log.w(
                    TAG,
                    "OSTRACK LOW SCORE " +
                            "frame=$trackCallCount " +
                            "score=${"%.6f".format(decoded.score)} " +
                            "threshold=${"%.3f".format(minValidScore)} " +
                            "pass=false " +
                            "action=MISS"
                )
                return null
            }

            Log.i(
                TAG,
                "[OSTRACK_THRESHOLD_ACCEPT] " +
                        "frame=$trackCallCount " +
                        "score=${"%.6f".format(decoded.score)} " +
                        "threshold=${"%.3f".format(minValidScore)} " +
                        "pass=true " +
                        "action=ACCEPT"
            )

            val boxInSearch =
                decoded.box

            /*
             * Convert the normalized search-crop coordinates
             * back into normalized original-image coordinates.
             */
            val boxInImage =
                mapSearchBoxToImage(
                    boxInSearch,
                    currentCrop
                )

            val sane =
                sanitizeNormalizedBox(
                    boxInImage
                )

            if (sane.width() <= 0f ||
                sane.height() <= 0f
            ) {
                lastPostprocessMsValue = elapsedMs(postprocessStart)
                Log.w(
                    TAG,
                    "Decoded invalid box: " +
                            "search=$boxInSearch " +
                            "image=$boxInImage " +
                            "crop=$currentCrop"
                )

                return null
            }

            lastPostprocessMsValue = elapsedMs(postprocessStart)

            return Result(
                box = sane,
                score = decoded.score.coerceIn(
                    0f,
                    1f
                ),
                inferenceMs = lastInferenceMsValue,
                searchFactor = searchFactor
            )

        } catch (t: Throwable) {

            lastInferenceMsValue =
                elapsedMs(totalStart)

            Log.e(
                TAG,
                "OSTrack track failed",
                t
            )

            return null
        }
    }

    fun resetTemplate() {
        templateReady = false
    }

    fun lastInferenceMs(): Float = lastInferenceMsValue

    fun lastPreprocessMs(): Float = lastPreprocessMsValue

    fun lastPostprocessMs(): Float = lastPostprocessMsValue

    override fun close() {

        if (destroyed) {
            return
        }

        destroyed = true

        try {
            runner.close()
        } catch (_: Throwable) {
        }

        try {
            templateBitmap.recycle()
        } catch (_: Throwable) {
        }

        try {
            searchBitmap.recycle()
        } catch (_: Throwable) {
        }
    }

    /*
     * ------------------------------------------------------------------------
     * Input preparation
     * ------------------------------------------------------------------------
     */

    private fun prepareTemplate(bitmap: Bitmap, box: RectF): SquareCrop {
        val square = makeSquareCrop(
            box = normalizedToPixels(box, bitmap.width, bitmap.height),
            factor = TEMPLATE_FACTOR,
            width = bitmap.width,
            height = bitmap.height
        )

        drawPaddedCrop(
            source = bitmap,
            crop = square,
            outBitmap = templateBitmap,
            canvas = templateCanvas,
            meanColor = MEAN_COLOR
        )

        if (!NativeRgbInputPacker.packBitmapImageNetNchwFloat32(
                templateBitmap, templateInput, TEMPLATE_SIZE, TEMPLATE_SIZE
            )) {
            bitmapToNchwNormalizedFallback(templateBitmap, templateInput)
        }

        return square
    }

    private fun inputStats(values: FloatArray): String {
        if (values.isEmpty()) return "empty"

        var min = Float.POSITIVE_INFINITY
        var max = Float.NEGATIVE_INFINITY
        var sum = 0.0

        for (v in values) {
            if (v < min) min = v
            if (v > max) max = v
            sum += v.toDouble()
        }

        val mean = sum / values.size

        return "min=${"%.5f".format(min)} " +
                "max=${"%.5f".format(max)} " +
                "mean=${"%.5f".format(mean)}"
    }

    private fun prepareSearch(
        bitmap: Bitmap,
        box: RectF,
        searchFactor: Float
    ) {

        currentCrop =
            makeSquareCrop(
                box = normalizedToPixels(
                    box,
                    bitmap.width,
                    bitmap.height
                ),
                factor = searchFactor,
                width = bitmap.width,
                height = bitmap.height
            )

        drawPaddedCrop(
            source = bitmap,
            crop = currentCrop,
            outBitmap = searchBitmap,
            canvas = searchCanvas,
            meanColor = MEAN_COLOR
        )

        if (!NativeRgbInputPacker.packBitmapImageNetNchwFloat32(
                searchBitmap, searchInput, SEARCH_SIZE, SEARCH_SIZE
            )) {
            bitmapToNchwNormalizedFallback(searchBitmap, searchInput)
        }
    }

    /*
     * ------------------------------------------------------------------------
     * OSTrack center-head decode
     * ------------------------------------------------------------------------
     *
     * Output layout:
     *
     *   score:
     *       [256]
     *
     *   size:
     *       [width  256]
     *       [height 256]
     *
     *   offset:
     *       [x 256]
     *       [y 256]
     *
     * For a score peak at grid (x, y):
     *
     *   centerX = (x + offsetX) / 16
     *   centerY = (y + offsetY) / 16
     *
     * sizeW/sizeH are already normalized to search-image coordinates.
     *
     * IMPORTANT:
     *
     * The decoded center is NOT the left/top corner.
     * The bounding box must therefore be:
     *
     *   left   = centerX - sizeW/2
     *   top    = centerY - sizeH/2
     *   right  = centerX + sizeW/2
     *   bottom = centerY + sizeH/2
     *
     * This is the critical correction relative to the previous version.
     */
    private fun decodeScoreIndex(raw: FloatArray): Int {
        var bestIndex = -1
        var bestScore = -Float.MAX_VALUE

        for (i in 0 until SCORE_ELEMENTS) {
            val score = raw[SCORE_OFFSET + i]

            if (!score.isFinite()) {
                continue
            }

            if (score > bestScore) {
                bestScore = score
                bestIndex = i
            }
        }

        return bestIndex
    }

    private fun decodeCenterHead(
        raw: FloatArray
    ): Decoded? {

        if (raw.size < OUTPUT_ELEMENTS) {
            return null
        }

        val bestIndex = decodeScoreIndex(raw)

        var bestScore = -Float.MAX_VALUE
        if (bestIndex >= 0) {
            bestScore = raw[SCORE_OFFSET + bestIndex]
        }

        if (bestIndex < 0 || !bestScore.isFinite()) {
            return null
        }

        val y =
            bestIndex / SEARCH_FEAT_SIZE

        val x =
            bestIndex % SEARCH_FEAT_SIZE

        /*
         * Native code already reordered the tensors to logical NCHW:
         *
         * size:
         *   channel 0 = width
         *   channel 1 = height
         *
         * offset:
         *   channel 0 = x
         *   channel 1 = y
         */
        val sizeW =
            raw[
                SIZE_OFFSET +
                        bestIndex
            ]

        val sizeH =
            raw[
                SIZE_OFFSET +
                        SEARCH_FEAT_ELEMENTS +
                        bestIndex
            ]

        val offX =
            raw[
                OFFSET_OFFSET +
                        bestIndex
            ]

        val offY =
            raw[
                OFFSET_OFFSET +
                        SEARCH_FEAT_ELEMENTS +
                        bestIndex
            ]

        if (!sizeW.isFinite() ||
            !sizeH.isFinite() ||
            !offX.isFinite() ||
            !offY.isFinite()
        ) {
            return null
        }

        /*
         * Grid cell center decoded into normalized
         * search-image coordinates.
         */
        val centerX =
            (
                    x.toFloat() +
                            offX
                    ) / SEARCH_FEAT_SIZE.toFloat()

        val centerY =
            (
                    y.toFloat() +
                            offY
                    ) / SEARCH_FEAT_SIZE.toFloat()

        /*
         * OSTrack size output is normalized width/height.
         *
         * Convert center + size to standard xyxy box.
         */
        val halfW =
            sizeW * 0.5f

        val halfH =
            sizeH * 0.5f

        val left =
            centerX - halfW

        val top =
            centerY - halfH

        val right =
            centerX + halfW

        val bottom =
            centerY + halfH

        val box =
            RectF(
                left,
                top,
                right,
                bottom
            )

        /*
         * Keep this diagnostic log because it lets us directly compare
         * the native QNN values against the Android-side interpretation.
         */
        Log.d(
            TAG,
            "decode peak " +
                    "grid=($y,$x) " +
                    "score=$bestScore " +
                    "size=($sizeW,$sizeH) " +
                    "offset=($offX,$offY) " +
                    "center=($centerX,$centerY) " +
                    "box=$box"
        )

        return Decoded(
            box = box,
            score = bestScore
        )
    }

    /*
     * ------------------------------------------------------------------------
     * Search crop -> original image
     * ------------------------------------------------------------------------
     */

    private fun mapSearchBoxToImage(
        box: RectF,
        crop: SquareCrop
    ): RectF {

        val left =
            crop.left +
                    box.left * crop.size

        val top =
            crop.top +
                    box.top * crop.size

        val right =
            crop.left +
                    box.right * crop.size

        val bottom =
            crop.top +
                    box.bottom * crop.size

        val w =
            crop.bitmapWidth.coerceAtLeast(1)

        val h =
            crop.bitmapHeight.coerceAtLeast(1)

        return RectF(
            left / w.toFloat(),
            top / h.toFloat(),
            right / w.toFloat(),
            bottom / h.toFloat()
        )
    }

    /*
     * ------------------------------------------------------------------------
     * Box utilities
     * ------------------------------------------------------------------------
     */

    private fun sanitizeNormalizedBox(
        box: RectF
    ): RectF {

        val left =
            box.left.coerceIn(
                0f,
                1f
            )

        val top =
            box.top.coerceIn(
                0f,
                1f
            )

        val right =
            box.right.coerceIn(
                0f,
                1f
            )

        val bottom =
            box.bottom.coerceIn(
                0f,
                1f
            )

        return RectF(
            minOf(left, right),
            minOf(top, bottom),
            maxOf(left, right),
            maxOf(top, bottom)
        )
    }

    private fun normalizedToPixels(
        box: RectF,
        width: Int,
        height: Int
    ): RectF {

        return RectF(
            box.left * width,
            box.top * height,
            box.right * width,
            box.bottom * height
        )
    }

    private fun makeSquareCrop(
        box: RectF,
        factor: Float,
        width: Int,
        height: Int
    ): SquareCrop {

        val cx =
            (box.left + box.right) * 0.5f

        val cy =
            (box.top + box.bottom) * 0.5f

        val area =
            max(
                1f,
                box.width() * box.height()
            )

        val size =
            max(
                8f,
                sqrt(area) * factor
            )

        return SquareCrop(
            left = cx - size * 0.5f,
            top = cy - size * 0.5f,
            size = size,
            bitmapWidth = width,
            bitmapHeight = height
        )
    }

    /*
     * ------------------------------------------------------------------------
     * Crop rendering
     * ------------------------------------------------------------------------
     */

    private fun drawPaddedCrop(
        source: Bitmap,
        crop: SquareCrop,
        outBitmap: Bitmap,
        canvas: Canvas,
        meanColor: Int
    ) {

        canvas.drawColor(meanColor)

        val srcLeft =
            crop.left.coerceAtLeast(
                0f
            )

        val srcTop =
            crop.top.coerceAtLeast(
                0f
            )

        val srcRight =
            (
                    crop.left +
                            crop.size
                    ).coerceAtMost(
                    source.width.toFloat()
                )

        val srcBottom =
            (
                    crop.top +
                            crop.size
                    ).coerceAtMost(
                    source.height.toFloat()
                )

        if (srcRight <= srcLeft ||
            srcBottom <= srcTop
        ) {
            return
        }

        val dstLeft =
            (
                    srcLeft -
                            crop.left
                    ) / crop.size *
                    outBitmap.width

        val dstTop =
            (
                    srcTop -
                            crop.top
                    ) / crop.size *
                    outBitmap.height

        val dstRight =
            (
                    srcRight -
                            crop.left
                    ) / crop.size *
                    outBitmap.width

        val dstBottom =
            (
                    srcBottom -
                            crop.top
                    ) / crop.size *
                    outBitmap.height

        canvas.drawBitmap(
            source,
            Rect(
                srcLeft.toInt(),
                srcTop.toInt(),
                srcRight.toInt(),
                srcBottom.toInt()
            ),
            RectF(
                dstLeft,
                dstTop,
                dstRight,
                dstBottom
            ),
            paint
        )
    }

    /*
     * ------------------------------------------------------------------------
     * Bitmap -> NCHW ImageNet FP32
     * ------------------------------------------------------------------------
     */

    private fun bitmapToNchwNormalizedFallback(
        bitmap: Bitmap,
        dst: FloatArray
    ) {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val plane = bitmap.width * bitmap.height
        for (i in 0 until plane) {
            val px = pixels[i]
            val r = ((px ushr 16) and 0xFF) / 255f
            val g = ((px ushr 8) and 0xFF) / 255f
            val b = (px and 0xFF) / 255f
            dst[i] = (r - 0.485f) / 0.229f
            dst[plane + i] = (g - 0.456f) / 0.224f
            dst[2 * plane + i] = (b - 0.406f) / 0.225f
        }
    }

    private fun elapsedMs(
        startNs: Long
    ): Float {

        return (
                SystemClock.elapsedRealtimeNanos() -
                        startNs
                ) / 1_000_000f
    }

    /*
     * ------------------------------------------------------------------------
     * Data
     * ------------------------------------------------------------------------
     */

    private data class Decoded(
        val box: RectF,
        val score: Float
    )

    private data class SquareCrop(
        val left: Float,
        val top: Float,
        val size: Float,
        val bitmapWidth: Int = 1,
        val bitmapHeight: Int = 1
    )

    companion object {

        private const val TAG =
            "OstrackQnnTracker"

        private const val TEMPLATE_SIZE =
            128

        private const val SEARCH_SIZE =
            256

        private const val SEARCH_FEAT_SIZE =
            16

        private const val SEARCH_FEAT_ELEMENTS =
            SEARCH_FEAT_SIZE *
                    SEARCH_FEAT_SIZE

        private const val TEMPLATE_FACTOR =
            2.0f

        private const val SEARCH_FACTOR_BASE =
            4.0f

        private const val SCORE_ELEMENTS =
            256

        // Dense center-head peak score below this value is considered an
        // invalid/weak tracking response and is surfaced to the controller
        // as a tracker miss.
        private const val SIZE_ELEMENTS =
            512

        private const val OFFSET_ELEMENTS =
            512

        private const val SCORE_OFFSET =
            0

        private const val SIZE_OFFSET =
            SCORE_ELEMENTS

        private const val OFFSET_OFFSET =
            SCORE_ELEMENTS +
                    SIZE_ELEMENTS

        private const val OUTPUT_ELEMENTS =
            OstrackQnnContextBinaryRunner.OUTPUT_ELEMENTS

        /*
         * Standard ImageNet normalization.
         */
        private val MEAN =
            floatArrayOf(
                0.485f,
                0.456f,
                0.406f
            )

        private val STD =
            floatArrayOf(
                0.229f,
                0.224f,
                0.225f
            )

        /*
         * Padding color.
         */
        private val MEAN_COLOR =
            Color.rgb(
                124,
                116,
                103
            )

        fun tryCreate(
            context: Context,
            minValidScore: Float = 0.50f,
            assetName: String = OstrackQnnContextBinaryRunner.DEFAULT_ASSET
        ): OstrackQnnTracker? {

            return try {

                OstrackQnnTracker(
                    context,
                    minValidScore.coerceIn(0.20f, 0.95f),
                    assetName
                )

            } catch (t: Throwable) {

                Log.e(
                    TAG,
                    "OSTrack tracker creation failed",
                    t
                )

                null
            }
        }
    }
}