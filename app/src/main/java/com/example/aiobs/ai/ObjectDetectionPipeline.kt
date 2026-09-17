package com.example.aiobs.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.example.aiobs.camera.Yuv420ToBitmapConverter
import com.example.aiobs.core.ModelRuntimeConfig
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Live perception facade.
 *
 * Responsibilities are deliberately limited to model execution and image
 * preparation; lifecycle/state decisions stay in VisualRadarController and
 * track mutation stays in ObjectTracker.
 *
 * Live path:
 *   NORMAL  -> YOLO11n-Seg
 *   LOCKED  -> OSTrack + periodic YOLO mask + XFeat template refresh
 *   RECOVERY-> YOLO proposals + XFeat identity verification
 */
class ObjectDetectionPipeline(
    context: Context,
    private val runtimeConfig: ModelRuntimeConfig.Snapshot =
        ModelRuntimeConfig.Snapshot(),
    private val tuningConfig: com.example.aiobs.core.RuntimeTuningConfig =
        com.example.aiobs.core.RuntimeTuningConfig()
) {
    private val appContext = context.applicationContext
    private val destroyed = AtomicBoolean(false)
    private val yuvConverter = Yuv420ToBitmapConverter(640)

    private val yolo = Yolo11SegDetector(
        appContext,
        numThreads = 2,
        runtimeBackend = runtimeConfig.yoloBackend,
        modelAsset = runtimeConfig.yoloModelAsset,
        selectedClassIds = tuningConfig.yoloSelectedClassIds
    )

    private val xfeat: XFeatLocalIdentityMatcher? = if (runtimeConfig.xfeatEnabled) {
        try {
            XFeatLocalIdentityMatcher(
                context = appContext,
                assetName = runtimeConfig.xfeatModelAsset,
                identityThreshold = tuningConfig.xfeatIdentityThreshold
            )
        } catch (t: Throwable) {
            Log.e(TAG, "XFeat matcher initialization failed", t)
            null
        }
    } else {
        null
    }

    private var ostrack: OstrackQnnTracker? = null

    private var lastInferenceMs = 0f
    private var lastPrepareMs = 0f
    private var lastParseMs = 0f
    // Baseline-only YOLO timing buckets.
    private var lastYoloPrepMs = 0f
    private var lastYoloPreprocessMs = 0f
    private var lastYoloPostprocessMs = 0f
    private var lastPerceptionTotalMs = 0f
    private var lastOstrackMs = 0f
    private var lastOstrackPrepMs = 0f
    private var lastOstrackPreprocessMs = 0f
    private var lastOstrackInferMs = 0f
    private var lastOstrackPostprocessMs = 0f
    private var lastOstrackTotalMs = 0f
    private var lastDinov2PreprocessMs = 0f
    private var lastDinov2InferMs = 0f
    private var lastDinov2PostprocessMs = 0f
    private var lastDinov2TotalMs = 0f

    data class PreparedFrame(
        val bitmap: Bitmap,
        val rotationDegrees: Int,
        val prepareStartNs: Long,
        val prepareEndNs: Long
    ) {
        @Volatile
        private var canonicalBitmapCache: Bitmap? = null

        /** Lazy canonical/oriented frame shared by stages that need pixel coordinates. */
        @Synchronized
        fun canonicalBitmap(): Bitmap {
            if (rotationDegrees == 0) return bitmap
            canonicalBitmapCache?.let {
                if (!it.isRecycled) return it
            }
            val rotated = Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height,
                android.graphics.Matrix().apply { postRotate(rotationDegrees.toFloat()) },
                true
            )
            canonicalBitmapCache = rotated
            return rotated
        }

        @Synchronized
        fun release() {
            canonicalBitmapCache?.let {
                if (it !== bitmap && !it.isRecycled) {
                    try { it.recycle() } catch (_: Throwable) {}
                }
            }
            canonicalBitmapCache = null
            if (!bitmap.isRecycled) {
                try { bitmap.recycle() } catch (_: Throwable) {}
            }
        }
    }

    fun prepareFrame(
        frame: AiFrame,
        rotationDegrees: Int
    ): PreparedFrame? {
        if (destroyed.get()) return null

        val start = SystemClock.elapsedRealtimeNanos()

        return try {
            val bitmap = yuvConverter.convert((frame as AiFrame.CameraImage).frame)

            lastPrepareMs = elapsed(start)
            BaselinePerfLogger.framePrep(lastPrepareMs)

            PreparedFrame(
                bitmap = bitmap,
                rotationDegrees =
                    if (frame is AiFrame.CameraImage) {
                        normRot(rotationDegrees)
                    } else {
                        0
                    },
                prepareStartNs = start,
                prepareEndNs = SystemClock.elapsedRealtimeNanos()
            )
        } catch (t: Throwable) {
            Log.e(
                TAG,
                "prepareFrame failed",
                t
            )
            null
        }
    }

    fun runYolo(
        prepared: PreparedFrame,
        mode: YoloDetectionMode = YoloDetectionMode.BASIC
    ): List<DetectionResult> {
        if (destroyed.get()) return emptyList()

        val start = SystemClock.elapsedRealtimeNanos()
        val prepStart = SystemClock.elapsedRealtimeNanos()
        val workingBitmap = if (mode == YoloDetectionMode.BASIC) {
            // NORMAL keeps YOLO's existing fused-rotation letterbox path.
            prepared.bitmap
        } else {
            // LOCKED/RECOVERY share the same canonical bitmap with other stages.
            prepared.canonicalBitmap()
        }
        val prepMs = elapsed(prepStart)

        return try {
            /*
             * Detection coordinates and masks produced by YOLO must all
             * belong to this same working bitmap.
             *
             * Rotation is therefore performed exactly once here and YOLO is
             * called with rotationDegrees=0 to prevent a second rotation.
             */
            val result =
                yolo.process(
                    sourceBitmap = workingBitmap,
                    rotationDegrees = if (mode == YoloDetectionMode.BASIC) prepared.rotationDegrees else 0,
                    scoreThreshold = tuningConfig.yoloConfidenceThreshold,
                    mode = mode,
                    decodeMask = mode != YoloDetectionMode.BASIC
                )

            val stats =
                yolo.stats()

            lastInferenceMs = stats.inferenceMs
            lastParseMs = stats.decodeMs
            lastYoloPrepMs = prepMs
            lastYoloPreprocessMs = stats.inputPrepMs
            lastYoloPostprocessMs = (elapsed(start) - prepMs - stats.inputPrepMs - stats.inferenceMs).coerceAtLeast(0f)
            lastPerceptionTotalMs = elapsed(start)

            BaselinePerfLogger.model(
                name = "YOLO",
                prepMs = lastYoloPrepMs,
                preprocessMs = lastYoloPreprocessMs,
                inferMs = stats.inferenceMs,
                postprocessMs = lastYoloPostprocessMs,
                totalMs = lastPerceptionTotalMs
            )

            result
        } catch (t: Throwable) {
            Log.e(
                TAG,
                "YOLO failed",
                t
            )
            emptyList()
        } finally {
            // PreparedFrame owns the shared canonical bitmap lifecycle.
        }
    }

    /** Runs YOLO-Seg and selects a nearby same-class detection for mask refresh. */
    data class NearestYoloDetection(
        val detection: DetectionResult,
        val boxPx: RectF,
        val centerDistancePx: Float,
        val iou: Float
    )

    fun findSameClassYoloDetectionForMask(
        prepared: PreparedFrame,
        ostrackBox: RectF,
        expectedLabel: String
    ): NearestYoloDetection? {
        val detections = runYolo(prepared, YoloDetectionMode.LOCKED)
        if (detections.isEmpty()) return null

        val frameWidth = if (prepared.rotationDegrees % 180 == 0) prepared.bitmap.width else prepared.bitmap.height
        val frameHeight = if (prepared.rotationDegrees % 180 == 0) prepared.bitmap.height else prepared.bitmap.width

        val targetBox = RectF(
            ostrackBox.left.coerceIn(0f, frameWidth.toFloat()),
            ostrackBox.top.coerceIn(0f, frameHeight.toFloat()),
            ostrackBox.right.coerceIn(0f, frameWidth.toFloat()),
            ostrackBox.bottom.coerceIn(0f, frameHeight.toFloat())
        )
        if (targetBox.width() <= 1f || targetBox.height() <= 1f) return null

        val targetCx = targetBox.centerX()
        val targetCy = targetBox.centerY()

        // 1. 获取所有符合空间约束的候选者
        val validCandidates = detections
            .asSequence()
            .filter { it.label.equals(expectedLabel, ignoreCase = true) }
            .map { detection ->
                val box = RectF(
                    (detection.cx - detection.w * 0.5f) * frameWidth,
                    (detection.cy - detection.h * 0.5f) * frameHeight,
                    (detection.cx + detection.w * 0.5f) * frameWidth,
                    (detection.cy + detection.h * 0.5f) * frameHeight
                )
                val dx = box.centerX() - targetCx
                val dy = box.centerY() - targetCy
                val centerDistance = kotlin.math.sqrt(dx * dx + dy * dy).toFloat()
                val iou = boxIou(targetBox, box)
                NearestYoloDetection(detection, box, centerDistance, iou)
            }
            .filter { candidate ->
                // 【修改：大幅收紧警戒区，彻底消除远距离目标的误警报】
                // 条件1：两个框在画面中必须有实质性的交集重叠 (IoU > 0.2)，说明发生了物理贴靠
                // 条件2：或者，该候选者的中心点极其贴近 OSTrack 中心 (横/纵向距离小于框长的 30%)
                val thresholdX = targetBox.width() * 0.3f
                val thresholdY = targetBox.height() * 0.3f
                val dxAbs = kotlin.math.abs(candidate.boxPx.centerX() - targetCx)
                val dyAbs = kotlin.math.abs(candidate.boxPx.centerY() - targetCy)
                val isCenterVeryClose = dxAbs < thresholdX && dyAbs < thresholdY

                candidate.iou > 0.2f || isCenterVeryClose
            }
            .toList() // 转为 List 以便计算数量

        // 2. 【防同类遮挡的核心】：排他性判定 (Strict Uniqueness)
        if (validCandidates.isEmpty()) {
            return null // 没找到目标，放弃本次刷新
        }

        if (validCandidates.size != 1) {
            // Reject crowded neighborhoods to avoid refreshing from an ambiguous target.
            // 这说明当前正在发生同类贴身遮挡、密集交汇。
            // 此时绝不能盲目相信 centerDistancePx，否则极易发生特征污染。
            // 策略：放弃本次 10 秒刷新周期，保留现有的、干净的 Temporal XFeat Template。
            Log.w(TAG, "Ambiguous matches (${validCandidates.size}) near OSTrack target. Rejecting refresh to prevent template poisoning.")
            return null
        }

        // 3. 只有当附近有且仅有一个合法目标时，才绝对安全地返回它
        return validCandidates.single()
    }

    private fun boxIou(a: RectF, b: RectF): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        val iw = (right - left).coerceAtLeast(0f)
        val ih = (bottom - top).coerceAtLeast(0f)
        val intersection = iw * ih
        if (intersection <= 0f) return 0f
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union > 0f) intersection / union else 0f
    }

    /** Runs one locked-stage OSTrack step on the prepared frame. */
    fun runOstrack(
        prepared: PreparedFrame,
        currentBox: RectF
    ): OstrackQnnTracker.Result? {
        if (destroyed.get()) return null

        val tracker = ensureOstrack() ?: return null
        val start = SystemClock.elapsedRealtimeNanos()
        val prepStart = SystemClock.elapsedRealtimeNanos()
        val ostrackBitmap = prepared.canonicalBitmap()
        val prepMs = elapsed(prepStart)

        try {
            val result = tracker.track(
                bitmap = ostrackBitmap,
                currentBox = currentBox
            )

            lastOstrackPrepMs = prepMs
            lastOstrackPreprocessMs = tracker.lastPreprocessMs()
            lastOstrackInferMs = tracker.lastInferenceMs()
            lastOstrackPostprocessMs = tracker.lastPostprocessMs()
            lastOstrackTotalMs = elapsed(start)
            lastOstrackMs = lastOstrackInferMs
            lastInferenceMs = lastOstrackInferMs
            lastPerceptionTotalMs = lastOstrackTotalMs

            BaselinePerfLogger.model(
                name = "OSTrack",
                prepMs = lastOstrackPrepMs,
                preprocessMs = lastOstrackPreprocessMs,
                inferMs = lastOstrackInferMs,
                postprocessMs = lastOstrackPostprocessMs,
                totalMs = lastOstrackTotalMs
            )

            return result
        } finally {
            // PreparedFrame owns the shared canonical bitmap lifecycle.
        }
    }


    private fun ensureOstrack(): OstrackQnnTracker? {
        if (ostrack != null) return ostrack

        ostrack = OstrackQnnTracker.tryCreate(
            appContext,
            tuningConfig.ostrackMinScore,
            runtimeConfig.ostrackModelAsset
        )

        if (ostrack == null) {
            Log.e(TAG, "OSTrack QNN tracker unavailable")
        }
        return ostrack
    }

    fun resetOstrackTemplate() {
        try {
            ostrack?.resetTemplate()
        } catch (_: Throwable) {
        }
    }

    fun currentInferenceMs(): Float = lastInferenceMs
    fun yoloBackendName(): String = yolo.backendName()
    fun currentParseMs(): Float = lastParseMs
    fun currentPrepareMs(): Float = lastPrepareMs
    fun currentPerceptionTotalMs(): Float = lastPerceptionTotalMs
    fun currentOstrackMs(): Float = lastOstrackMs
    fun currentOstrackPrepMs(): Float = lastOstrackPrepMs
    fun currentOstrackPreprocessMs(): Float = lastOstrackPreprocessMs
    fun currentOstrackInferMs(): Float = lastOstrackInferMs
    fun currentOstrackPostprocessMs(): Float = lastOstrackPostprocessMs
    fun currentOstrackTotalMs(): Float = lastOstrackTotalMs

    fun extractXFeat(
        bitmap: Bitmap,
        box: RectF,
        mask: SegmentationMask? = null
    ): XFeatFeatureSet? {
        if (destroyed.get()) {
            return null
        }
        val matcher = xfeat
        if (matcher == null) {
            return null
        }
        return try {
            val totalStart = SystemClock.elapsedRealtimeNanos()
            val prepStart = SystemClock.elapsedRealtimeNanos()
            val patch = buildXFeatPatch(bitmap, box, mask)
            val prepMs = elapsed(prepStart)
            val result = matcher.extract(patch)
            val totalMs = elapsed(totalStart)
            BaselinePerfLogger.model(
                name = "XFeat",
                prepMs = prepMs,
                preprocessMs = matcher.lastInputMs.toFloat(),
                inferMs = matcher.lastInferMs.toFloat(),
                postprocessMs = matcher.lastPostprocessMs.toFloat(),
                totalMs = totalMs
            )
            result
        } catch (t: Throwable) {
            Log.e(TAG, "XFeat extraction failed", t)
            null
        }
    }

    fun matchXFeat(
        reference: XFeatFeatureSet?,
        candidate: XFeatFeatureSet?,
        bankIndex: Int = 0 // 🌟 恢复为默认的 bankIndex
    ): XFeatLocalIdentityMatcher.MatchResult? {
        val matcher = xfeat ?: return null
        if (reference == null || candidate == null) return null
        return try {
            val startNs = SystemClock.elapsedRealtimeNanos()
            val result = matcher.match(reference, candidate, bankIndex)
            BaselinePerfLogger.xfeatMatch(elapsed(startNs))
            result
        } catch (t: Throwable) {
            Log.e(TAG, "XFeat match failed", t)
            null
        }
    }

    fun isXFeatEnabled(): Boolean =
        runtimeConfig.xfeatEnabled && xfeat?.available == true

    fun xfeatBackendName(): String =
        if (runtimeConfig.xfeatEnabled) {
            xfeat?.backendName ?: "UNAVAILABLE"
        } else {
            "DISABLED"
        }

    /**
     * Builds the canonical local patch used by BOTH Anchor and Recovery XFeat.
     *
     * The old implementation cropped exactly to the bbox and then stretched
     * that arbitrary aspect ratio to 640x640 in the matcher. That is especially
     * harmful for tall people / wide vehicles because local geometry is warped.
     *
     * The canonical policy here is: small, symmetric context padding around the
     * bbox, followed by aspect-preserving letterbox inside XFeatLocalIdentityMatcher.
     * Anchor and Recovery therefore see the same geometric representation.
     */
    private fun buildXFeatPatch(
        bitmap: Bitmap,
        box: RectF,
        mask: SegmentationMask?
    ): LocalGrayPatch? {
        if (bitmap.width <= 1 || bitmap.height <= 1) return null

        val rawLeft = box.left.coerceIn(0f, bitmap.width.toFloat() - 1f)
        val rawTop = box.top.coerceIn(0f, bitmap.height.toFloat() - 1f)
        val rawRight = box.right.coerceIn(rawLeft + 1f, bitmap.width.toFloat())
        val rawBottom = box.bottom.coerceIn(rawTop + 1f, bitmap.height.toFloat())

        val rawWidth = rawRight - rawLeft
        val rawHeight = rawBottom - rawTop
        if (rawWidth < 8f || rawHeight < 8f) return null

        // Same protection padding used by the LOCKED YOLO crop policy.
        // It keeps a small amount of context without letting the patch become
        // dominated by background. It is symmetric in both axes.
        val padding = maxOf(rawWidth, rawHeight) * XFEAT_PATCH_PADDING_RATIO
        val centerX = (rawLeft + rawRight) * 0.5f
        val centerY = (rawTop + rawBottom) * 0.5f

        val left = (centerX - rawWidth * 0.5f - padding)
            .coerceIn(0f, bitmap.width.toFloat() - 1f)
        val top = (centerY - rawHeight * 0.5f - padding)
            .coerceIn(0f, bitmap.height.toFloat() - 1f)
        val right = (centerX + rawWidth * 0.5f + padding)
            .coerceIn(left + 1f, bitmap.width.toFloat())
        val bottom = (centerY + rawHeight * 0.5f + padding)
            .coerceIn(top + 1f, bitmap.height.toFloat())

        val rect = Rect(
            left.toInt().coerceIn(0, bitmap.width - 1),
            top.toInt().coerceIn(0, bitmap.height - 1),
            maxOf(left.toInt() + 1, right.toInt()).coerceAtMost(bitmap.width),
            maxOf(top.toInt() + 1, bottom.toInt()).coerceAtMost(bitmap.height)
        )
        if (rect.width() < 8 || rect.height() < 8) return null

        Log.d(
            TAG,
            "[XFEAT_PATCH] " +
                "bbox=${fmt(rawWidth)}x${fmt(rawHeight)} " +
                "patch=${rect.width()}x${rect.height()} " +
                "aspect=${fmt(rect.width().toFloat() / rect.height().coerceAtLeast(1))} " +
                "padding=${fmt(padding)}"
        )

        val gray = ByteArray(rect.width() * rect.height())
        if (NativeBitmapPreprocessor.extractGrayPatch(
                bitmap = bitmap,
                rectLeft = rect.left,
                rectTop = rect.top,
                rectWidth = rect.width(),
                rectHeight = rect.height(),
                mask = mask,
                rawLeft = rawLeft,
                rawTop = rawTop,
                rawRight = rawRight,
                rawBottom = rawBottom,
                output = gray
            )
        ) {
            Log.d(TAG, "[XFEAT_PATCH_NATIVE] backend=NATIVE_BITMAP_GRAY patch=${rect.width()}x${rect.height()}")
            return LocalGrayPatch(rect.width(), rect.height(), gray)
        }

        // Kotlin fallback for devices where the native helper is unavailable.
        val crop = Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height())
        return try {
            val pixels = IntArray(crop.width * crop.height)
            crop.getPixels(pixels, 0, crop.width, 0, 0, crop.width, crop.height)
            val fallbackGray = ByteArray(pixels.size)
            val usableMask = mask != null &&
                mask.width > 0 && mask.height > 0 &&
                mask.pixels.isNotEmpty() && mask.area > 0 &&
                rawWidth > 0f && rawHeight > 0f
            for (y in 0 until crop.height) {
                val sourceY = rect.top + y + 0.5f
                val my = if (usableMask && sourceY >= rawTop && sourceY < rawBottom) {
                    (((sourceY - rawTop) / rawHeight) * mask!!.height).toInt().coerceIn(0, mask.height - 1)
                } else -1
                for (x in 0 until crop.width) {
                    val pixel = pixels[y * crop.width + x]
                    val grayValue = (0.299f * Color.red(pixel) + 0.587f * Color.green(pixel) + 0.114f * Color.blue(pixel)).toInt().coerceIn(0, 255)
                    val sourceX = rect.left + x + 0.5f
                    val keep = if (usableMask && my >= 0 && sourceX >= rawLeft && sourceX < rawRight) {
                        val mx = (((sourceX - rawLeft) / rawWidth) * mask!!.width).toInt().coerceIn(0, mask.width - 1)
                        mask.contains(mx, my)
                    } else if (usableMask) false else true
                    fallbackGray[y * crop.width + x] = if (keep) grayValue.toByte() else 0
                }
            }
            LocalGrayPatch(crop.width, crop.height, fallbackGray)
        } finally {
            try { crop.recycle() } catch (_: Throwable) {}
        }
    }

    // 1. 成员变量声明
    // 1. 成员变量声明：读取 runtimeConfig 里的动态名字
    private val dinov2: Dinov2QnnContextBinaryRunner? = if (runtimeConfig.dinov2Enabled) {
        try {
            Dinov2QnnContextBinaryRunner(
                appContext,
                runtimeConfig.dinov2ModelAsset // 🌟 不再硬编码，读配置！
            )
        } catch (t: Throwable) {
            Log.e(TAG, "DINOv2 init failed", t)
            null
        }
    } else null
    private var dinov2InputArray: FloatArray? = null

    // 2. 提取特征的方法 (利用预分配内存，零 GC)
    // 2. 提取特征的方法 (利用预分配内存，零 GC)
    fun extractDinov2(bitmap: Bitmap, box: RectF): FloatArray? {
        val runner = dinov2 ?: return null
        val targetW = runner.inputWidth
        val targetH = runner.inputHeight

        val totalStartNs = System.nanoTime()
        val prepStartNs = System.nanoTime()
        val patch = buildDinov2Patch(bitmap, box, targetW, targetH) ?: return null
        val prepMs = (System.nanoTime() - prepStartNs) / 1_000_000.0f

        if (dinov2InputArray == null || dinov2InputArray!!.size != 3 * targetW * targetH) {
            dinov2InputArray = FloatArray(3 * targetW * targetH)
        }

        val input = dinov2InputArray!!

        val preprocessStartNs = System.nanoTime()
        val packed = NativeRgbInputPacker.packBitmapImageNetNchwFloat32(
            patch,
            input,
            targetW,
            targetH
        )
        if (!packed) {
            packDinov2Fallback(patch, input)
        }
        if (patch !== bitmap) patch.recycle() // 及时回收中间图

        val preprocessMs = (System.nanoTime() - preprocessStartNs) / 1_000_000.0f
        lastDinov2PreprocessMs = preprocessMs
        val result = runner.infer(input)
        lastDinov2InferMs = runner.lastInferenceMs
        lastDinov2PostprocessMs = 0f
        lastDinov2TotalMs = (System.nanoTime() - totalStartNs) / 1_000_000.0f
        BaselinePerfLogger.model(
            name = "DINOv2",
            prepMs = prepMs,
            preprocessMs = lastDinov2PreprocessMs,
            inferMs = lastDinov2InferMs,
            postprocessMs = lastDinov2PostprocessMs,
            totalMs = lastDinov2TotalMs
        )
        return result
    }

    private fun packDinov2Fallback(bitmap: Bitmap, dst: FloatArray) {
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

    // 3. 裁图逻辑 (不要像 XFeat 那么紧凑，DINO 喜欢看上下文)
    private fun buildDinov2Patch(bitmap: Bitmap, box: RectF, targetW: Int, targetH: Int): Bitmap? {
        if (bitmap.width <= 1 || bitmap.height <= 1) return null

        val rawLeft = box.left.coerceIn(0f, bitmap.width.toFloat() - 1f)
        val rawTop = box.top.coerceIn(0f, bitmap.height.toFloat() - 1f)
        val rawRight = box.right.coerceIn(rawLeft + 1f, bitmap.width.toFloat())
        val rawBottom = box.bottom.coerceIn(rawTop + 1f, bitmap.height.toFloat())

        val rawWidth = rawRight - rawLeft
        val rawHeight = rawBottom - rawTop
        if (rawWidth < 8f || rawHeight < 8f) return null

        // DINOv2 喜欢大背景，给它 20% 的扩充
        val padding = maxOf(rawWidth, rawHeight) * 0.20f
        val centerX = (rawLeft + rawRight) * 0.5f
        val centerY = (rawTop + rawBottom) * 0.5f

        val left = (centerX - rawWidth * 0.5f - padding).coerceIn(0f, bitmap.width.toFloat() - 1f)
        val top = (centerY - rawHeight * 0.5f - padding).coerceIn(0f, bitmap.height.toFloat() - 1f)
        val right = (centerX + rawWidth * 0.5f + padding).coerceIn(left + 1f, bitmap.width.toFloat())
        val bottom = (centerY + rawHeight * 0.5f + padding).coerceIn(top + 1f, bitmap.height.toFloat())

        val cropW = (right - left).toInt()
        val cropH = (bottom - top).toInt()
        if (cropW < 8 || cropH < 8) return null

        val crop = Bitmap.createBitmap(bitmap, left.toInt(), top.toInt(), cropW, cropH)
        val scaled = Bitmap.createScaledBitmap(crop, targetW, targetH, true)
        if (crop !== scaled) crop.recycle()
        return scaled
    }

    fun currentDinov2PreprocessMs(): Float = lastDinov2PreprocessMs
    fun currentDinov2InferMs(): Float = lastDinov2InferMs
    fun currentDinov2PostprocessMs(): Float = lastDinov2PostprocessMs
    fun currentDinov2TotalMs(): Float = lastDinov2TotalMs

    fun delegateName(): String =
        "Normal=YOLO(${yolo.backendName()}) | " +
                "Locked=OSTrack/QNN-HTP | " +
                "Lost=YOLO+XFeat(W8A16) | " +
                "XFeat=${if (runtimeConfig.xfeatEnabled) xfeatBackendName() else "DISABLED"}"

    fun destroy() {
        if (
            !destroyed.compareAndSet(
                false,
                true
            )
        ) {
            return
        }

        try {
            yolo.close()
        } catch (_: Throwable) {
        }


        try {
            xfeat?.close()
        } catch (_: Throwable) {
        }

        try {
            ostrack?.close()
        } catch (_: Throwable) {
        }

        ostrack = null
    }

    private fun elapsed(
        start: Long
    ): Float =
        (
                SystemClock.elapsedRealtimeNanos() -
                        start
                ) / 1_000_000f

    private fun normRot(
        value: Int
    ): Int =
        (
                (value % 360) +
                        360
                ) % 360

    private fun fmt(
        v: Float
    ): String =
        String.format(
            java.util.Locale.US,
            "%.3f",
            v
        )

    companion object {
        private const val XFEAT_PATCH_PADDING_RATIO = 0.055f
        private const val TAG =
            "SimplePerception"

    }
}
