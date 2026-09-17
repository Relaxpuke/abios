package com.example.aiobs.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.example.aiobs.core.ModelRuntimeConfig
import com.example.aiobs.core.YoloClassCatalog
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Per-frame materialization policy. */
enum class YoloDetectionMode {
    BASIC,
    LOCKED,
    RECOVERY
}

class Yolo11SegDetector(
    context: Context,
    private val numThreads: Int = 2,
    private val runtimeBackend: ModelRuntimeConfig.Backend = ModelRuntimeConfig.Backend.QNN,
    private val modelAsset: String = DEFAULT_MODEL_ASSET,
    selectedClassIds: Set<Int> = DEFAULT_TARGET_CLASS_IDS
) {

    data class InferenceStats(
        val loadMs: Float,
        val allocateMs: Float,
        val inferenceMs: Float,
        val decodeMs: Float,
        val candidateCount: Int,
        val outputCount: Int,
        val backend: String,
        val inputPrepMs: Float = 0f,
        val rotationMs: Float = 0f,
        val letterboxMs: Float = 0f,
        val fillInputMs: Float = 0f,
        val outputDecodeMs: Float = 0f,
        val totalMs: Float = 0f
    )

    private data class Letterbox(
        val bitmap: Bitmap,
        val scale: Float,
        val padX: Float,
        val padY: Float,
        val rotatedWidth: Int,
        val rotatedHeight: Int
    )

    private data class Candidate(
        val classId: Int,
        val confidence: Float,
        val boxInput: RectF,
        val anchorIndex: Int
    )

    private data class ClassScore(
        val classId: Int,
        val confidence: Float
    )

    private data class TensorQuant(
        val scale: Float,
        val zeroPoint: Int
    )

    private val appContext = context.applicationContext

    private val selectedClassIds: IntArray = selectedClassIds
        .filter { it in 0 until NUM_CLASSES }
        .distinct()
        .sorted()
        .toIntArray()

    private var interpreter: Interpreter? = null
    private var tfliteHandle: TfliteAcceleration.InterpreterHandle? = null

    private val modelBuffer: ByteBuffer
    private var activeBackendName = "UNINITIALIZED"

    private var inputType: DataType = DataType.FLOAT32
    private var inputQuant: TensorQuant? = null
    private var inputElementCount: Int = 0

    private var output0Type: DataType = DataType.FLOAT32
    private var output1Type: DataType = DataType.FLOAT32

    private var output0Quant: TensorQuant? = null
    private var output1Quant: TensorQuant? = null

    private lateinit var inputBuffer: ByteBuffer
    private var detOutputBuffer: Any? = null
    private var protoOutputBuffer: Any? = null

    private val detOutput = Array(1) { Array(DET_CHANNELS) { FloatArray(ANCHOR_COUNT) } }
    private val protoOutput = Array(1) { Array(PROTO_CHANNELS) { Array(PROTO_SIZE) { FloatArray(PROTO_SIZE) } } }

    private val inputPixels = IntArray(MODEL_SIZE * MODEL_SIZE)
    private val inputFloatArray = FloatArray(3 * MODEL_SIZE * MODEL_SIZE)
    private val detFloatArray = FloatArray(DET_CHANNELS * ANCHOR_COUNT)
    private val protoFloatArray = FloatArray(PROTO_CHANNELS * PROTO_SIZE * PROTO_SIZE)

    private val quantizedLut = ByteArray(256)
    private var quantizedLutType: DataType? = null
    private var quantizedLutScale: Float = Float.NaN
    private var quantizedLutZeroPoint: Int = Int.MIN_VALUE

    private val letterboxBitmap = Bitmap.createBitmap(MODEL_SIZE, MODEL_SIZE, Bitmap.Config.ARGB_8888)
    private val letterboxCanvas = Canvas(letterboxBitmap)
    private val resizePaint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)

    private val outputMap = HashMap<Int, Any>(2)
    private var lastStats = InferenceStats(0f, 0f, 0f, 0f, 0, 0, "UNINITIALIZED")
    private var destroyed = false

    private var initialLoadMs = 0f
    private var initialAllocateMs = 0f
    private var lastThresholdDebugLogNs = 0L

    private var lastInputPrepMs = 0f
    private var lastRotationMs = 0f
    private var lastLetterboxMs = 0f
    private var lastFillInputMs = 0f
    private var lastOutputDecodeMs = 0f
    private var lastTotalMs = 0f

    private val labels: List<String>

    init {
        labels = loadLabels()
        val bytes = appContext.assets.open(modelAsset).use { it.readBytes() }
        require(bytes.isNotEmpty()) { "YOLO model asset is empty: $modelAsset" }

        modelBuffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
            put(bytes)
            rewind()
        }
        Log.i(TAG, "Created detector model=$modelAsset runtimeBackend=$runtimeBackend size=${bytes.size} bytes")
    }

    private fun ensureInitialized() {
        if (interpreter != null) return
        val loadStart = SystemClock.elapsedRealtimeNanos()

        tfliteHandle = TfliteAcceleration.create(
            context = appContext,
            modelBuffer = modelBuffer,
            numThreads = numThreads,
            preferQnn = runtimeBackend == ModelRuntimeConfig.Backend.QNN,
            preferGpu = runtimeBackend == ModelRuntimeConfig.Backend.GPU,
            qnnHtpPrecision = TfliteAcceleration.QnnHtpPrecision.QUANTIZED,
            tag = TAG
        )

        interpreter = tfliteHandle!!.interpreter
        activeBackendName = tfliteHandle!!.backend.name
        initialLoadMs = elapsedMs(loadStart)

        val allocStart = SystemClock.elapsedRealtimeNanos()
        interpreter!!.allocateTensors()
        initialAllocateMs = elapsedMs(allocStart)

        configureTensorContract(interpreter!!)
    }

    fun process(
        sourceBitmap: Bitmap,
        rotationDegrees: Int,
        scoreThreshold: Float,
        mode: YoloDetectionMode = YoloDetectionMode.BASIC,
        decodeMask: Boolean = true
    ): List<DetectionResult> {
        check(!destroyed) { "YOLO detector already closed" }
        ensureInitialized()

        val safeInterpreter = interpreter!!
        var working = sourceBitmap
        var rotated: Bitmap? = null
        val processStartNs = SystemClock.elapsedRealtimeNanos()

        var rotationMs = 0f
        var letterboxMs = 0f
        var fillInputMs = 0f
        var outputDecodeMs = 0f

        try {
            val normalizedRotation = normalizeRotation(rotationDegrees)

            // 【性能核心优化】：只有在需要提取小图特征时（RECOVERY）
            // 才真正去分配 8MB 内存做整图旋转。日常 BASIC 或 LOCKED 模式，直接跳过，零内存分配！
            val mightNeedPatches = mode == YoloDetectionMode.RECOVERY

            val rotationStartNs = SystemClock.elapsedRealtimeNanos()
            if (normalizedRotation != 0 && mightNeedPatches) {
                rotated = rotateBitmap(sourceBitmap, normalizedRotation)
                working = rotated ?: sourceBitmap
            }
            rotationMs = elapsedMs(rotationStartNs)

            // Letterbox 矩阵融合：在缩放的同时利用 Canvas 硬件加速直接完成旋转
            val letterboxStartNs = SystemClock.elapsedRealtimeNanos()
            val lb = letterbox(
                source = if (mightNeedPatches) working else sourceBitmap,
                size = MODEL_SIZE,
                rotation = if (mightNeedPatches) 0 else normalizedRotation
            )
            letterboxMs = elapsedMs(letterboxStartNs)

            val fillStartNs = SystemClock.elapsedRealtimeNanos()
            fillInput(lb.bitmap, inputBuffer, inputType, inputQuant)
            fillInputMs = elapsedMs(fillStartNs)

            outputMap.clear()
            val detBuffer = detOutputBuffer ?: error("YOLO detection output buffer not initialized")
            val protoBuffer = protoOutputBuffer ?: error("YOLO proto output buffer not initialized")
            resetOutputBuffersForRun()
            outputMap[0] = detBuffer
            outputMap[1] = protoBuffer

            val inferenceStartNs = SystemClock.elapsedRealtimeNanos()
            safeInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
            val inferenceMs = elapsedMs(inferenceStartNs)

            val outputDecodeStartNs = SystemClock.elapsedRealtimeNanos()

            val candidates = if (output0Type.isQuantizedTensor() && output1Type.isQuantizedTensor()) {
                decodeQuantizedCandidates(
                    detOutputBuffer as ByteBuffer,
                    output0Quant ?: error("Quantized output0 missing quantization"),
                    scoreThreshold
                )
            } else {
                decodeRuntimeOutputs(decodeMask)
                decodeCandidates(detOutput[0], scoreThreshold)
            }
            outputDecodeMs = elapsedMs(outputDecodeStartNs)

            val kept = nms(candidates, NMS_IOU)
            val decodeStart = SystemClock.elapsedRealtimeNanos()

            // 虚拟画面宽高（无论是物理旋转还是矩阵旋转，这里获取的都是正确的直立宽高）
            val virtualWidth = lb.rotatedWidth
            val virtualHeight = lb.rotatedHeight

            val results = ArrayList<DetectionResult>(kept.size)
            for (candidate in kept) {
                val boxOriginal = mapBoxToOriginal(candidate.boxInput, lb, virtualWidth, virtualHeight)
                if (boxOriginal.width() < MIN_BOX_PX || boxOriginal.height() < MIN_BOX_PX) continue

                val label = labels.getOrNull(candidate.classId) ?: fallbackLabel(candidate.classId)
                if (!isSelectedClass(candidate.classId)) continue

                val cx = ((boxOriginal.left + boxOriginal.right) * 0.5f / virtualWidth).coerceIn(0f, 1f)
                val cy = ((boxOriginal.top + boxOriginal.bottom) * 0.5f / virtualHeight).coerceIn(0f, 1f)
                val w = (boxOriginal.width() / virtualWidth).coerceIn(0f, 1f)
                val h = (boxOriginal.height() / virtualHeight).coerceIn(0f, 1f)

                val needMask = mode != YoloDetectionMode.BASIC

                val mask = if (needMask && decodeMask) {
                    if (output0Type.isQuantizedTensor() && output1Type.isQuantizedTensor()) {
                        decodeLocalMaskQuantized(
                            detBuffer = detOutputBuffer as ByteBuffer,
                            detQuant = output0Quant!!,
                            protoBuffer = protoOutputBuffer as ByteBuffer,
                            protoQuant = output1Quant!!,
                            anchorIndex = candidate.anchorIndex,
                            boxInput = candidate.boxInput
                        )
                    } else {
                        decodeLocalMaskFloat32NativeFallback(
                            detBuffer = detOutputBuffer as ByteBuffer,
                            protoBuffer = protoOutputBuffer as ByteBuffer,
                            anchorIndex = candidate.anchorIndex,
                            boxInput = candidate.boxInput
                        )
                    }
                } else null

                results += DetectionResult(
                    label = label,
                    confidence = candidate.confidence,
                    cx = cx,
                    cy = cy,
                    w = w,
                    h = h,
                    segmentationMask = mask
                )
            }

            val decodeMs = elapsedMs(decodeStart)
            val totalMs = elapsedMs(processStartNs)

            lastInputPrepMs = rotationMs + letterboxMs + fillInputMs
            lastRotationMs = rotationMs
            lastLetterboxMs = letterboxMs
            lastFillInputMs = fillInputMs
            lastOutputDecodeMs = outputDecodeMs
            lastTotalMs = totalMs

            lastStats = InferenceStats(initialLoadMs, initialAllocateMs, inferenceMs, decodeMs, candidates.size, results.size, activeBackendName, lastInputPrepMs, rotationMs, letterboxMs, fillInputMs, outputDecodeMs, totalMs)

            return results

        } catch (t: Throwable) {
            Log.e(TAG, "YOLO inference failed model=$modelAsset backend=$activeBackendName", t)
            return emptyList()
        } finally {
            if (rotated != null && rotated !== sourceBitmap) {
                try { rotated.recycle() } catch (_: Throwable) {}
            }
        }
    }

    private fun configureTensorContract(interpreter: Interpreter) {
        val input = interpreter.getInputTensor(0)
        val out0 = interpreter.getOutputTensor(0)
        val out1 = interpreter.getOutputTensor(1)

        inputType = input.dataType()
        inputElementCount = input.numElements()
        inputQuant = readQuantization(input, inputType.isQuantizedTensor())
        inputBuffer = ByteBuffer.allocateDirect(inputElementCount * bytesPerElement(inputType)).order(ByteOrder.nativeOrder())

        output0Type = out0.dataType()
        output0Quant = readQuantization(out0, output0Type.isQuantizedTensor())
        output1Type = out1.dataType()
        output1Quant = readQuantization(out1, output1Type.isQuantizedTensor())

        detOutputBuffer = createOutputBuffer(output0Type, out0.numElements())
        protoOutputBuffer = createOutputBuffer(output1Type, out1.numElements())
    }

    private fun fillInput(bitmap: Bitmap, buffer: ByteBuffer, type: DataType, quant: TensorQuant?) {
        buffer.rewind()
        when (type) {
            DataType.FLOAT32 -> {
                if (!NativeYoloInputPacker.packBitmapNchwFloat32(bitmap, buffer, MODEL_SIZE, MODEL_SIZE)) {
                    bitmap.getPixels(inputPixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)
                    if (!NativeYoloInputPacker.packNchwFloat32(inputPixels, buffer, MODEL_SIZE, MODEL_SIZE)) fillInputFloat32(buffer)
                }
            }
            DataType.INT8, DataType.UINT8 -> {
                val q = requireNotNull(quant)
                ensureQuantizedLut(type, q.scale, q.zeroPoint)
                if (!NativeYoloInputPacker.packBitmapNchwQuantized(bitmap, buffer, quantizedLut, MODEL_SIZE, MODEL_SIZE)) {
                    bitmap.getPixels(inputPixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)
                    if (!NativeYoloInputPacker.packNchwQuantized(inputPixels, buffer, quantizedLut, MODEL_SIZE, MODEL_SIZE)) fillInputQuantizedFallback(buffer, type, q)
                }
            }
            else -> error("Unsupported input type=$type")
        }
        buffer.rewind()
    }

    private fun fillInputFloat32(buffer: ByteBuffer) {
        var index = 0
        for (channel in 0 until 3) {
            for (i in inputPixels.indices) {
                val value = when (channel) {
                    0 -> (inputPixels[i] shr 16) and 0xFF
                    1 -> (inputPixels[i] shr 8) and 0xFF
                    else -> inputPixels[i] and 0xFF
                }
                inputFloatArray[index++] = value / 255f
            }
        }
        buffer.asFloatBuffer().put(inputFloatArray)
    }

    private fun ensureQuantizedLut(type: DataType, scale: Float, zeroPoint: Int) {
        if (quantizedLutType == type && quantizedLutScale == scale && quantizedLutZeroPoint == zeroPoint) return
        for (value in 0..255) {
            val q = (value / 255f / scale + zeroPoint).roundToInt().coerceIn(if (type == DataType.INT8) -128 else 0, if (type == DataType.INT8) 127 else 255)
            quantizedLut[value] = q.toByte()
        }
        quantizedLutType = type; quantizedLutScale = scale; quantizedLutZeroPoint = zeroPoint
    }

    private fun fillInputQuantizedFallback(buffer: ByteBuffer, type: DataType, quant: TensorQuant) {
        if (type == DataType.INT8) fillInputInt8(buffer, quant) else fillInputUInt8(buffer, quant)
    }

    private fun fillInputInt8(buffer: ByteBuffer, quant: TensorQuant) {
        for (channel in 0 until 3) {
            for (i in inputPixels.indices) {
                val value255 = when (channel) { 0 -> (inputPixels[i] shr 16) and 0xFF; 1 -> (inputPixels[i] shr 8) and 0xFF; else -> inputPixels[i] and 0xFF }
                buffer.put((value255 / 255f / quant.scale + quant.zeroPoint).roundToInt().coerceIn(-128, 127).toByte())
            }
        }
    }

    private fun fillInputUInt8(buffer: ByteBuffer, quant: TensorQuant) {
        for (channel in 0 until 3) {
            for (i in inputPixels.indices) {
                val value255 = when (channel) { 0 -> (inputPixels[i] shr 16) and 0xFF; 1 -> (inputPixels[i] shr 8) and 0xFF; else -> inputPixels[i] and 0xFF }
                buffer.put((value255 / 255f / quant.scale + quant.zeroPoint).roundToInt().coerceIn(0, 255).toByte())
            }
        }
    }

    private fun decodeRuntimeOutputs(decodeMask: Boolean) {
        val buffer = detOutputBuffer ?: return
        when (output0Type) {
            DataType.FLOAT32 -> {
                val bb = buffer as ByteBuffer
                bb.rewind()
                bb.asFloatBuffer().get(detFloatArray)
                var index = 0
                for (channel in 0 until DET_CHANNELS) {
                    for (anchor in 0 until ANCHOR_COUNT) detOutput[0][channel][anchor] = detFloatArray[index++]
                }
            }
            else -> {} // Kept clean for brevity
        }

        if (decodeMask && !NativeLockedMaskKernel.isAvailable()) {
            val protoBuff = protoOutputBuffer ?: return
            if (output1Type == DataType.FLOAT32) {
                val bb = protoBuff as ByteBuffer
                bb.rewind()
                bb.asFloatBuffer().get(protoFloatArray)
                var index = 0
                for (channel in 0 until PROTO_CHANNELS) {
                    for (y in 0 until PROTO_SIZE) {
                        for (x in 0 until PROTO_SIZE) protoOutput[0][channel][y][x] = protoFloatArray[index++]
                    }
                }
            }
        }
    }

    private fun resetOutputBuffersForRun() {
        (detOutputBuffer as? ByteBuffer)?.rewind()
        (protoOutputBuffer as? ByteBuffer)?.rewind()
    }

    private fun createOutputBuffer(type: DataType, elementCount: Int): Any {
        return when (type) {
            DataType.FLOAT32 -> ByteBuffer.allocateDirect(elementCount * 4).order(ByteOrder.nativeOrder())
            else -> ByteBuffer.allocateDirect(elementCount).order(ByteOrder.nativeOrder())
        }
    }

    private fun readQuantization(tensor: org.tensorflow.lite.Tensor, requiredForQuantized: Boolean): TensorQuant? {
        if (!requiredForQuantized) return null
        val params = tensor.quantizationParams()
        return TensorQuant(scale = params.scale, zeroPoint = params.zeroPoint)
    }

    private fun bytesPerElement(type: DataType): Int = if (type == DataType.FLOAT32) 4 else 1
    private fun DataType.isQuantizedTensor(): Boolean = this == DataType.INT8 || this == DataType.UINT8

    private fun decodeQuantizedCandidates(
        buffer: ByteBuffer,
        quant: TensorQuant,
        scoreThreshold: Float
    ): List<Candidate> {
        val candidates = ArrayList<Candidate>(ANCHOR_COUNT / 2)

        val view = buffer.duplicate().order(ByteOrder.nativeOrder())

        // ============================================================
        // YOLO threshold diagnostics
        // 只用于确认：
        // 1. 模型到底有没有产生 car/person score
        // 2. 最大 score 到多少
        // 3. 有多少 anchor 被不同 threshold 保留下来
        //
        // 不改变实际检测逻辑。
        // ============================================================
        var maxScore = 0f
        var maxScoreClassId = -1
        var maxScoreAnchor = -1

        var countGe001 = 0
        var countGe005 = 0
        var countGe010 = 0
        var countGe020 = 0
        var countGe030 = 0
        var countGe050 = 0
        var countGeThreshold = 0

        for (anchor in 0 until ANCHOR_COUNT) {
            val score0 = dequantizedByteAt(
                view,
                CLASS_OFFSET * ANCHOR_COUNT + anchor,
                quant,
                output0Type
            ).coerceIn(0f, 1f)

            val score2 = dequantizedByteAt(
                view,
                (CLASS_OFFSET + 2) * ANCHOR_COUNT + anchor,
                quant,
                output0Type
            ).coerceIn(0f, 1f)

            var best = 0f
            var bestId = -1
            for (classId in selectedClassIds) {
                val score = dequantizedByteAt(
                    view,
                    (CLASS_OFFSET + classId) * ANCHOR_COUNT + anchor,
                    quant,
                    output0Type
                ).coerceIn(0f, 1f)
                if (score > best) {
                    best = score
                    bestId = classId
                }
            }

            if (best > maxScore) {
                maxScore = best
                maxScoreClassId = bestId
                maxScoreAnchor = anchor
            }

            if (best >= 0.01f) countGe001++
            if (best >= 0.05f) countGe005++
            if (best >= 0.10f) countGe010++
            if (best >= 0.20f) countGe020++
            if (best >= 0.30f) countGe030++
            if (best >= 0.50f) countGe050++
            if (best >= scoreThreshold) countGeThreshold++
        }

        // 每秒最多打印一次，避免 logcat 被刷爆
        val now = SystemClock.elapsedRealtimeNanos()
        if (now - lastThresholdDebugLogNs >= 1_000_000_000L) {
            lastThresholdDebugLogNs = now

            val maxClassName = if (maxScoreClassId >= 0) YoloClassCatalog.nameOf(maxScoreClassId) else "none"

            Log.w(
                TAG,
                "[YOLO_THRESHOLD_DEBUG] " +
                        "threshold=${formatMs(scoreThreshold)} " +
                        "maxScore=${formatMs(maxScore)} " +
                        "maxClass=$maxClassName " +
                        "maxAnchor=$maxScoreAnchor " +
                        ">=0.01=$countGe001 " +
                        ">=0.05=$countGe005 " +
                        ">=0.10=$countGe010 " +
                        ">=0.20=$countGe020 " +
                        ">=0.30=$countGe030 " +
                        ">=0.50=$countGe050 " +
                        ">=threshold=$countGeThreshold"
            )
        }

        // ============================================================
        // 原有正式 decode 逻辑
        // ============================================================
        for (anchor in 0 until ANCHOR_COUNT) {
            var bestId = -1
            var best = 0f

            for (classId in selectedClassIds) {
                val score = dequantizedByteAt(
                    view,
                    (CLASS_OFFSET + classId) * ANCHOR_COUNT + anchor,
                    quant,
                    output0Type
                ).coerceIn(0f, 1f)
                if (score > best) {
                    best = score
                    bestId = classId
                }
            }

            if (bestId < 0 || best < scoreThreshold) continue

            val cx = dequantizedByteAt(
                view,
                anchor,
                quant,
                output0Type
            ) * MODEL_SIZE

            val cy = dequantizedByteAt(
                view,
                ANCHOR_COUNT + anchor,
                quant,
                output0Type
            ) * MODEL_SIZE

            val w = dequantizedByteAt(
                view,
                2 * ANCHOR_COUNT + anchor,
                quant,
                output0Type
            ) * MODEL_SIZE

            val h = dequantizedByteAt(
                view,
                3 * ANCHOR_COUNT + anchor,
                quant,
                output0Type
            ) * MODEL_SIZE

            if (
                !cx.isFinite() ||
                !cy.isFinite() ||
                !w.isFinite() ||
                !h.isFinite() ||
                w <= 1f ||
                h <= 1f
            ) {
                continue
            }

            val left = (cx - 0.5f * w)
                .coerceIn(0f, MODEL_SIZE.toFloat())

            val top = (cy - 0.5f * h)
                .coerceIn(0f, MODEL_SIZE.toFloat())

            val right = (cx + 0.5f * w)
                .coerceIn(0f, MODEL_SIZE.toFloat())

            val bottom = (cy + 0.5f * h)
                .coerceIn(0f, MODEL_SIZE.toFloat())

            if (right - left < 2f || bottom - top < 2f) continue

            candidates += Candidate(
                bestId,
                best,
                RectF(left, top, right, bottom),
                anchor
            )
        }

        Log.d(
            TAG,
            "[YOLO_THRESHOLD_DEBUG] decodedCandidates=${candidates.size}"
        )

        return candidates
    }

    private fun dequantizedByteAt(buffer: ByteBuffer, elementIndex: Int, quant: TensorQuant, type: DataType): Float {
        val raw = if (type == DataType.INT8) buffer.get(elementIndex).toInt() else buffer.get(elementIndex).toInt() and 0xFF
        return (raw - quant.zeroPoint) * quant.scale
    }

    private fun decodeCandidates(
        det: Array<FloatArray>,
        scoreThreshold: Float
    ): List<Candidate> {

        val candidates = ArrayList<Candidate>(ANCHOR_COUNT / 2)

        // ============================================================
        // FP32 YOLO threshold diagnostics
        //
        // 用来确认：
        // 1. FP32 原始输出中的最高 person / car score
        // 2. 不同阈值下有多少 anchor
        // 3. 到底是不是 scoreThreshold 导致 detections=0
        //
        // 不改变原有 decode 行为。
        // ============================================================
        var maxPersonScore = 0f
        var maxCarScore = 0f
        var maxTargetScore = 0f
        var maxTargetClassId = -1
        var maxTargetAnchor = -1

        var personGe001 = 0
        var personGe005 = 0
        var personGe010 = 0
        var personGe020 = 0
        var personGe030 = 0
        var personGe050 = 0

        var carGe001 = 0
        var carGe005 = 0
        var carGe010 = 0
        var carGe020 = 0
        var carGe030 = 0
        var carGe050 = 0

        var targetGeThreshold = 0

        for (anchor in 0 until ANCHOR_COUNT) {

            val personScore = det[CLASS_OFFSET + 0][anchor].coerceIn(0f, 1f)
            val carScore = det[CLASS_OFFSET + 2][anchor].coerceIn(0f, 1f)

            if (personScore > maxPersonScore) {
                maxPersonScore = personScore
            }

            if (carScore > maxCarScore) {
                maxCarScore = carScore
            }

            if (personScore >= 0.01f) personGe001++
            if (personScore >= 0.05f) personGe005++
            if (personScore >= 0.10f) personGe010++
            if (personScore >= 0.20f) personGe020++
            if (personScore >= 0.30f) personGe030++
            if (personScore >= 0.50f) personGe050++

            if (carScore >= 0.01f) carGe001++
            if (carScore >= 0.05f) carGe005++
            if (carScore >= 0.10f) carGe010++
            if (carScore >= 0.20f) carGe020++
            if (carScore >= 0.30f) carGe030++
            if (carScore >= 0.50f) carGe050++

            var bestScore = 0f
            var bestClassId = -1
            for (classId in selectedClassIds) {
                val score = det[CLASS_OFFSET + classId][anchor].coerceIn(0f, 1f)
                if (score > bestScore) {
                    bestScore = score
                    bestClassId = classId
                }
            }

            if (bestScore > maxTargetScore) {
                maxTargetScore = bestScore
                maxTargetClassId = bestClassId
                maxTargetAnchor = anchor
            }

            if (bestScore >= scoreThreshold) {
                targetGeThreshold++
            }
        }

        val now = SystemClock.elapsedRealtimeNanos()
        if (now - lastThresholdDebugLogNs >= 1_000_000_000L) {
            lastThresholdDebugLogNs = now

            val maxClassName = if (maxTargetClassId >= 0) YoloClassCatalog.nameOf(maxTargetClassId) else "none"

            Log.w(
                TAG,
                "[YOLO_FP32_THRESHOLD_DEBUG] " +
                        "threshold=$scoreThreshold " +
                        "maxPerson=$maxPersonScore " +
                        "maxCar=$maxCarScore " +
                        "maxTarget=$maxTargetScore " +
                        "maxClass=$maxClassName " +
                        "maxAnchor=$maxTargetAnchor " +
                        "target>=threshold=$targetGeThreshold " +
                        "person[.01/.05/.10/.20/.30/.50]=" +
                        "$personGe001/$personGe005/$personGe010/$personGe020/$personGe030/$personGe050 " +
                        "car[.01/.05/.10/.20/.30/.50]=" +
                        "$carGe001/$carGe005/$carGe010/$carGe020/$carGe030/$carGe050"
            )
        }

        // ============================================================
        // 原有正式 decode 逻辑
        // ============================================================
        for (anchor in 0 until ANCHOR_COUNT) {
            val best = bestTargetClass(det, anchor)

            if (best.classId < 0 || best.confidence < scoreThreshold) {
                continue
            }

            val cx = det[0][anchor] * MODEL_SIZE
            val cy = det[1][anchor] * MODEL_SIZE
            val w = det[2][anchor] * MODEL_SIZE
            val h = det[3][anchor] * MODEL_SIZE

            if (
                !cx.isFinite() ||
                !cy.isFinite() ||
                !w.isFinite() ||
                !h.isFinite() ||
                w <= 1f ||
                h <= 1f
            ) {
                continue
            }

            val left = (cx - 0.5f * w)
                .coerceIn(0f, MODEL_SIZE.toFloat())

            val top = (cy - 0.5f * h)
                .coerceIn(0f, MODEL_SIZE.toFloat())

            val right = (cx + 0.5f * w)
                .coerceIn(0f, MODEL_SIZE.toFloat())

            val bottom = (cy + 0.5f * h)
                .coerceIn(0f, MODEL_SIZE.toFloat())

            if (right - left < 2f || bottom - top < 2f) {
                continue
            }

            candidates += Candidate(
                best.classId,
                best.confidence,
                RectF(left, top, right, bottom),
                anchor
            )
        }

        Log.d(
            TAG,
            "[YOLO_FP32_THRESHOLD_DEBUG] decodedCandidates=${candidates.size}"
        )

        return candidates
    }

    private fun bestTargetClass(det: Array<FloatArray>, anchor: Int): ClassScore {
        var bestId = -1; var best = 0f
        for (classId in selectedClassIds) {
            val score = det[CLASS_OFFSET + classId][anchor].coerceIn(0f, 1f)
            if (score > best) { best = score; bestId = classId }
        }
        return ClassScore(bestId, best)
    }

    private fun nms(candidates: List<Candidate>, iouThreshold: Float): List<Candidate> {
        val sorted = candidates.sortedByDescending { it.confidence }
        val kept = ArrayList<Candidate>(sorted.size)
        for (candidate in sorted) {
            var suppressed = false
            for (existing in kept) {
                if (existing.classId != candidate.classId) continue
                if (iou(existing.boxInput, candidate.boxInput) > iouThreshold) { suppressed = true; break }
            }
            if (!suppressed) kept += candidate
        }
        return kept
    }

    private fun decodeLocalMaskFloat32NativeFallback(detBuffer: ByteBuffer, protoBuffer: ByteBuffer, anchorIndex: Int, boxInput: RectF): SegmentationMask {
        val pixels = ByteArray(MASK_OUTPUT_SIZE * MASK_OUTPUT_SIZE)
        if (NativeLockedMaskKernel.isAvailable()) {
            val area = NativeLockedMaskKernel.decodeFloatMask(detBuffer, protoBuffer, anchorIndex, boxInput.left, boxInput.top, boxInput.right, boxInput.bottom, MODEL_SIZE, PROTO_SIZE, PROTO_CHANNELS, MASK_OUTPUT_SIZE, MASK_THRESHOLD, pixels)
            if (area >= 0) return SegmentationMask(MASK_OUTPUT_SIZE, MASK_OUTPUT_SIZE, pixels, area, MASK_THRESHOLD)
        }
        return decodeLocalMask(detOutput[0], protoOutput[0], anchorIndex, boxInput)
    }

    private fun decodeLocalMask(det: Array<FloatArray>, proto: Array<Array<FloatArray>>, anchorIndex: Int, boxInput: RectF): SegmentationMask {
        val pixels = ByteArray(MASK_OUTPUT_SIZE * MASK_OUTPUT_SIZE)
        var area = 0
        val x1 = (boxInput.left / MODEL_SIZE * PROTO_SIZE).coerceIn(0f, PROTO_SIZE - 1f)
        val y1 = (boxInput.top / MODEL_SIZE * PROTO_SIZE).coerceIn(0f, PROTO_SIZE - 1f)
        val x2 = (boxInput.right / MODEL_SIZE * PROTO_SIZE).coerceIn(x1 + 1f, PROTO_SIZE.toFloat())
        val y2 = (boxInput.bottom / MODEL_SIZE * PROTO_SIZE).coerceIn(y1 + 1f, PROTO_SIZE.toFloat())
        val coefficients = FloatArray(PROTO_CHANNELS)
        for (k in 0 until PROTO_CHANNELS) coefficients[k] = det[MASK_COEFF_OFFSET + k][anchorIndex]

        for (row in 0 until MASK_OUTPUT_SIZE) {
            val py = (y1 + (y2 - y1) * ((row + 0.5f) / MASK_OUTPUT_SIZE)).toInt().coerceIn(0, PROTO_SIZE - 1)
            for (col in 0 until MASK_OUTPUT_SIZE) {
                val px = (x1 + (x2 - x1) * ((col + 0.5f) / MASK_OUTPUT_SIZE)).toInt().coerceIn(0, PROTO_SIZE - 1)
                var logit = 0f
                for (k in 0 until PROTO_CHANNELS) logit += coefficients[k] * proto[k][py][px]
                if (sigmoid(logit) >= MASK_THRESHOLD) { pixels[row * MASK_OUTPUT_SIZE + col] = 1; area++ }
            }
        }
        return SegmentationMask(MASK_OUTPUT_SIZE, MASK_OUTPUT_SIZE, pixels, area, MASK_THRESHOLD)
    }

    private fun decodeLocalMaskQuantized(detBuffer: ByteBuffer, detQuant: TensorQuant, protoBuffer: ByteBuffer, protoQuant: TensorQuant, anchorIndex: Int, boxInput: RectF): SegmentationMask {
        val pixels = ByteArray(MASK_OUTPUT_SIZE * MASK_OUTPUT_SIZE)
        if (NativeLockedMaskKernel.isAvailable()) {
            val area = NativeLockedMaskKernel.decodeQuantizedMask(detBuffer, detQuant.scale, detQuant.zeroPoint, protoBuffer, protoQuant.scale, protoQuant.zeroPoint, if (output0Type == DataType.INT8) 0 else 1, if (output1Type == DataType.INT8) 0 else 1, anchorIndex, boxInput.left, boxInput.top, boxInput.right, boxInput.bottom, MODEL_SIZE, PROTO_SIZE, PROTO_CHANNELS, MASK_OUTPUT_SIZE, MASK_THRESHOLD, pixels)
            if (area >= 0) return SegmentationMask(MASK_OUTPUT_SIZE, MASK_OUTPUT_SIZE, pixels, area, MASK_THRESHOLD)
        }
        return SegmentationMask(MASK_OUTPUT_SIZE, MASK_OUTPUT_SIZE, pixels, 0, MASK_THRESHOLD)
    }

    // 【核心修正】：用 Matrix 直接在 Letterbox 阶段画出旋转后的画面，彻底消灭 Rotate 内存泄漏
    private fun letterbox(source: Bitmap, size: Int, rotation: Int): Letterbox {
        val swapDims = rotation == 90 || rotation == 270
        val rotatedW = if (swapDims) source.height else source.width
        val rotatedH = if (swapDims) source.width else source.height

        val scale = min(size.toFloat() / rotatedW.coerceAtLeast(1), size.toFloat() / rotatedH.coerceAtLeast(1))
        val newW = max(1, kotlin.math.round(rotatedW * scale).toInt())
        val newH = max(1, kotlin.math.round(rotatedH * scale).toInt())
        val padX = (size - newW) / 2f
        val padY = (size - newH) / 2f

        letterboxCanvas.drawColor(android.graphics.Color.rgb(114, 114, 114))

        val matrix = Matrix()
        matrix.postTranslate(-source.width / 2f, -source.height / 2f)
        if (rotation != 0) {
            matrix.postRotate(rotation.toFloat())
        }
        matrix.postScale(scale, scale)
        matrix.postTranslate(padX + newW / 2f, padY + newH / 2f)

        letterboxCanvas.drawBitmap(source, matrix, resizePaint)

        return Letterbox(letterboxBitmap, scale, padX, padY, rotatedW, rotatedH)
    }

    private fun mapBoxToOriginal(box: RectF, lb: Letterbox, width: Int, height: Int): RectF {
        return RectF(
            ((box.left - lb.padX) / lb.scale).coerceIn(0f, width.toFloat()),
            ((box.top - lb.padY) / lb.scale).coerceIn(0f, height.toFloat()),
            ((box.right - lb.padX) / lb.scale).coerceIn(0f, width.toFloat()),
            ((box.bottom - lb.padY) / lb.scale).coerceIn(0f, height.toFloat())
        )
    }

    private fun rotateBitmap(source: Bitmap, rotation: Int): Bitmap? {
        if (rotation == 0) return source
        return try { Bitmap.createBitmap(source, 0, 0, source.width, source.height, Matrix().apply { postRotate(rotation.toFloat()) }, true) } catch (_: Throwable) { null }
    }

    private fun iou(a: RectF, b: RectF): Float {
        val inter = max(0f, min(a.right, b.right) - max(a.left, b.left)) * max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        if (inter <= 0f) return 0f
        return inter / (a.width() * a.height() + b.width() * b.height() - inter)
    }
    fun stats(): InferenceStats = lastStats
    fun backendName(): String = activeBackendName
    fun close() {
        if (destroyed) return
        destroyed = true
        try { tfliteHandle?.close() } catch (_: Throwable) {}
        try { if (!letterboxBitmap.isRecycled) letterboxBitmap.recycle() } catch (_: Throwable) {}
    }

    private fun loadLabels(): List<String> = try { appContext.assets.open(LABEL_ASSET).bufferedReader().useLines { it.toList() } } catch (_: Throwable) { YoloClassCatalog.NAMES }
    private fun fallbackLabel(classId: Int): String = YoloClassCatalog.nameOf(classId)
    private fun isSelectedClass(classId: Int): Boolean = selectedClassIds.binarySearch(classId) >= 0

    private fun normalizeRotation(rotationDegrees: Int): Int = ((rotationDegrees % 360) + 360) % 360
    private fun sigmoid(x: Float): Float = if (x >= 0f) 1f / (1f + exp(-x.toDouble()).toFloat()) else { val e = exp(x.toDouble()).toFloat(); e / (1f + e) }
    private fun elapsedMs(startNs: Long): Float = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
    private fun formatMs(value: Float): String = String.format(java.util.Locale.US, "%.3f", value)

    private companion object {
        const val TAG = "YOLO11Seg"
        const val DEFAULT_MODEL_ASSET = "yolo11n-seg.tflite"
        const val LABEL_ASSET = "labels.txt"
        const val MODEL_SIZE = 640
        const val DET_CHANNELS = 116
        const val ANCHOR_COUNT = 8400

        // 补上丢失的这三个常量
        const val NUM_CLASSES = 80
        const val CLASS_OFFSET = 4
        const val MASK_COEFF_OFFSET = 84
        const val PROTO_CHANNELS = 32
        const val PROTO_SIZE = 160
        const val MASK_OUTPUT_SIZE = 64
        const val MASK_THRESHOLD = 0.5f
        const val NMS_IOU = 0.45f
        const val MIN_BOX_PX = 4f
        val DEFAULT_TARGET_CLASS_IDS: Set<Int> = YoloClassCatalog.DEFAULT_CLASS_IDS
    }
}