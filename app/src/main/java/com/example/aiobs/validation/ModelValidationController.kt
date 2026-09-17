package com.example.aiobs.validation

import android.content.Context
import android.os.Build
import android.os.Debug
import android.util.Log
import com.example.aiobs.ai.TfliteAcceleration
import com.example.aiobs.ai.XFeatPostProcessor
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Isolated on-device model validation.
 *
 * This class never touches the live tracker/recovery path.
 *
 * Validates independently:
 *   1) YOLO11n-seg
 *   2) XFeat
 *   3) Depth Anything V2 Small
 *   4) Standard OSNet x1.0 FP16
 *
 * OSNet contract:
 *   input  = FLOAT32 [1,3,256,128] NCHW
 *   output = FLOAT32 [1,512]
 *   model  = FP16-weight TFLite compiled for Snapdragon 8 Gen 2 / HTP
 */
class ModelValidationController(
    private val context: Context
) {

    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "AIOBS-Model-Validation").apply {
            isDaemon = true
            priority = Thread.NORM_PRIORITY
        }
    }

    @Volatile
    private var running = false

    fun validateAsync(callback: (ModelValidationReport) -> Unit) {
        synchronized(this) {
            if (running) return
            running = true
        }

        executor.execute {
            try {
                callback(validateInternal())
            } catch (t: Throwable) {
                Log.e(TAG, "Model validation failed", t)
                callback(
                    ModelValidationReport(
                        generatedAtMs = System.currentTimeMillis(),
                        deviceSummary = deviceSummary(),
                        yolo11nSeg = failedCheck(YOLO_ASSET, "YOLO11n-seg", t),
                        xfeat = failedCheck(XFEAT_ASSET, "XFeat", t),
                        depth = failedCheck(DEPTH_ASSET, "Depth Anything V2 Small", t),
                        osnet = failedCheck(OSNET_ASSET, "OSNet x1.0 FP16", t),
                        overallPassed = false,
                        notes = listOf(
                            "Validation controller failed before completing all checks."
                        )
                    )
                )
            } finally {
                running = false
            }
        }
    }

    fun isRunning(): Boolean = running

    fun shutdown() {
        executor.shutdownNow()
    }

    private fun validateInternal(): ModelValidationReport {
        val yolo = validateYolo11nSeg()
        val xfeat = validateXFeat()
        val depth = validateDepth()
        val osnet = validateOsnet()

        val notes = buildList {
            add("Validation is isolated from the live tracker/recovery pipeline.")
            add("Each model is loaded independently and the report records the actual backend selected by TfliteAcceleration.")
            add("YOLO11n-seg is validated as a raw tensor contract plus output sanity probe.")
            add("XFeat validation separates LiteRT inference from raw-output decoding and Android post-processing sanity.")
            add("Depth Anything V2 validation is tensor-type aware and supports FLOAT32 plus INT8/UINT8 I/O when exposed by the asset.")
            add("OSNet is the standard OSNet x1.0 model; the deployment asset is the Galaxy S23 FP16 TFLite model, not OSNet-IBN.")
            add("OSNet preprocessing for the deployment model is RGB / 255, NCHW, with no ImageNet mean/std normalization.")
        }

        return ModelValidationReport(
            generatedAtMs = System.currentTimeMillis(),
            deviceSummary = deviceSummary(),
            yolo11nSeg = yolo,
            xfeat = xfeat,
            depth = depth,
            osnet = osnet,
            overallPassed =
                yolo.passed &&
                    xfeat.passed &&
                    depth.passed &&
                    osnet.passed,
            notes = notes
        )
    }

    // -------------------------------------------------------------------------
    // YOLO11n-seg
    // -------------------------------------------------------------------------

    private fun validateYolo11nSeg(): ModelValidationReport.ModelCheck {
        val asset = loadAsset(YOLO_ASSET)
            ?: return missingCheck("YOLO11n-seg", YOLO_ASSET)

        var handle: TfliteAcceleration.InterpreterHandle? = null

        try {
            val loadStart = System.nanoTime()
            val created = TfliteAcceleration.create(
                context = context,
                modelBuffer = asset.buffer,
                numThreads = 2,
                preferQnn = true,
                preferGpu = false,
                tag = YOLO_TAG
            )
            handle = created

            val interpreter = created.interpreter
            val backend = created.backend.name
            val loadMs = elapsedMs(loadStart)

            val allocStart = System.nanoTime()
            interpreter.allocateTensors()
            val allocationMs = elapsedMs(allocStart)

            val input = interpreter.getInputTensor(0)
            val outputs = (0 until interpreter.outputTensorCount).map {
                interpreter.getOutputTensor(it)
            }

            val inputOk =
                input.dataType() == DataType.FLOAT32 &&
                    input.shape().contentEquals(intArrayOf(1, 3, 640, 640))

            val outputOk =
                outputs.size == 2 &&
                    outputs[0].dataType() == DataType.FLOAT32 &&
                    outputs[1].dataType() == DataType.FLOAT32 &&
                    outputs[0].shape().contentEquals(intArrayOf(1, 116, 8400)) &&
                    outputs[1].shape().contentEquals(intArrayOf(1, 32, 160, 160))

            val inputSummary = "${shapeString(input.shape())} ${input.dataType()}"
            val outputSummary = outputs.joinToString(" | ") {
                "${shapeString(it.shape())} ${it.dataType()}"
            }

            if (!inputOk || !outputOk) {
                return ModelValidationReport.ModelCheck(
                    name = "YOLO11n-seg",
                    assetName = YOLO_ASSET,
                    assetPresent = true,
                    assetSizeBytes = asset.sizeBytes,
                    loadMs = loadMs,
                    allocationMs = allocationMs,
                    warmupCount = 0,
                    benchmarkCount = 0,
                    meanMs = 0.0,
                    medianMs = 0.0,
                    minMs = 0.0,
                    maxMs = 0.0,
                    inputSummary = inputSummary,
                    outputSummary = outputSummary,
                    sanitySummary = "NOT RUN",
                    contractSummary = "FAIL: expected [1,3,640,640] FLOAT32 -> [1,116,8400] + [1,32,160,160] FLOAT32",
                    passed = false,
                    warnings = listOf("YOLO11n-seg tensor contract mismatch."),
                    backend = backend
                )
            }

            val inputBuffer = deterministicYoloInput(input.shape())
            val warmupCount = 2
            val benchmarkCount = 5

            repeat(warmupCount) {
                interpreter.runForMultipleInputsOutputs(
                    arrayOf(inputBuffer.duplicateForRun()),
                    allocateOutputBuffers(interpreter)
                )
            }

            val times = DoubleArray(benchmarkCount)
            var lastOutputs: MutableMap<Int, Any> = HashMap()

            for (i in 0 until benchmarkCount) {
                val outputsForRun = allocateOutputBuffers(interpreter)
                val start = System.nanoTime()
                interpreter.runForMultipleInputsOutputs(
                    arrayOf(inputBuffer.duplicateForRun()),
                    outputsForRun
                )
                times[i] = elapsedMsDouble(start)
                lastOutputs = outputsForRun
            }

            val output0 = (lastOutputs[0] as? ByteBuffer)?.let(::bufferToFloatArray)
                ?: return failedCheck(
                    YOLO_ASSET,
                    "YOLO11n-seg",
                    IllegalStateException("YOLO output0 buffer missing"),
                    asset.sizeBytes,
                    backend
                )

            val output1 = (lastOutputs[1] as? ByteBuffer)?.let(::bufferToFloatArray)
                ?: return failedCheck(
                    YOLO_ASSET,
                    "YOLO11n-seg",
                    IllegalStateException("YOLO output1 buffer missing"),
                    asset.sizeBytes,
                    backend
                )

            val sanity = inspectYoloOutputs(output0, output1)
            val stats = summarizeTimes(times)

            return ModelValidationReport.ModelCheck(
                name = "YOLO11n-seg",
                assetName = YOLO_ASSET,
                assetPresent = true,
                assetSizeBytes = asset.sizeBytes,
                loadMs = loadMs,
                allocationMs = allocationMs,
                warmupCount = warmupCount,
                benchmarkCount = benchmarkCount,
                meanMs = stats.mean,
                medianMs = stats.median,
                minMs = stats.min,
                maxMs = stats.max,
                inputSummary = inputSummary,
                outputSummary = outputSummary,
                sanitySummary = sanity.summary,
                contractSummary = if (sanity.passed) {
                    "PASS: YOLO11n-seg tensor contract and output sanity passed."
                } else {
                    "FAIL: YOLO11n-seg execution completed but output sanity failed."
                },
                passed = sanity.passed,
                warnings = sanity.warnings,
                backend = backend
            )
        } catch (t: Throwable) {
            Log.e(TAG, "YOLO11n-seg validation failed", t)
            return failedCheck(YOLO_ASSET, "YOLO11n-seg", t, asset.sizeBytes, handle?.backend?.name ?: "CPU")
        } finally {
            try { handle?.close() } catch (_: Throwable) {}
        }
    }

    private fun inspectYoloOutputs(
        detection: FloatArray,
        proto: FloatArray
    ): SanityResult {
        val warnings = mutableListOf<String>()

        if (detection.size != 116 * 8400) {
            return SanityResult(false, "output0=${detection.size}, expected=${116 * 8400}", warnings)
        }

        if (proto.size != 32 * 160 * 160) {
            return SanityResult(false, "output1=${proto.size}, expected=${32 * 160 * 160}", warnings)
        }

        if (!allFinite(detection) || !allFinite(proto)) {
            return SanityResult(false, "non-finite YOLO tensor value detected", warnings)
        }

        val detAbsMax = detection.maxOf { abs(it) }
        val protoAbsMax = proto.maxOf { abs(it) }
        val detMean = detection.average().toFloat()
        val protoMean = proto.average().toFloat()
        val detRms = rms(detection)
        val protoRms = rms(proto)
        val detNonZero = detection.count { abs(it) > 1e-5f }
        val protoNonZero = proto.count { abs(it) > 1e-5f }

        if (detAbsMax < 1e-6f) warnings += "YOLO detection output is nearly all zero."
        if (protoRms < 1e-5f) warnings += "YOLO prototype tensor is nearly flat."

        val summary =
            "finite=PASS | " +
                "det[min=${fmt4(detection.minOrNull() ?: 0f)}, max=${fmt4(detection.maxOrNull() ?: 0f)}, " +
                "mean=${fmt4(detMean)}, rms=${fmt4(detRms)}] | " +
                "proto[min=${fmt4(proto.minOrNull() ?: 0f)}, max=${fmt4(proto.maxOrNull() ?: 0f)}, " +
                "mean=${fmt4(protoMean)}, rms=${fmt4(protoRms)}] | " +
                "detAbsMax=${fmt4(detAbsMax)} | protoAbsMax=${fmt4(protoAbsMax)} | " +
                "detNonZero=$detNonZero/${detection.size} | protoNonZero=$protoNonZero/${proto.size}"

        return SanityResult(true, summary, warnings)
    }

    // -------------------------------------------------------------------------
    // XFeat
    // -------------------------------------------------------------------------

    private fun validateXFeat(): ModelValidationReport.ModelCheck {
        val asset = loadAsset(XFEAT_ASSET)
            ?: return missingCheck("XFeat", XFEAT_ASSET)

        var handle: TfliteAcceleration.InterpreterHandle? = null

        try {
            val loadStart = System.nanoTime()
            val created = TfliteAcceleration.create(
                context = context,
                modelBuffer = asset.buffer,
                numThreads = 2,
                preferQnn = false,
                preferGpu = true,
                tag = XFEAT_TAG
            )
            handle = created

            val interpreter = created.interpreter
            val backend = created.backend.name
            val loadMs = elapsedMs(loadStart)

            val allocStart = System.nanoTime()
            interpreter.allocateTensors()
            val allocationMs = elapsedMs(allocStart)

            val input = interpreter.getInputTensor(0)
            val outputs = (0 until interpreter.outputTensorCount).map {
                interpreter.getOutputTensor(it)
            }

            val inputOk =
                input.dataType() == DataType.FLOAT32 &&
                    input.shape().contentEquals(intArrayOf(1, 640, 640, 1))

            val outputOk =
                outputs.size == 3 &&
                    outputs.all { it.dataType() == DataType.FLOAT32 } &&
                    outputs[0].shape().contentEquals(intArrayOf(1, 64, 80, 80)) &&
                    outputs[1].shape().contentEquals(intArrayOf(1, 65, 80, 80)) &&
                    outputs[2].shape().contentEquals(intArrayOf(1, 1, 80, 80))

            val inputSummary = "${shapeString(input.shape())} ${input.dataType()}"
            val outputSummary = outputs.joinToString(" | ") {
                "${shapeString(it.shape())} ${it.dataType()}"
            }

            if (!inputOk || !outputOk) {
                return ModelValidationReport.ModelCheck(
                    name = "XFeat",
                    assetName = XFEAT_ASSET,
                    assetPresent = true,
                    assetSizeBytes = asset.sizeBytes,
                    loadMs = loadMs,
                    allocationMs = allocationMs,
                    warmupCount = 0,
                    benchmarkCount = 0,
                    meanMs = 0.0,
                    medianMs = 0.0,
                    minMs = 0.0,
                    maxMs = 0.0,
                    inputSummary = inputSummary,
                    outputSummary = outputSummary,
                    sanitySummary = "NOT RUN",
                    contractSummary = "FAIL: XFeat tensor contract mismatch.",
                    passed = false,
                    warnings = listOf("XFeat tensor contract mismatch."),
                    backend = backend
                )
            }

            val inputBuffer = deterministicXFeatInput(input.shape())
            val warmupCount = 2
            val benchmarkCount = 5

            repeat(warmupCount) {
                interpreter.runForMultipleInputsOutputs(
                    arrayOf(inputBuffer.duplicateForRun()),
                    allocateOutputBuffers(interpreter)
                )
            }

            val inferenceTimes = DoubleArray(benchmarkCount)
            val decodeTimes = DoubleArray(benchmarkCount)
            val postTimes = DoubleArray(benchmarkCount)
            val totalTimes = DoubleArray(benchmarkCount)
            var lastOutputs: MutableMap<Int, Any> = HashMap()
            var lastPost: XFeatPostProcessor.Result? = null
            var lastFeat = FloatArray(0)
            var lastLogits = FloatArray(0)
            var lastHeatmap = FloatArray(0)

            for (i in 0 until benchmarkCount) {
                val totalStart = System.nanoTime()
                val outputsForRun = allocateOutputBuffers(interpreter)

                val inferenceStart = System.nanoTime()
                interpreter.runForMultipleInputsOutputs(
                    arrayOf(inputBuffer.duplicateForRun()),
                    outputsForRun
                )
                inferenceTimes[i] = elapsedMsDouble(inferenceStart)

                val decodeStart = System.nanoTime()
                val featBuffer = outputsForRun[0] as? ByteBuffer
                val logitsBuffer = outputsForRun[1] as? ByteBuffer
                val heatmapBuffer = outputsForRun[2] as? ByteBuffer
                if (featBuffer == null || logitsBuffer == null || heatmapBuffer == null) {
                    return failedCheck(
                        XFEAT_ASSET,
                        "XFeat",
                        IllegalStateException("XFeat output buffer missing"),
                        asset.sizeBytes,
                        backend
                    )
                }

                lastFeat = bufferToFloatArray(featBuffer)
                lastLogits = bufferToFloatArray(logitsBuffer)
                lastHeatmap = bufferToFloatArray(heatmapBuffer)
                decodeTimes[i] = elapsedMsDouble(decodeStart)

                val postStart = System.nanoTime()
                lastPost = XFeatPostProcessor.process(
                    feats = lastFeat,
                    keypointLogits = lastLogits,
                    reliability = lastHeatmap,
                    imageWidth = 640,
                    imageHeight = 640,
                    modelSize = 640,
                    topK = 256,
                    detectionThreshold = 0.05f
                )
                postTimes[i] = elapsedMsDouble(postStart)

                totalTimes[i] = elapsedMsDouble(totalStart)
                lastOutputs = outputsForRun
            }

            val decodedPost = lastPost
                ?: return failedCheck(
                    XFEAT_ASSET,
                    "XFeat",
                    IllegalStateException("No XFeat post-processing result"),
                    asset.sizeBytes,
                    backend
                )

            val sanity = inspectXFeatOutputs(
                lastFeat,
                lastLogits,
                lastHeatmap,
                decodedPost
            )
            val inferenceStats = summarizeTimes(inferenceTimes)
            val decodeStats = summarizeTimes(decodeTimes)
            val postStats = summarizeTimes(postTimes)
            val totalStats = summarizeTimes(totalTimes)

            Log.i(
                XFEAT_TAG,
                "XFeat timing backend=$backend " +
                    "inference=${fmtMs(inferenceStats.mean)} " +
                    "decode=${fmtMs(decodeStats.mean)} " +
                    "post=${fmtMs(postStats.mean)} " +
                    "total=${fmtMs(totalStats.mean)}"
            )

            return ModelValidationReport.ModelCheck(
                name = "XFeat",
                assetName = XFEAT_ASSET,
                assetPresent = true,
                assetSizeBytes = asset.sizeBytes,
                loadMs = loadMs,
                allocationMs = allocationMs,
                warmupCount = warmupCount,
                benchmarkCount = benchmarkCount,
                meanMs = totalStats.mean,
                medianMs = totalStats.median,
                minMs = totalStats.min,
                maxMs = totalStats.max,
                inputSummary = inputSummary,
                outputSummary = outputSummary,
                sanitySummary =
                    sanity.summary +
                        " | inferenceMean=${fmtMs(inferenceStats.mean)}" +
                        " | decodeMean=${fmtMs(decodeStats.mean)}" +
                        " | postMean=${fmtMs(postStats.mean)}",
                contractSummary = if (sanity.passed) {
                    "PASS: XFeat tensor contract and output sanity passed."
                } else {
                    "FAIL: XFeat execution completed but output sanity failed."
                },
                passed = sanity.passed,
                warnings = sanity.warnings,
                backend = backend
            )
        } catch (t: Throwable) {
            Log.e(TAG, "XFeat validation failed", t)
            return failedCheck(XFEAT_ASSET, "XFeat", t, asset.sizeBytes, handle?.backend?.name ?: "CPU")
        } finally {
            try { handle?.close() } catch (_: Throwable) {}
        }
    }

    private fun inspectXFeatOutputs(
        feat: FloatArray,
        logits: FloatArray,
        heatmap: FloatArray,
        post: XFeatPostProcessor.Result
    ): SanityResult {
        val warnings = mutableListOf<String>()
        if (!allFinite(feat) || !allFinite(logits) || !allFinite(heatmap)) {
            return SanityResult(false, "non-finite XFeat tensor value detected", warnings)
        }

        val heatMin = heatmap.minOrNull() ?: 0f
        val heatMax = heatmap.maxOrNull() ?: 0f
        val heatMean = heatmap.average().toFloat()
        if (heatMin < -1e-3f || heatMax > 1.001f) {
            warnings += "XFeat reliability heatmap outside expected [0,1]: min=$heatMin max=$heatMax"
        }

        val logitsAbs = logits.maxOfOrNull { abs(it) } ?: 0f
        val featRms = rms(feat)
        val strongCells = softmaxPeakCount(logits)
        if (post.selectedCount < 8) {
            warnings += "Synthetic probe produced only ${post.selectedCount} post-processed features."
        }

        val summary =
            "finite=PASS | " +
                "heatmap[min=${fmt4(heatMin)}, max=${fmt4(heatMax)}, mean=${fmt4(heatMean)}] | " +
                "logitAbsMax=${fmt4(logitsAbs)} | featureRMS=${fmt4(featRms)} | " +
                "strongKeypointCells=$strongCells | postFeatures=${post.selectedCount} | " +
                "meanScore=${fmt4(post.meanScore)} | postMs=${post.elapsedMs}"

        return SanityResult(true, summary, warnings)
    }

    private fun softmaxPeakCount(logits: FloatArray): Int {
        if (logits.size != 65 * 80 * 80) return 0

        val cells = 80 * 80
        var strong = 0
        for (cell in 0 until cells) {
            var maxLogit = -Float.MAX_VALUE
            for (c in 0 until 65) {
                maxLogit = maxOf(maxLogit, logits[c * cells + cell])
            }

            var sum = 0.0
            for (c in 0 until 65) {
                sum += exp((logits[c * cells + cell] - maxLogit).toDouble())
            }

            val peak = (1.0 / sum).toFloat()
            if (peak > 0.20f) strong++
        }
        return strong
    }

    // -------------------------------------------------------------------------
    // Depth Anything V2
    // -------------------------------------------------------------------------

    private fun validateDepth(): ModelValidationReport.ModelCheck {
        val asset = loadAsset(DEPTH_ASSET)
            ?: return missingCheck("Depth Anything V2 Small", DEPTH_ASSET)

        var handle: TfliteAcceleration.InterpreterHandle? = null

        try {
            // Probe the TFLite contract before creating QNN so that the correct
            // precision mode can be requested for quantized assets.
            val probe = Interpreter(
                asset.buffer.duplicateForRun(),
                Interpreter.Options().setNumThreads(1)
            )

            val probeType: DataType
            val probeShape: IntArray
            try {
                probe.allocateTensors()
                probeType = probe.getInputTensor(0).dataType()
                probeShape = probe.getInputTensor(0).shape().copyOf()
            } finally {
                try { probe.close() } catch (_: Throwable) {}
            }

            val qnnPrecision = if (
                isQuantizedType(probeType) ||
                    probeShape.contentEquals(intArrayOf(1, 3, 518, 518))
            ) {
                TfliteAcceleration.QnnHtpPrecision.QUANTIZED
            } else {
                TfliteAcceleration.QnnHtpPrecision.FP16
            }

            Log.i(
                DEPTH_TAG,
                "Depth probe type=$probeType shape=${probeShape.contentToString()} precision=$qnnPrecision"
            )

            val loadStart = System.nanoTime()
            val created = TfliteAcceleration.create(
                context = context,
                modelBuffer = asset.buffer.duplicateForRun(),
                numThreads = 2,
                preferQnn = true,
                preferGpu = false,
                qnnHtpPrecision = qnnPrecision,
                tag = DEPTH_TAG
            )
            handle = created

            val interpreter = created.interpreter
            val backend = created.backend.name
            val loadMs = elapsedMs(loadStart)

            val allocStart = System.nanoTime()
            interpreter.allocateTensors()
            val allocationMs = elapsedMs(allocStart)

            val input = interpreter.getInputTensor(0)
            val output = interpreter.getOutputTensor(0)
            val inputType = input.dataType()
            val outputType = output.dataType()
            val inputShape = input.shape()
            val outputShape = output.shape()
            val inputQuant = readQuantization(input)
            val outputQuant = readQuantization(output)

            val inputOk = isSupportedDepthInput(inputShape, inputType) &&
                (inputType == DataType.FLOAT32 || inputQuant != null)

            val outputOk = output.numElements() == 518 * 518 &&
                (outputType == DataType.FLOAT32 ||
                    (isQuantizedType(outputType) && outputQuant != null))

            val inputSummary = buildString {
                append(shapeString(inputShape))
                append(' ')
                append(inputType)
                if (inputQuant != null) {
                    append(" scale=${fmt8(inputQuant.scale)} zero=${inputQuant.zeroPoint}")
                }
            }

            val outputSummary = buildString {
                append(shapeString(outputShape))
                append(' ')
                append(outputType)
                if (outputQuant != null) {
                    append(" scale=${fmt8(outputQuant.scale)} zero=${outputQuant.zeroPoint}")
                }
            }

            if (!inputOk || !outputOk || backend != TfliteAcceleration.Backend.QNN.name) {
                return ModelValidationReport.ModelCheck(
                    name = "Depth Anything V2 Small",
                    assetName = DEPTH_ASSET,
                    assetPresent = true,
                    assetSizeBytes = asset.sizeBytes,
                    loadMs = loadMs,
                    allocationMs = allocationMs,
                    warmupCount = 0,
                    benchmarkCount = 0,
                    meanMs = 0.0,
                    medianMs = 0.0,
                    minMs = 0.0,
                    maxMs = 0.0,
                    inputSummary = inputSummary,
                    outputSummary = outputSummary,
                    sanitySummary = "NOT RUN",
                    contractSummary =
                        "FAIL: expected Depth 518x518 FLOAT32 or quantized input/output contract and QNN backend; actual backend=$backend",
                    passed = false,
                    warnings = listOf("Depth tensor contract/backend mismatch."),
                    backend = backend
                )
            }

            val inputBuffer = deterministicDepthInputBuffer(input, inputQuant)
            val outputBytes = output.numElements() * bytesPerElement(outputType)
            val outputBuffer = ByteBuffer
                .allocateDirect(outputBytes)
                .order(ByteOrder.nativeOrder())

            val warmupCount = 1
            repeat(warmupCount) {
                interpreter.run(
                    inputBuffer.duplicateForRun(),
                    outputBuffer.duplicateForRunForOutput()
                )
            }

            val benchmarkCount = 3
            val times = DoubleArray(benchmarkCount)
            var finalOutput: ByteBuffer? = null

            for (i in 0 until benchmarkCount) {
                val runOutput = outputBuffer.duplicateForRunForOutput()
                val start = System.nanoTime()
                interpreter.run(
                    inputBuffer.duplicateForRun(),
                    runOutput
                )
                times[i] = elapsedMsDouble(start)
                finalOutput = runOutput

                Log.i(
                    DEPTH_TAG,
                    "Depth benchmark[${i + 1}/$benchmarkCount] backend=$backend " +
                        "in=${fmtMs(times[i])}"
                )
            }

            val outputForSanity = finalOutput
                ?: return failedCheck(
                    DEPTH_ASSET,
                    "Depth Anything V2 Small",
                    IllegalStateException("Depth output buffer missing"),
                    asset.sizeBytes,
                    backend
                )

            val depth = depthByteBufferToFloatArray(
                outputForSanity,
                outputType,
                outputQuant
            )

            val sanity = inspectDepthOutput(depth)
            val stats = summarizeTimes(times)

            return ModelValidationReport.ModelCheck(
                name = "Depth Anything V2 Small",
                assetName = DEPTH_ASSET,
                assetPresent = true,
                assetSizeBytes = asset.sizeBytes,
                loadMs = loadMs,
                allocationMs = allocationMs,
                warmupCount = warmupCount,
                benchmarkCount = benchmarkCount,
                meanMs = stats.mean,
                medianMs = stats.median,
                minMs = stats.min,
                maxMs = stats.max,
                inputSummary = inputSummary,
                outputSummary = outputSummary,
                sanitySummary = sanity.summary,
                contractSummary = if (sanity.passed) {
                    "PASS: Depth Anything V2 executed through QNN/HTP with tensor-aware I/O."
                } else {
                    "FAIL: Depth execution completed but output sanity failed."
                },
                passed = sanity.passed,
                warnings = sanity.warnings,
                backend = backend
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Depth validation failed", t)
            return failedCheck(
                DEPTH_ASSET,
                "Depth Anything V2 Small",
                t,
                asset.sizeBytes,
                handle?.backend?.name ?: "QNN"
            )
        } finally {
            try { handle?.close() } catch (_: Throwable) {}
        }
    }

    private fun isSupportedDepthInput(
        shape: IntArray,
        type: DataType
    ): Boolean {
        val fp32Nhwc = type == DataType.FLOAT32 &&
            shape.contentEquals(intArrayOf(1, 518, 518, 3))
        val fp32Nchw = type == DataType.FLOAT32 &&
            shape.contentEquals(intArrayOf(1, 3, 518, 518))
        val quantNchw = isQuantizedType(type) &&
            shape.contentEquals(intArrayOf(1, 3, 518, 518))
        return fp32Nhwc || fp32Nchw || quantNchw
    }

    private fun deterministicDepthInputBuffer(
        input: Tensor,
        quant: QuantizationInfo?
    ): ByteBuffer {
        val shape = input.shape()
        val type = input.dataType()
        val nchw = shape.contentEquals(intArrayOf(1, 3, 518, 518))
        val height = if (nchw) shape[2] else shape[1]
        val width = if (nchw) shape[3] else shape[2]
        val channels = if (nchw) shape[1] else shape[3]

        val buffer = ByteBuffer
            .allocateDirect(input.numElements() * bytesPerElement(type))
            .order(ByteOrder.nativeOrder())

        fun value(c: Int, x: Int, y: Int): Float {
            val xf = x / (width - 1f)
            val yf = y / (height - 1f)
            return when (c) {
                0 -> xf
                1 -> yf
                else -> ((xf + yf) * 0.5f).coerceIn(0f, 1f)
            }
        }

        if (type == DataType.FLOAT32) {
            if (nchw) {
                for (c in 0 until channels) {
                    for (y in 0 until height) {
                        for (x in 0 until width) {
                            buffer.putFloat(value(c, x, y))
                        }
                    }
                }
            } else {
                for (y in 0 until height) {
                    for (x in 0 until width) {
                        for (c in 0 until channels) {
                            buffer.putFloat(value(c, x, y))
                        }
                    }
                }
            }
        } else {
            val q = quant ?: error("Quantization parameters missing for $type depth input")
            if (nchw) {
                for (c in 0 until channels) {
                    for (y in 0 until height) {
                        for (x in 0 until width) {
                            putQuantized(buffer, value(c, x, y), q, type)
                        }
                    }
                }
            } else {
                for (y in 0 until height) {
                    for (x in 0 until width) {
                        for (c in 0 until channels) {
                            putQuantized(buffer, value(c, x, y), q, type)
                        }
                    }
                }
            }
        }

        buffer.rewind()
        return buffer
    }

    private fun inspectDepthOutput(depth: FloatArray): SanityResult {
        val warnings = mutableListOf<String>()
        if (depth.isEmpty()) {
            return SanityResult(false, "empty depth output", warnings)
        }
        if (!allFinite(depth)) {
            return SanityResult(false, "non-finite depth value detected", warnings)
        }

        val sorted = depth.sorted()
        val min = sorted.first()
        val max = sorted.last()
        val p25 = quantile(sorted, 0.25)
        val median = quantile(sorted, 0.50)
        val p75 = quantile(sorted, 0.75)
        val mean = depth.average()
        val std = sqrt(depth.map { (it - mean) * (it - mean) }.average())

        if (std < 1e-4) warnings += "Depth output is nearly flat (std=${fmt6(std.toFloat())})."

        val summary =
            "finite=${depth.size}/${depth.size} | " +
                "min=${fmt4(min)} | p25=${fmt4(p25)} | median=${fmt4(median)} | " +
                "p75=${fmt4(p75)} | max=${fmt4(max)} | mean=${fmt4(mean.toFloat())} | std=${fmt4(std.toFloat())}"

        return SanityResult(true, summary, warnings)
    }

    private fun depthByteBufferToFloatArray(
        buffer: ByteBuffer,
        type: DataType,
        quant: QuantizationInfo?
    ): FloatArray {
        val duplicate = buffer
            .duplicate()
            .order(ByteOrder.nativeOrder())
            .also { it.rewind() }

        return when (type) {
            DataType.FLOAT32 -> {
                val view = duplicate.asFloatBuffer()
                FloatArray(view.remaining()).also { view.get(it) }
            }
            DataType.INT8 -> {
                val q = quant ?: return FloatArray(0)
                FloatArray(duplicate.remaining()) {
                    (duplicate.get().toInt() - q.zeroPoint) * q.scale
                }
            }
            DataType.UINT8 -> {
                val q = quant ?: return FloatArray(0)
                FloatArray(duplicate.remaining()) {
                    ((duplicate.get().toInt() and 0xFF) - q.zeroPoint) * q.scale
                }
            }
            else -> FloatArray(0)
        }
    }

    // -------------------------------------------------------------------------
    // Standard OSNet x1.0 FP16
    // -------------------------------------------------------------------------

    private fun validateOsnet(): ModelValidationReport.ModelCheck {
        val asset = loadAsset(OSNET_ASSET)
            ?: return missingCheck("OSNet x1.0 FP16", OSNET_ASSET)

        var handle: TfliteAcceleration.InterpreterHandle? = null

        try {
            val loadStart = System.nanoTime()
            val created = TfliteAcceleration.create(
                context = context,
                modelBuffer = asset.buffer,
                numThreads = 2,
                preferQnn = true,
                preferGpu = false,
                qnnHtpPrecision = TfliteAcceleration.QnnHtpPrecision.FP16,
                tag = OSNET_TAG
            )
            handle = created

            val interpreter = created.interpreter
            val backend = created.backend.name
            val loadMs = elapsedMs(loadStart)

            val allocStart = System.nanoTime()
            interpreter.allocateTensors()
            val allocationMs = elapsedMs(allocStart)

            val input = interpreter.getInputTensor(0)
            val output = interpreter.getOutputTensor(0)
            val inputShape = input.shape()
            val outputShape = output.shape()
            val inputNchw = when {
                inputShape.contentEquals(intArrayOf(1, 3, 256, 128)) -> true
                inputShape.contentEquals(intArrayOf(1, 256, 128, 3)) -> false
                else -> null
            }

            val inputOk = input.dataType() == DataType.FLOAT32 && inputNchw != null
            val outputOk =
                output.dataType() == DataType.FLOAT32 &&
                    output.numElements() == 512

            val inputSummary =
                "${shapeString(inputShape)} ${input.dataType()} " +
                    "layout=${if (inputNchw == true) "NCHW" else if (inputNchw == false) "NHWC" else "UNSUPPORTED"}"

            val outputSummary =
                "${shapeString(outputShape)} ${output.dataType()} elements=${output.numElements()}"

            Log.i(
                OSNET_TAG,
                "OSNet tensor contract input=$inputSummary output=$outputSummary backend=$backend"
            )

            if (!inputOk || !outputOk || backend != TfliteAcceleration.Backend.QNN.name) {
                return ModelValidationReport.ModelCheck(
                    name = "OSNet x1.0 FP16",
                    assetName = OSNET_ASSET,
                    assetPresent = true,
                    assetSizeBytes = asset.sizeBytes,
                    loadMs = loadMs,
                    allocationMs = allocationMs,
                    warmupCount = 0,
                    benchmarkCount = 0,
                    meanMs = 0.0,
                    medianMs = 0.0,
                    minMs = 0.0,
                    maxMs = 0.0,
                    inputSummary = inputSummary,
                    outputSummary = outputSummary,
                    sanitySummary = "NOT RUN",
                    contractSummary =
                        "FAIL: expected FLOAT32 [1,3,256,128] -> FLOAT32 [1,512] with QNN/HTP FP16 backend; actual backend=$backend",
                    passed = false,
                    warnings = listOf("OSNet tensor contract/backend mismatch."),
                    backend = backend
                )
            }

            val inputBuffer = deterministicOsnetInputBuffer(input, inputNchw == true)
            val outputBuffer = ByteBuffer
                .allocateDirect(output.numElements() * 4)
                .order(ByteOrder.nativeOrder())

            val warmupCount = 2
            repeat(warmupCount) {
                interpreter.run(
                    inputBuffer.duplicateForRun(),
                    outputBuffer.duplicateForRunForOutput()
                )
            }

            val benchmarkCount = 8
            val times = DoubleArray(benchmarkCount)
            var finalOutput: ByteBuffer? = null

            for (i in 0 until benchmarkCount) {
                val runOutput = outputBuffer.duplicateForRunForOutput()
                val start = System.nanoTime()
                interpreter.run(
                    inputBuffer.duplicateForRun(),
                    runOutput
                )
                times[i] = elapsedMsDouble(start)
                finalOutput = runOutput

                Log.i(
                    OSNET_TAG,
                    "OSNet benchmark[${i + 1}/$benchmarkCount] backend=$backend " +
                        "inference=${fmtMs(times[i])}"
                )
            }

            val embedding = finalOutput?.let(::bufferToFloatArray)
                ?: return failedCheck(
                    OSNET_ASSET,
                    "OSNet x1.0 FP16",
                    IllegalStateException("No OSNet output buffer"),
                    asset.sizeBytes,
                    backend
                )

            val finite = allFinite(embedding)
            val absMax = embedding.maxOfOrNull { abs(it) } ?: 0f
            val mean = if (embedding.isNotEmpty()) embedding.average().toFloat() else 0f
            val rmsValue = rms(embedding)
            val norm = sqrt(embedding.sumOf { it.toDouble() * it.toDouble() }).toFloat()
            val warnings = mutableListOf<String>()

            if (embedding.size != 512) {
                warnings += "OSNet embedding dimension=${embedding.size}, expected=512."
            }
            if (!finite) {
                warnings += "OSNet embedding contains non-finite values."
            }
            if (rmsValue < 1e-6f) {
                warnings += "OSNet embedding is nearly zero/flat."
            }
            if (norm <= 1e-8f) {
                warnings += "OSNet embedding L2 norm is too small."
            }

            val sanityPassed =
                embedding.size == 512 &&
                    finite &&
                    rmsValue >= 1e-6f &&
                    norm > 1e-8f

            val sanitySummary =
                "finite=$finite | dim=${embedding.size} | " +
                    "min=${fmt6(embedding.minOrNull() ?: 0f)} | " +
                    "max=${fmt6(embedding.maxOrNull() ?: 0f)} | " +
                    "mean=${fmt6(mean)} | rms=${fmt6(rmsValue)} | " +
                    "absMax=${fmt6(absMax)} | l2=${fmt6(norm)}"

            val stats = summarizeTimes(times)

            Log.i(
                OSNET_TAG,
                "OSNet validation backend=$backend " +
                    "mean=${fmtMs(stats.mean)} median=${fmtMs(stats.median)} " +
                    "min=${fmtMs(stats.min)} max=${fmtMs(stats.max)} " +
                    "sanity=$sanitySummary"
            )

            return ModelValidationReport.ModelCheck(
                name = "OSNet x1.0 FP16",
                assetName = OSNET_ASSET,
                assetPresent = true,
                assetSizeBytes = asset.sizeBytes,
                loadMs = loadMs,
                allocationMs = allocationMs,
                warmupCount = warmupCount,
                benchmarkCount = benchmarkCount,
                meanMs = stats.mean,
                medianMs = stats.median,
                minMs = stats.min,
                maxMs = stats.max,
                inputSummary = inputSummary,
                outputSummary = outputSummary,
                sanitySummary = sanitySummary,
                contractSummary = if (sanityPassed) {
                    "PASS: Standard OSNet x1.0 FP16 TFLite executed through QNN/HTP and returned a stable 512-D embedding."
                } else {
                    "FAIL: OSNet execution completed but embedding sanity failed."
                },
                passed = sanityPassed,
                warnings = warnings,
                backend = backend
            )
        } catch (t: Throwable) {
            Log.e(TAG, "OSNet validation failed", t)
            return failedCheck(
                OSNET_ASSET,
                "OSNet x1.0 FP16",
                t,
                asset.sizeBytes,
                handle?.backend?.name ?: "QNN"
            )
        } finally {
            try { handle?.close() } catch (_: Throwable) {}
        }
    }

    private fun deterministicOsnetInputBuffer(
        input: Tensor,
        nchw: Boolean
    ): ByteBuffer {
        val shape = input.shape()
        val height = if (nchw) shape[2] else shape[1]
        val width = if (nchw) shape[3] else shape[2]
        val buffer = ByteBuffer
            .allocateDirect(input.numElements() * 4)
            .order(ByteOrder.nativeOrder())

        fun r(x: Int): Float = x / (width - 1f)
        fun g(y: Int): Float = y / (height - 1f)
        fun b(x: Int, y: Int): Float = ((r(x) + g(y)) * 0.5f).coerceIn(0f, 1f)

        // IMPORTANT: standard OSNet deployment model uses RGB/255 only.
        if (nchw) {
            for (y in 0 until height) for (x in 0 until width) buffer.putFloat(r(x))
            for (y in 0 until height) for (x in 0 until width) buffer.putFloat(g(y))
            for (y in 0 until height) for (x in 0 until width) buffer.putFloat(b(x, y))
        } else {
            for (y in 0 until height) {
                for (x in 0 until width) {
                    buffer.putFloat(r(x))
                    buffer.putFloat(g(y))
                    buffer.putFloat(b(x, y))
                }
            }
        }

        buffer.rewind()
        return buffer
    }

    // -------------------------------------------------------------------------
    // Deterministic inputs / generic tensor helpers
    // -------------------------------------------------------------------------

    private fun deterministicYoloInput(shape: IntArray): ByteBuffer {
        val buffer = ByteBuffer
            .allocateDirect(shape.product() * 4)
            .order(ByteOrder.nativeOrder())

        val h = shape[2]
        val w = shape[3]
        for (c in 0 until 3) {
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val xf = x / (w - 1f)
                    val yf = y / (h - 1f)
                    val checker = if (((x / 32) + (y / 32)) % 2 == 0) 0.20f else 0.80f
                    val value = when (c) {
                        0 -> checker * 0.75f + xf * 0.25f
                        1 -> checker * 0.65f + yf * 0.35f
                        else -> checker * 0.55f + (xf + yf) * 0.225f
                    }
                    buffer.putFloat(value.coerceIn(0f, 1f))
                }
            }
        }

        buffer.rewind()
        return buffer
    }

    private fun deterministicXFeatInput(shape: IntArray): ByteBuffer {
        val buffer = ByteBuffer
            .allocateDirect(shape.product() * 4)
            .order(ByteOrder.nativeOrder())

        val h = shape[1]
        val w = shape[2]
        for (y in 0 until h) {
            for (x in 0 until w) {
                val checker = if (((x / 16) + (y / 16)) % 2 == 0) 0.12f else 0.88f
                val gradient = (x / (w - 1f)) * 0.10f + (y / (h - 1f)) * 0.10f
                buffer.putFloat((checker + gradient).coerceIn(0f, 1f))
            }
        }

        buffer.rewind()
        return buffer
    }

    private fun allocateOutputBuffers(interpreter: Interpreter): MutableMap<Int, Any> {
        val outputs = HashMap<Int, Any>()
        for (i in 0 until interpreter.outputTensorCount) {
            val tensor = interpreter.getOutputTensor(i)
            outputs[i] = ByteBuffer
                .allocateDirect(tensor.numElements() * bytesPerElement(tensor.dataType()))
                .order(ByteOrder.nativeOrder())
        }
        return outputs
    }

    private fun bufferToFloatArray(buffer: ByteBuffer): FloatArray {
        val duplicate = buffer
            .duplicate()
            .order(ByteOrder.nativeOrder())
            .also { it.rewind() }
        val view = duplicate.asFloatBuffer()
        return FloatArray(view.remaining()).also { view.get(it) }
    }

    // -------------------------------------------------------------------------
    // Quantization helpers
    // -------------------------------------------------------------------------

    private data class QuantizationInfo(
        val scale: Float,
        val zeroPoint: Int
    )

    private fun readQuantization(tensor: Tensor): QuantizationInfo? = try {
        val p = tensor.quantizationParams()
        if (p.scale > 0f) QuantizationInfo(p.scale, p.zeroPoint) else null
    } catch (_: Throwable) {
        null
    }

    private fun isQuantizedType(type: DataType): Boolean =
        type == DataType.INT8 || type == DataType.UINT8

    private fun bytesPerElement(type: DataType): Int = when (type) {
        DataType.FLOAT32 -> 4
        DataType.INT8, DataType.UINT8 -> 1
        DataType.INT16 -> 2
        DataType.INT32 -> 4
        else -> throw IllegalArgumentException("Unsupported tensor type=$type")
    }

    private fun putQuantized(
        buffer: ByteBuffer,
        value: Float,
        q: QuantizationInfo,
        type: DataType
    ) {
        val quantized = (value / q.scale + q.zeroPoint).roundToInt()
        when (type) {
            DataType.INT8 -> buffer.put(quantized.coerceIn(-128, 127).toByte())
            DataType.UINT8 -> buffer.put(quantized.coerceIn(0, 255).toByte())
            else -> error("Unsupported quantized type=$type")
        }
    }

    // -------------------------------------------------------------------------
    // Asset / report helpers
    // -------------------------------------------------------------------------

    private data class LoadedAsset(
        val buffer: ByteBuffer,
        val sizeBytes: Long
    )

    private fun loadAsset(name: String): LoadedAsset? = try {
        val bytes = context.assets.open(name).use { it.readBytes() }
        LoadedAsset(
            buffer = ByteBuffer
                .allocateDirect(bytes.size)
                .order(ByteOrder.nativeOrder())
                .apply {
                    put(bytes)
                    rewind()
                },
            sizeBytes = bytes.size.toLong()
        )
    } catch (_: Throwable) {
        null
    }

    private fun missingCheck(
        name: String,
        asset: String
    ): ModelValidationReport.ModelCheck =
        ModelValidationReport.ModelCheck(
            name = name,
            assetName = asset,
            assetPresent = false,
            assetSizeBytes = 0L,
            loadMs = 0L,
            allocationMs = 0L,
            warmupCount = 0,
            benchmarkCount = 0,
            meanMs = 0.0,
            medianMs = 0.0,
            minMs = 0.0,
            maxMs = 0.0,
            inputSummary = "-",
            outputSummary = "-",
            sanitySummary = "NOT RUN",
            contractSummary = "FAIL: asset missing",
            passed = false,
            error = "Asset '$asset' not found in app/src/main/assets/"
        )

    private fun failedCheck(
        asset: String,
        name: String,
        throwable: Throwable,
        sizeBytes: Long = 0L,
        backend: String = "CPU"
    ): ModelValidationReport.ModelCheck =
        ModelValidationReport.ModelCheck(
            name = name,
            assetName = asset,
            assetPresent = sizeBytes > 0L,
            assetSizeBytes = sizeBytes,
            loadMs = 0L,
            allocationMs = 0L,
            warmupCount = 0,
            benchmarkCount = 0,
            meanMs = 0.0,
            medianMs = 0.0,
            minMs = 0.0,
            maxMs = 0.0,
            inputSummary = "-",
            outputSummary = "-",
            sanitySummary = "FAILED",
            contractSummary = "FAIL",
            passed = false,
            error = throwable.message ?: throwable.javaClass.simpleName,
            backend = backend
        )

    // -------------------------------------------------------------------------
    // Generic math / timing
    // -------------------------------------------------------------------------

    private data class SanityResult(
        val passed: Boolean,
        val summary: String,
        val warnings: List<String>
    )

    private data class TimeStats(
        val mean: Double,
        val median: Double,
        val min: Double,
        val max: Double
    )

    private fun summarizeTimes(times: DoubleArray): TimeStats {
        if (times.isEmpty()) return TimeStats(0.0, 0.0, 0.0, 0.0)
        val sorted = times.sorted()
        val median = if (sorted.size % 2 == 0) {
            (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) * 0.5
        } else {
            sorted[sorted.size / 2]
        }
        return TimeStats(
            mean = times.average(),
            median = median,
            min = sorted.first(),
            max = sorted.last()
        )
    }

    private fun allFinite(values: FloatArray): Boolean = values.all { it.isFinite() }

    private fun rms(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        return sqrt(values.map { it.toDouble() * it.toDouble() }.average()).toFloat()
    }

    private fun quantile(sorted: List<Float>, q: Double): Float {
        if (sorted.isEmpty()) return 0f
        val idx = ((sorted.size - 1) * q)
            .toInt()
            .coerceIn(0, sorted.lastIndex)
        return sorted[idx]
    }

    private fun shapeString(shape: IntArray): String =
        shape.joinToString(prefix = "[", postfix = "]", separator = ",")

    private fun IntArray.product(): Int =
        fold(1) { acc, value -> acc * value }

    private fun deviceSummary(): String {
        val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE
        return "Device: ${Build.MANUFACTURER} ${Build.MODEL} | " +
            "SoC=$soc | Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}) | " +
            "Heap=${Debug.getNativeHeapAllocatedSize() / (1024 * 1024)} MB"
    }

    private fun elapsedMs(startNs: Long): Long =
        (System.nanoTime() - startNs) / 1_000_000L

    private fun elapsedMsDouble(startNs: Long): Double =
        (System.nanoTime() - startNs) / 1_000_000.0

    private fun ByteBuffer.duplicateForRun(): ByteBuffer =
        duplicate()
            .order(ByteOrder.nativeOrder())
            .also { it.rewind() }

    private fun ByteBuffer.duplicateForRunForOutput(): ByteBuffer =
        duplicate()
            .order(ByteOrder.nativeOrder())
            .also { it.clear() }

    private fun fmtMs(value: Double): String =
        "%.3f ms".format(value)

    private fun fmt4(value: Float): String =
        "%.4f".format(value)

    private fun fmt6(value: Float): String =
        "%.6f".format(value)

    private fun fmt8(value: Float): String =
        "%.8f".format(value)

    companion object {
        const val YOLO_ASSET = "yolo11n-seg.tflite"
        const val XFEAT_ASSET = "xfeat_gray_640.tflite"
        const val DEPTH_ASSET = "depth_anything_v2_small.tflite"
        const val OSNET_ASSET = "osnet/osnet_x1_0_embedding_fp16_s23.tflite"

        private const val TAG = "ModelValidation"
        private const val YOLO_TAG = "ModelValidation-YOLO11n-seg"
        private const val XFEAT_TAG = "ModelValidation-XFeat"
        private const val DEPTH_TAG = "ModelValidation-Depth"
        private const val OSNET_TAG = "ModelValidation-OSNet"
    }
}
