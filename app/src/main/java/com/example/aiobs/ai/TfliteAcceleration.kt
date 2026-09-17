package com.example.aiobs.ai

import android.content.Context
import android.system.Os
import android.util.Log
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

/**
 * TFLite / QNN / GPU acceleration selector.
 *
 * QNN HTP performance mode is applied when a QnnDelegate is created.
 *
 * Important:
 * - MainActivity controls [qnnPerformanceMode].
 * - Normal runtime calls [create()] through this object.
 * - No Depth model needs to be initialized first to "wake up" HTP.
 * - The performance mode is applied directly to QnnDelegate.Options.
 */
object TfliteAcceleration {

    private const val TAG_DEFAULT = "TfliteAcceleration"

    // ------------------------------------------------------------------------
    // QNN HTP performance modes
    // ------------------------------------------------------------------------

    enum class QnnPerformanceMode {
        DEFAULT,
        BURST,
        SUSTAINED_HIGH_PERFORMANCE,
        POWER_SAVER
    }

    /**
     * MainActivity already changes this field through its four-mode UI.
     *
     * Keep DEFAULT as the compatibility/default behavior.
     *
     * For the normal runtime, selecting:
     *
     *   SUSTAINED_HIGH_PERFORMANCE
     *
     * will request the HTP sustained high performance mode.
     *
     * Selecting BURST / POWER_SAVER will request those modes respectively.
     */
    private const val PREF_NAME = "aiobs_tflite_acceleration"
    private const val KEY_QNN_PERFORMANCE_MODE = "qnn_performance_mode"

    @Volatile
    var qnnPerformanceMode: QnnPerformanceMode =
        QnnPerformanceMode.DEFAULT

    /**
     * Restore the persisted QNN HTP performance mode.
     *
     * MainActivity calls this once during startup so the four-mode UI reflects
     * the actual selected runtime policy before the first AI controller starts.
     */
    fun initialize(context: Context) {
        val prefs = context.applicationContext
            .getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

        qnnPerformanceMode = runCatching {
            QnnPerformanceMode.valueOf(
                prefs.getString(
                    KEY_QNN_PERFORMANCE_MODE,
                    QnnPerformanceMode.DEFAULT.name
                ) ?: QnnPerformanceMode.DEFAULT.name
            )
        }.getOrDefault(QnnPerformanceMode.DEFAULT)

        Log.i(
            TAG_DEFAULT,
            "QNN performance mode restored: $qnnPerformanceMode"
        )
    }

    /**
     * Change the QNN HTP performance mode and immediately apply it to the
     * process-wide native HTP performance manager.
     */
    fun setQnnPerformanceMode(
        context: Context,
        mode: QnnPerformanceMode
    ): Boolean {
        qnnPerformanceMode = mode

        context.applicationContext
            .getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_QNN_PERFORMANCE_MODE, mode.name)
            .apply()

        val ok = HtpPerformanceManager.setPerformanceMode(
            context.applicationContext,
            mode
        )

        Log.i(
            TAG_DEFAULT,
            "QNN performance mode changed -> $mode nativeApplied=$ok"
        )

        return ok
    }

    // ------------------------------------------------------------------------
    // Backend
    // ------------------------------------------------------------------------

    enum class Backend {
        QNN,
        GPU,
        CPU
    }

    // ------------------------------------------------------------------------
    // QNN HTP precision
    // ------------------------------------------------------------------------

    enum class QnnHtpPrecision {
        QUANTIZED,
        FP16
    }

    // ------------------------------------------------------------------------
    // Interpreter handle
    // ------------------------------------------------------------------------

    data class InterpreterHandle(
        val interpreter: Interpreter,
        val backend: Backend,
        private val delegate: Any?
    ) : Closeable {

        override fun close() {

            try {
                interpreter.close()
            } catch (_: Throwable) {
            }

            try {
                when (delegate) {
                    is Closeable -> delegate.close()
                    is AutoCloseable -> delegate.close()
                }
            } catch (_: Throwable) {
            }
        }
    }

    // ------------------------------------------------------------------------
    // Create interpreter
    // ------------------------------------------------------------------------

    fun create(
        context: Context,
        modelBuffer: ByteBuffer,
        numThreads: Int = 2,
        preferQnn: Boolean = true,
        preferGpu: Boolean = false,
        qnnHtpPrecision: QnnHtpPrecision = QnnHtpPrecision.QUANTIZED,
        tag: String = TAG_DEFAULT
    ): InterpreterHandle {

        require(modelBuffer.capacity() > 0) {
            "modelBuffer is empty"
        }

        // The global HTP performance vote is independent from any one model.
        // Ensure the current UI-selected mode is applied before constructing
        // a QNN delegate/interpreter.
        if (preferQnn) {
            try {
                HtpPerformanceManager.setPerformanceMode(
                    context.applicationContext,
                    qnnPerformanceMode
                )
            } catch (t: Throwable) {
                Log.w(
                    tag,
                    "Unable to apply HTP performance mode=$qnnPerformanceMode",
                    t
                )
            }
        }

        // --------------------------------------------------------------------
        // Model-specific precision override
        // --------------------------------------------------------------------

        val actualQnnPrecision =
            if (tag.contains(
                    "YOLO11Seg",
                    ignoreCase = true
                )
            ) {

                if (qnnHtpPrecision !=
                    QnnHtpPrecision.QUANTIZED
                ) {

                    Log.w(
                        tag,
                        "YOLO11Seg requested precision=$qnnHtpPrecision; " +
                                "forcing QUANTIZED for INT8 YOLO model"
                    )
                }

                QnnHtpPrecision.QUANTIZED

            } else {

                qnnHtpPrecision
            }

        Log.i(
            tag,
            "QNN precision resolved " +
                    "requested=$qnnHtpPrecision " +
                    "actual=$actualQnnPrecision"
        )

        // --------------------------------------------------------------------
        // QNN / HTP
        // --------------------------------------------------------------------

        if (preferQnn) {

            var qnnDelegate: QnnDelegate? = null

            try {

                val nativeLibDir =
                    context.applicationInfo.nativeLibraryDir

                // ------------------------------------------------------------
                // ADSP library path
                // ------------------------------------------------------------

                try {

                    Os.setenv(
                        "ADSP_LIBRARY_PATH",
                        nativeLibDir,
                        true
                    )

                    Log.i(
                        tag,
                        "ADSP_LIBRARY_PATH=$nativeLibDir"
                    )

                } catch (t: Throwable) {

                    Log.w(
                        tag,
                        "Unable to set ADSP_LIBRARY_PATH",
                        t
                    )
                }

                Log.i(
                    tag,
                    "QNN HTP init " +
                            "nativeLibDir=$nativeLibDir " +
                            "precision=$actualQnnPrecision " +
                            "performanceMode=$qnnPerformanceMode"
                )

                // ------------------------------------------------------------
                // QNN options
                // ------------------------------------------------------------

                val qnnOptions =
                    QnnDelegate.Options().apply {

                        setBackendType(
                            QnnDelegate.Options.BackendType.HTP_BACKEND
                        )

                        // ----------------------------------------------------
                        // HTP precision
                        // ----------------------------------------------------

                        setHtpPrecision(
                            when (actualQnnPrecision) {

                                QnnHtpPrecision.QUANTIZED ->
                                    QnnDelegate.Options.HtpPrecision
                                        .HTP_PRECISION_QUANTIZED

                                QnnHtpPrecision.FP16 ->
                                    QnnDelegate.Options.HtpPrecision
                                        .HTP_PRECISION_FP16
                            }
                        )

                        // ----------------------------------------------------
                        // HTP optimization
                        // ----------------------------------------------------

                        setHtpOptimizationStrategy(
                            QnnDelegate.Options
                                .HtpOptimizationStrategy
                                .HTP_OPTIMIZE_FOR_INFERENCE_O3
                        )

                        // ----------------------------------------------------
                        // HMX
                        // ----------------------------------------------------

                        setHtpUseConvHmx(
                            QnnDelegate.Options
                                .HtpUseConvHmx
                                .HTP_CONV_HMX_ON
                        )

                        Log.i(
                            tag,
                            "QNN HTP optimized execution profile active " +
                                    "mode=$qnnPerformanceMode " +
                                    "precision=$actualQnnPrecision " +
                                    "O3=true " +
                                    "HMX=true"
                        )
                    }

                // ------------------------------------------------------------
                // Create delegate
                // ------------------------------------------------------------

                qnnDelegate =
                    QnnDelegate(qnnOptions)

                Log.i(
                    tag,
                    "QNN HTP delegate created successfully " +
                            "performanceMode=$qnnPerformanceMode"
                )

                // ------------------------------------------------------------
                // Interpreter options
                // ------------------------------------------------------------

                val interpreterOptions =
                    Interpreter.Options().apply {

                        setNumThreads(numThreads)

                        addDelegate(qnnDelegate)
                    }

                // ------------------------------------------------------------
                // Interpreter model buffer
                // ------------------------------------------------------------

                val interpreterBuffer =
                    modelBuffer
                        .duplicate()
                        .order(ByteOrder.nativeOrder())
                        .apply {
                            rewind()
                        }

                val interpreter =
                    Interpreter(
                        interpreterBuffer,
                        interpreterOptions
                    )

                // ------------------------------------------------------------
                // Allocate tensors
                // ------------------------------------------------------------

                val allocateStart =
                    System.nanoTime()

                interpreter.allocateTensors()

                val allocateMs =
                    (System.nanoTime() - allocateStart) /
                            1_000_000.0

                // ------------------------------------------------------------
                // Tensor contract
                // ------------------------------------------------------------

                val inputTensor =
                    interpreter.getInputTensor(0)

                val outputTensor =
                    interpreter.getOutputTensor(0)

                Log.i(
                    tag,
                    "TFLite backend=QNN/HTP " +
                            "precision=$actualQnnPrecision " +
                            "performanceMode=$qnnPerformanceMode " +
                            "O3 HMX " +
                            "allocateMs=${formatMs(allocateMs)}"
                )

                Log.i(
                    tag,
                    "QNN input " +
                            "name=${inputTensor.name()} " +
                            "shape=${inputTensor.shape().contentToString()} " +
                            "type=${inputTensor.dataType()}"
                )

                Log.i(
                    tag,
                    "QNN output " +
                            "name=${outputTensor.name()} " +
                            "shape=${outputTensor.shape().contentToString()} " +
                            "type=${outputTensor.dataType()}"
                )

                Log.i(
                    tag,
                    "QNN HTP CONTRACT " +
                            "backend=HTP " +
                            "precision=$actualQnnPrecision " +
                            "performanceMode=$qnnPerformanceMode " +
                            "inputType=${inputTensor.dataType()} " +
                            "outputType=${outputTensor.dataType()}"
                )

                return InterpreterHandle(
                    interpreter = interpreter,
                    backend = Backend.QNN,
                    delegate = qnnDelegate
                )

            } catch (t: Throwable) {

                try {
                    qnnDelegate?.close()
                } catch (_: Throwable) {
                }

                Log.e(
                    tag,
                    "QNN HTP initialization failed",
                    t
                )
            }
        }

        // --------------------------------------------------------------------
        // GPU
        // --------------------------------------------------------------------

        if (preferGpu) {

            var gpuDelegate: GpuDelegate? = null

            try {

                val compatibility =
                    CompatibilityList()

                if (!compatibility.isDelegateSupportedOnThisDevice) {

                    Log.i(
                        tag,
                        "GPU delegate not supported on this device"
                    )

                } else {

                    gpuDelegate =
                        GpuDelegate()

                    val gpuOptions =
                        Interpreter.Options().apply {

                            setNumThreads(numThreads)

                            addDelegate(gpuDelegate)
                        }

                    val interpreterBuffer =
                        modelBuffer
                            .duplicate()
                            .order(ByteOrder.nativeOrder())
                            .apply {
                                rewind()
                            }

                    val interpreter =
                        Interpreter(
                            interpreterBuffer,
                            gpuOptions
                        )

                    val allocateStart =
                        System.nanoTime()

                    interpreter.allocateTensors()

                    val allocateMs =
                        (System.nanoTime() - allocateStart) /
                                1_000_000.0

                    val inputTensor =
                        interpreter.getInputTensor(0)

                    val outputTensor =
                        interpreter.getOutputTensor(0)

                    Log.i(
                        tag,
                        "TFLite backend=GPU " +
                                "allocateMs=${formatMs(allocateMs)}"
                    )

                    Log.i(
                        tag,
                        "GPU input " +
                                "name=${inputTensor.name()} " +
                                "shape=${inputTensor.shape().contentToString()} " +
                                "type=${inputTensor.dataType()}"
                    )

                    Log.i(
                        tag,
                        "GPU output " +
                                "name=${outputTensor.name()} " +
                                "shape=${outputTensor.shape().contentToString()} " +
                                "type=${outputTensor.dataType()}"
                    )

                    return InterpreterHandle(
                        interpreter = interpreter,
                        backend = Backend.GPU,
                        delegate = gpuDelegate
                    )
                }

            } catch (t: Throwable) {

                try {
                    gpuDelegate?.close()
                } catch (_: Throwable) {
                }

                Log.w(
                    tag,
                    "GPU delegate initialization failed; falling back to CPU",
                    t
                )
            }
        }

        // --------------------------------------------------------------------
        // CPU fallback
        // --------------------------------------------------------------------

        try {

            val cpuOptions =
                Interpreter.Options().apply {
                    setNumThreads(numThreads)
                }

            val interpreterBuffer =
                modelBuffer
                    .duplicate()
                    .order(ByteOrder.nativeOrder())
                    .apply {
                        rewind()
                    }

            val interpreter =
                Interpreter(
                    interpreterBuffer,
                    cpuOptions
                )

            val allocateStart =
                System.nanoTime()

            interpreter.allocateTensors()

            val allocateMs =
                (System.nanoTime() - allocateStart) /
                        1_000_000.0

            val inputTensor =
                interpreter.getInputTensor(0)

            val outputTensor =
                interpreter.getOutputTensor(0)

            Log.w(
                tag,
                "TFLite backend=CPU " +
                        "allocateMs=${formatMs(allocateMs)}"
            )

            Log.i(
                tag,
                "CPU input " +
                        "name=${inputTensor.name()} " +
                        "shape=${inputTensor.shape().contentToString()} " +
                        "type=${inputTensor.dataType()}"
            )

            Log.i(
                tag,
                "CPU output " +
                        "name=${outputTensor.name()} " +
                        "shape=${outputTensor.shape().contentToString()} " +
                        "type=${outputTensor.dataType()}"
            )

            return InterpreterHandle(
                interpreter = interpreter,
                backend = Backend.CPU,
                delegate = null
            )

        } catch (t: Throwable) {

            Log.e(
                tag,
                "CPU interpreter initialization failed",
                t
            )

            throw t
        }
    }

    // ========================================================================
    // Formatting
    // ========================================================================

    private fun formatMs(
        value: Double
    ): String {

        return String.format(
            Locale.US,
            "%.3f",
            value
        )
    }

    // ========================================================================
    // ByteBuffer helper
    // ========================================================================

    private fun ByteBuffer.duplicateForInterpreter():
            ByteBuffer {

        return duplicate()
            .order(ByteOrder.nativeOrder())
            .apply {
                rewind()
            }
    }
}