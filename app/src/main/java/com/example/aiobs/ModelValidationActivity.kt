package com.example.aiobs

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.aiobs.ai.DepthQnnContextBinaryRunner
import com.example.aiobs.ai.SiglipQnnContextBinaryRunner
import com.example.aiobs.ai.XFeatQnnContextBinaryRunner
import com.example.aiobs.validation.ModelValidationController
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

class ModelValidationActivity : AppCompatActivity() {

    private lateinit var controller: ModelValidationController

    private lateinit var tvStatus: TextView

    private lateinit var btnRun: Button
    private lateinit var btnQnnDepth: Button
    private lateinit var btnQnnSiglip: Button
    private lateinit var btnSiglipPrecision: Button

    private lateinit var btnXFeat256Precision: Button
    private lateinit var btnXFeat384: Button
    private lateinit var btnXFeat480: Button
    private lateinit var btnXFeat512: Button
    private lateinit var btnXFeat640: Button
    private lateinit var btnXFeatInt8: Button
    private lateinit var btnClose: Button

    private var depthQnnRunner: DepthQnnContextBinaryRunner? = null
    private var siglipQnnRunner: SiglipQnnContextBinaryRunner? = null

    private var xfeatW8A16Runner: XFeatQnnContextBinaryRunner? = null
    private var xfeatInt8Runner: XFeatQnnContextBinaryRunner? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        controller = ModelValidationController(this)
        buildUi()
    }

    private fun createStyledButton(title: String, onClickAction: () -> Unit): Button {
        return com.google.android.material.button.MaterialButton(this).apply {
            text = title
            setBackgroundColor(Color.parseColor("#2A2A2A"))
            setTextColor(Color.parseColor("#7EC8FF"))
            elevation = 0f
            cornerRadius = 12.dp()
            setOnClickListener { onClickAction() }
        }
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(8, 10, 12))
            setPadding(24, 24, 24, 24)
        }

        val title = TextView(this).apply {
            text = "V11.10 Model Validation"
            setTextColor(Color.WHITE)
            textSize = 22f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            gravity = Gravity.CENTER_VERTICAL
        }

        val subtitle = TextView(this).apply {
            text = "独立验证 YOLO11n-seg + XFeat + Depth Anything V2 + OSNet；另外提供 Depth / SigLIP2 / XFeat Direct QNN Context Binary 验证。"
            setTextColor(Color.LTGRAY)
            textSize = 13f
            setPadding(0, 10, 0, 18)
        }

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        btnRun = createStyledButton("重新验证") { runValidation() }
        btnQnnDepth = createStyledButton("Depth Direct QNN BIN 测试") { runDirectQnnDepthValidation() }
        btnQnnSiglip = createStyledButton("SigLIP2 Direct QNN BIN 测试") { runDirectQnnSiglipValidation() }
        btnSiglipPrecision = createStyledButton("SigLIP2 真实图片精度测试") { runSiglipRealImagePrecision() }
        btnXFeat256Precision = createStyledButton("XFeat-256 QNN 精度测试（Legacy）") { runXFeat256PrecisionTest() }
        btnXFeat384 = createStyledButton("XFeat-384 W8A16 Direct QNN BIN 测试") { runDirectQnnXFeatValidation(XFeatVariant.W8A16_384) }
        btnXFeat480 = createStyledButton("XFeat-480 W8A16 Direct QNN BIN 测试") { runDirectQnnXFeatValidation(XFeatVariant.W8A16_480) }
        btnXFeat512 = createStyledButton("XFeat-512 W8A16 Direct QNN BIN 测试") { runDirectQnnXFeatValidation(XFeatVariant.W8A16_512) }
        btnXFeat640 = createStyledButton("XFeat-640 W8A16 Direct QNN BIN 测试") { runDirectQnnXFeatValidation(XFeatVariant.W8A16_640) }
        btnXFeatInt8 = createStyledButton("XFeat-256 INT8 Direct QNN BIN 测试（Legacy）") { runDirectQnnXFeatValidation(XFeatVariant.INT8) }
        btnClose = createStyledButton("关闭") { finish() }

        addControl(controls, btnRun)
        addControl(controls, btnQnnDepth)
        addControl(controls, btnQnnSiglip)
        addControl(controls, btnSiglipPrecision)
        addControl(controls, btnXFeat256Precision)
        addControl(controls, btnXFeat384)
        addControl(controls, btnXFeat480)
        addControl(controls, btnXFeat512)
        addControl(controls, btnXFeat640)
        addControl(controls, btnXFeatInt8)
        addControl(controls, btnClose)

        tvStatus = TextView(this).apply {
            text = """
           准备就绪。

           ==================================================
           普通模型验证
           ==================================================

           模型：
           - yolo11n-seg.tflite
           - xfeat_gray_640.tflite
           - depth_anything_v2_small.tflite
           - osnet/osnet_x1_0_embedding_fp16_s23.tflite

           YOLO11n-seg：
           LiteRT / TFLite + QNN/HTP

           XFeat：
           LiteRT / TFLite

           Depth Anything V2：
           TFLite + Qualcomm QNN HTP

           OSNet x1.0 FP16：
           FP16-weight TFLite + Qualcomm QNN HTP

           ==================================================
           Direct QNN Context Binary
           ==================================================

           Depth Anything V2：

           Asset:
           depth_anything_v2_s23_w8a16.bin

           Target:
           Samsung Galaxy S23
           Snapdragon 8 Gen 2 / SM8550
           Hexagon V73

           Expected input:
           float32 [1,3,518,518]

           Expected output:
           float32 [518,518]

           --------------------------------------------------

           SigLIP2 ViT-Base 256：

           Asset:
           siglip2_vit_base_256_fp16.serialized.SM8550.bin

           Target:
           Samsung Galaxy S23
           Snapdragon 8 Gen 2 / SM8550
           Hexagon V73

           当前阶段：
           Context Binary
           ↓
           QNN HTP
           ↓
           metadata
           ↓
           graph retrieve
           ↓
           benchmark

           --------------------------------------------------

           XFeat-256：

           Logical input:
           float32 [1,1,256,256]

           Logical outputs:
           feats        [1,64,32,32]
           keypoints    [1,65,32,32]
           reliability  [1,1,32,32]

           W8A16 Asset:
           ${XFeatQnnContextBinaryRunner.W8A16_ASSET}

           INT8 Asset:
           ${XFeatQnnContextBinaryRunner.INT8_ASSET}

           Target:
           Samsung Galaxy S23
           Snapdragon 8 Gen 2 / SM8550
           Hexagon V73

           当前阶段：
           Context Binary
           ↓
           QNN HTP
           ↓
           metadata
           ↓
           graph retrieve
           ↓
           benchmark

           注意：
           XFeat Direct QNN 当前用于独立验证。
           正式生产链默认使用 W8A16。
           """.trimIndent()
            setTextColor(Color.WHITE)
            textSize = 13f
            setPadding(14, 16, 14, 16)
        }

        // 🌟 核心改进：统一用 ScrollView 包裹整个滚动视图，解决挤压与看不到报告的问题
        val scrollableContent = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(controls)
            addView(tvStatus, LinearLayout.LayoutParams(-1, -2).apply {
                topMargin = 24.dp()
            })
        }

        val scroll = ScrollView(this).apply {
            addView(scrollableContent)
        }

        root.addView(title)
        root.addView(subtitle)
        root.addView(
            scroll,
            LinearLayout.LayoutParams(
                -1,
                0,
                1f
            ).apply {
                topMargin = 12.dp()
            }
        )

        setContentView(root)
    }

    private fun addControl(parent: LinearLayout, button: Button) {
        parent.addView(
            button,
            LinearLayout.LayoutParams(
                -1,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = if (parent.childCount == 0) 0 else 12.dp()
            }
        )
    }

    private fun runValidation() {
        if (controller.isRunning()) return
        disableAllControls()
        btnRun.text = "验证中..."
        tvStatus.text = """
       正在执行普通模型验证：

       1. YOLO11n-seg
          LiteRT / TFLite / QNN HTP

       2. XFeat
          LiteRT / TFLite

       3. Depth Anything V2
          TFLite / Qualcomm QNN HTP

       4. OSNet x1.0 FP16
          FP16-weight TFLite / Qualcomm QNN HTP

       --------------------------------------------------

       Direct QNN Context Binary：

       Depth
       当前未启动

       SigLIP2
       当前未启动

       XFeat-256 W8A16
       当前未启动

       XFeat-256 INT8
       当前未启动

       验证运行在独立线程。
       不会启动主相机跟踪，
       不会修改 Recovery，
       不会修改正式运行模型选择。
       """.trimIndent()

        controller.validateAsync { report ->
            runOnUiThread {
                enableAllControls()
                btnRun.text = "重新验证"
                tvStatus.text = report.formatForUi()
                Toast.makeText(
                    this,
                    if (report.overallPassed) "V11.10 模型验证 PASS" else "V11.10 模型验证存在问题",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun runDirectQnnDepthValidation() {
        if (controller.isRunning()) {
            Toast.makeText(this, "普通模型验证正在运行，请先完成当前验证", Toast.LENGTH_SHORT).show()
            return
        }
        if (siglipQnnRunner != null || xfeatW8A16Runner != null || xfeatInt8Runner != null) {
            Toast.makeText(this, "其他 Direct QNN runner 正在使用中，请先关闭当前 runner", Toast.LENGTH_SHORT).show()
            return
        }
        if (depthQnnRunner != null) {
            runQnnBenchmark()
            return
        }
        disableAllControls()
        tvStatus.text = """
       Direct QNN Context Binary

       正在初始化 Depth Anything V2...

       Asset:
       depth_anything_v2_s23_w8a16.bin

       Target:
       Samsung Galaxy S23
       Snapdragon 8 Gen 2 / SM8550
       Hexagon V73

       Expected input:
       float32 [1,3,518,518]

       Expected output:
       float32 [518,518]

       当前阶段：

       Asset
       ↓
       QNN libraries
       ↓
       Context Binary
       ↓
       Graph
       """.trimIndent()

        Thread {
            try {
                val runner = DepthQnnContextBinaryRunner(this@ModelValidationActivity)
                if (!runner.isReady()) {
                    runner.close()
                    runOnUiThread {
                        enableAllControls()
                        tvStatus.text = """
                       Direct QNN Context Binary

                       ❌ Depth 初始化失败

                       nativeCreate() 返回无效 handle。

                       请检查：

                       adb logcat | findstr /I "DepthQnnCtxBin"
                       """.trimIndent()
                        Toast.makeText(this, "Depth Direct QNN BIN 初始化失败", Toast.LENGTH_LONG).show()
                    }
                    return@Thread
                }
                depthQnnRunner = runner
                val description = try { runner.describe() } catch (t: Throwable) { "describe() failed: ${t.message}" }
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = """
                   Depth Direct QNN Context Binary

                   ✅ Context Binary 初始化成功

                   $description

                   --------------------------------------------------

                   下一步：

                   warmup = 5
                   iterations = 20

                   计时范围：

                   native QnnGraph_execute()

                   不包含：

                   Camera
                   Bitmap
                   Kotlin UI
                   Java bookkeeping

                   点击：
                   Depth Direct QNN BIN 测试
                   """.trimIndent()
                    Toast.makeText(this, "Depth Direct QNN BIN 初始化成功", Toast.LENGTH_SHORT).show()
                    runQnnBenchmark()
                }
            } catch (t: Throwable) {
                Log.e(TAG_DEPTH_QNN, "Direct QNN initialization failed", t)
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = """
                   Direct QNN Context Binary

                   ❌ Depth 初始化异常

                   ${t.javaClass.name}

                   ${t.message ?: "<no message>"}
                   """.trimIndent()
                    Toast.makeText(this, "Depth Direct QNN BIN 初始化异常", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun runQnnBenchmark() {
        val runner = depthQnnRunner ?: return
        disableAllControls()
        tvStatus.text = """
       Depth Direct QNN Context Binary

       Runner READY.

       正在 benchmark...

       warmups    = 5
       iterations = 20
       """.trimIndent()

        Thread {
            try {
                val result = runner.benchmark(warmups = 5, iterations = 20)
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = """
                   Depth Direct QNN Context Binary

                   ✅ Benchmark 完成

                   $result
                   """.trimIndent()
                    Toast.makeText(this, "Depth Direct QNN BIN benchmark 完成", Toast.LENGTH_SHORT).show()
                }
            } catch (t: Throwable) {
                Log.e(TAG_DEPTH_QNN, "Direct QNN benchmark failed", t)
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = "❌ Benchmark 失败: ${t.message}"
                }
            }
        }.start()
    }

    private fun writeFloatArray(file: File, values: FloatArray) {
        val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { buffer.putFloat(it) }
        file.outputStream().use { output -> output.write(buffer.array()) }
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) (sorted[middle - 1] + sorted[middle]) / 2.0 else sorted[middle]
    }

    private fun runXFeat256PrecisionTest() {
        if (controller.isRunning()) {
            Toast.makeText(this, "普通模型验证正在运行，请先完成当前验证", Toast.LENGTH_SHORT).show()
            return
        }
        if (depthQnnRunner != null || siglipQnnRunner != null) {
            Toast.makeText(this, "请先关闭当前 Direct QNN runner，再执行 XFeat-256 精度测试", Toast.LENGTH_SHORT).show()
            return
        }
        if (xfeatW8A16Runner != null || xfeatInt8Runner != null) {
            Toast.makeText(this, "XFeat-256 精度 runner 已经初始化，正在执行测试", Toast.LENGTH_SHORT).show()
            return
        }

        val assetDir = "xfeat256_precision"
        val rawFiles = try {
            assets.list(assetDir)?.filter { it.endsWith(".raw", ignoreCase = true) }?.sorted() ?: emptyList()
        } catch (t: Throwable) {
            emptyList()
        }

        if (rawFiles.size != 32) {
            tvStatus.text = "XFeat-256 精度测试错误：期望 32 个 .raw 文件，实际发现 ${rawFiles.size} 个于 assets/$assetDir/"
            Toast.makeText(this, "XFeat-256 精度测试需要 32 个 raw", Toast.LENGTH_LONG).show()
            return
        }

        disableAllControls()
        tvStatus.text = "XFeat-256 QNN 精度测试：正在初始化 W8A16 + INT8..."

        Thread {
            var w8a16: XFeatQnnContextBinaryRunner? = null
            var int8: XFeatQnnContextBinaryRunner? = null
            try {
                w8a16 = XFeatQnnContextBinaryRunner(this@ModelValidationActivity, XFeatQnnContextBinaryRunner.W8A16_ASSET)
                check(w8a16.isReady()) { "W8A16 runner is not ready" }
                int8 = XFeatQnnContextBinaryRunner(this@ModelValidationActivity, XFeatQnnContextBinaryRunner.INT8_ASSET)
                check(int8.isReady()) { "INT8 runner is not ready" }

                xfeatW8A16Runner = w8a16
                xfeatInt8Runner = int8

                val outputRoot = File(getExternalFilesDir(null), "xfeat256_precision_results")
                if (outputRoot.exists()) outputRoot.deleteRecursively()
                check(outputRoot.mkdirs()) { "Failed to create output directory" }

                val timingsFile = File(outputRoot, "timings.csv")
                timingsFile.bufferedWriter().use { writer ->
                    writer.write("sample,w8a16_ms,int8_ms\n")
                    rawFiles.forEachIndexed { index, name ->
                        val inputBytes = assets.open("$assetDir/$name").use { it.readBytes() }
                        val input = FloatArray(XFeatQnnContextBinaryRunner.INPUT_ELEMENTS)
                        ByteBuffer.wrap(inputBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(input)

                        val wStart = System.nanoTime()
                        val wResult = w8a16.infer(input)
                        val wMs = (System.nanoTime() - wStart) / 1_000_000.0

                        val iStart = System.nanoTime()
                        val iResult = int8.infer(input)
                        val iMs = (System.nanoTime() - iStart) / 1_000_000.0

                        val stem = name.removeSuffix(".raw")
                        writeFloatArray(File(outputRoot, "${stem}_w8a16_feats.f32"), wResult.feats)
                        writeFloatArray(File(outputRoot, "${stem}_w8a16_keypoints.f32"), wResult.keypointLogits)
                        writeFloatArray(File(outputRoot, "${stem}_w8a16_heatmap.f32"), wResult.reliability)
                        writeFloatArray(File(outputRoot, "${stem}_int8_feats.f32"), iResult.feats)
                        writeFloatArray(File(outputRoot, "${stem}_int8_keypoints.f32"), iResult.keypointLogits)
                        writeFloatArray(File(outputRoot, "${stem}_int8_heatmap.f32"), iResult.reliability)

                        writer.write("$name,${"%.6f".format(Locale.US, wMs)},${"%.6f".format(Locale.US, iMs)}\n")
                        writer.flush()
                    }
                }

                val timingLines = timingsFile.readLines().drop(1).filter { it.isNotBlank() }
                val wTimes = timingLines.map { it.split(",")[1].toDouble() }
                val iTimes = timingLines.map { it.split(",")[2].toDouble() }

                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = """
                   XFeat-256 Direct QNN 精度测试

                   ✅ QNN 推理全部完成 (32 样本)

                   W8A16 Mean: ${"%.3f".format(Locale.US, wTimes.average())} ms
                   INT8 Mean:  ${"%.3f".format(Locale.US, iTimes.average())} ms

                   输出目录：
                   ${outputRoot.absolutePath}
                   """.trimIndent()
                    Toast.makeText(this, "XFeat-256 QNN 32 样本推理完成", Toast.LENGTH_LONG).show()
                }
            } catch (t: Throwable) {
                Log.e(TAG_XFEAT256_PRECISION, "Test failed", t)
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = "XFeat-256 精度测试失败: ${t.message}"
                }
            } finally {
                try { w8a16?.close() } catch (_: Throwable) {}
                try { int8?.close() } catch (_: Throwable) {}
                xfeatW8A16Runner = null
                xfeatInt8Runner = null
            }
        }.start()
    }

    private fun runDirectQnnSiglipValidation() {
        if (controller.isRunning()) {
            Toast.makeText(this, "普通模型验证正在运行，请先完成当前验证", Toast.LENGTH_SHORT).show()
            return
        }
        if (depthQnnRunner != null || xfeatW8A16Runner != null || xfeatInt8Runner != null) {
            Toast.makeText(this, "其他 Direct QNN runner 正在使用中", Toast.LENGTH_SHORT).show()
            return
        }
        if (siglipQnnRunner != null) {
            runSiglipQnnBenchmark()
            return
        }

        disableAllControls()
        tvStatus.text = "正在初始化 SigLIP2 Direct QNN Context Binary..."
        Thread {
            try {
                val runner = SiglipQnnContextBinaryRunner(this@ModelValidationActivity)
                if (!runner.isReady()) {
                    runner.close()
                    runOnUiThread { enableAllControls(); tvStatus.text = "SigLIP2 初始化失败" }
                    return@Thread
                }
                siglipQnnRunner = runner
                val description = try { runner.describe() } catch (t: Throwable) { "describe() failed: ${t.message}" }
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = "SigLIP2 初始化成功\n$description"
                    runSiglipQnnBenchmark()
                }
            } catch (t: Throwable) {
                runOnUiThread { enableAllControls(); tvStatus.text = "SigLIP2 异常: ${t.message}" }
            }
        }.start()
    }

    private fun runSiglipQnnBenchmark() {
        val runner = siglipQnnRunner ?: return
        disableAllControls()
        tvStatus.text = "正在执行 SigLIP2 QNN Benchmark..."
        Thread {
            try {
                val result = runner.benchmark(5, 20)
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = "SigLIP2 Benchmark 完成\n$result"
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = "SigLIP2 Benchmark 失败: ${t.message}"
                }
            }
        }.start()
    }

    private enum class XFeatVariant(val displayName: String, val assetName: String, val modelSize: Int) {
        W8A16_384("XFeat-384 W8A16", XFeatQnnContextBinaryRunner.W8A16_ASSET_384, 384),
        W8A16_480("XFeat-480 W8A16", XFeatQnnContextBinaryRunner.W8A16_ASSET_480, 480),
        W8A16_512("XFeat-512 W8A16", XFeatQnnContextBinaryRunner.W8A16_ASSET_512, 512),
        W8A16_640("XFeat-640 W8A16", XFeatQnnContextBinaryRunner.W8A16_ASSET_640, 640),
        INT8("XFeat-256 INT8 compatibility", XFeatQnnContextBinaryRunner.INT8_ASSET, 256);
        val gridSize: Int get() = modelSize / 8
    }

    private fun runDirectQnnXFeatValidation(variant: XFeatVariant) {
        if (controller.isRunning()) {
            Toast.makeText(this, "普通模型验证正在运行", Toast.LENGTH_SHORT).show()
            return
        }
        if (depthQnnRunner != null || siglipQnnRunner != null) {
            Toast.makeText(this, "其他 Direct QNN runner 正在使用中", Toast.LENGTH_SHORT).show()
            return
        }

        val existingRunner = when (variant) {
            XFeatVariant.W8A16_384, XFeatVariant.W8A16_480, XFeatVariant.W8A16_512, XFeatVariant.W8A16_640 -> xfeatW8A16Runner
            XFeatVariant.INT8 -> xfeatInt8Runner
        }

        if (existingRunner != null) {
            runXFeatQnnBenchmark(variant)
            return
        }

        disableAllControls()
        tvStatus.text = "正在初始化 ${variant.displayName}..."
        Thread {
            var runner: XFeatQnnContextBinaryRunner? = null
            try {
                runner = XFeatQnnContextBinaryRunner(this@ModelValidationActivity, variant.assetName)
                if (!runner.isReady()) {
                    runner.close()
                    runOnUiThread { enableAllControls(); tvStatus.text = "${variant.displayName} 初始化失败" }
                    return@Thread
                }
                when (variant) {
                    XFeatVariant.W8A16_384, XFeatVariant.W8A16_480, XFeatVariant.W8A16_512, XFeatVariant.W8A16_640 -> xfeatW8A16Runner = runner
                    XFeatVariant.INT8 -> xfeatInt8Runner = runner
                }
                val desc = try { runner.describe() } catch (t: Throwable) { "describe failed: ${t.message}" }
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = "${variant.displayName} 初始化成功\n$desc"
                    runXFeatQnnBenchmark(variant)
                }
            } catch (t: Throwable) {
                try { runner?.close() } catch (_: Throwable) {}
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = "${variant.displayName} 初始化异常: ${t.message}"
                }
            }
        }.start()
    }

    private fun runXFeatQnnBenchmark(variant: XFeatVariant) {
        val runner = when (variant) {
            XFeatVariant.W8A16_384, XFeatVariant.W8A16_480, XFeatVariant.W8A16_512, XFeatVariant.W8A16_640 -> xfeatW8A16Runner
            XFeatVariant.INT8 -> xfeatInt8Runner
        } ?: return

        disableAllControls()
        tvStatus.text = "正在执行 ${variant.displayName} Benchmark..."
        Thread {
            try {
                val result = runner.benchmark(5, 20)
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = "${variant.displayName} Benchmark 完成\n$result"
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = "${variant.displayName} Benchmark 失败: ${t.message}"
                }
            }
        }.start()
    }

    private fun runSiglipRealImagePrecision() {
        if (controller.isRunning()) { Toast.makeText(this, "验证运行中", Toast.LENGTH_SHORT).show(); return }
        if (depthQnnRunner != null || siglipQnnRunner != null || xfeatW8A16Runner != null || xfeatInt8Runner != null) {
            Toast.makeText(this, "请先关闭当前 Direct QNN runner", Toast.LENGTH_SHORT).show()
            return
        }

        val files = try {
            assets.list(REAL_IMAGE_ASSET_DIR)?.filter {
                val lower = it.lowercase()
                lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png")
            }?.sorted() ?: emptyList()
        } catch (t: Throwable) {
            emptyList()
        }

        if (files.isEmpty()) {
            tvStatus.text = "SigLIP2 真实图片精度测试：未在 assets/$REAL_IMAGE_ASSET_DIR/ 中找到测试图片。"
            Toast.makeText(this, "未找到测试图片", Toast.LENGTH_LONG).show()
            return
        }

        disableAllControls()
        tvStatus.text = "SigLIP2 真实图片精度测试：发现 ${files.size} 张图片，正在初始化推理..."

        Thread {
            var fp16: SiglipQnnContextBinaryRunner? = null
            var int8: SiglipQnnContextBinaryRunner? = null
            var w8a16: SiglipQnnContextBinaryRunner? = null
            try {
                fp16 = SiglipQnnContextBinaryRunner(this@ModelValidationActivity, SiglipQnnContextBinaryRunner.FP16_ASSET)
                int8 = SiglipQnnContextBinaryRunner(this@ModelValidationActivity, SiglipQnnContextBinaryRunner.INT8_ASSET)
                w8a16 = SiglipQnnContextBinaryRunner(this@ModelValidationActivity, SiglipQnnContextBinaryRunner.W8A16_ASSET)

                check(fp16.isReady()) { "FP16 init failed" }
                check(int8.isReady()) { "INT8 init failed" }
                check(w8a16.isReady()) { "W8A16 init failed" }

                val csv = StringBuilder("image,fp16_ok,int8_ok,w8a16_ok,int8_cosine,int8_mae,int8_rmse,int8_relative_l2,w8a16_cosine,w8a16_mae,w8a16_rmse,w8a16_relative_l2\n")
                val int8Stats = ArrayList<RealPrecisionRow>(files.size)
                val w8a16Stats = ArrayList<RealPrecisionRow>(files.size)

                for (fileName in files) {
                    val input = loadSiglipImageAsNormalizedRgb(fileName)
                    val reference = fp16.infer(input)
                    val int8Out = int8.infer(input)
                    val w8Out = w8a16.infer(input)

                    val int8Met = compareRealEmbeddings(reference, int8Out)
                    val w8Met = compareRealEmbeddings(reference, w8Out)

                    int8Stats += RealPrecisionRow(fileName, int8Met)
                    w8a16Stats += RealPrecisionRow(fileName, w8Met)

                    csv.append(fileName.csvEscape()).append(",true,true,true,").append(metricCsv(int8Met)).append(',').append(metricCsv(w8Met)).append('\n')
                }

                val resultFile = writeRealPrecisionCsv(csv.toString())
                val report = buildRealPrecisionReport(files.size, int8Stats, w8a16Stats, resultFile.absolutePath)

                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = report
                    Toast.makeText(this, "真实图片精度测试完成", Toast.LENGTH_LONG).show()
                }
            } catch (t: Throwable) {
                Log.e(TAG_REAL_PRECISION, "Test failed", t)
                runOnUiThread {
                    enableAllControls()
                    tvStatus.text = "真实图片精度测试失败: ${t.message}"
                }
            } finally {
                try { fp16?.close() } catch (_: Throwable) {}
                try { int8?.close() } catch (_: Throwable) {}
                try { w8a16?.close() } catch (_: Throwable) {}
            }
        }.start()
    }

    private fun loadSiglipImageAsNormalizedRgb(fileName: String): FloatArray {
        val path = "$REAL_IMAGE_ASSET_DIR/$fileName"
        val source = assets.open(path).use { BitmapFactory.decodeStream(it) } ?: error("无法解码图片：$path")
        val scaled = Bitmap.createScaledBitmap(source, 256, 256, true)
        if (scaled !== source) source.recycle()

        val pixels = IntArray(256 * 256)
        scaled.getPixels(pixels, 0, 256, 0, 0, 256, 256)
        scaled.recycle()

        val result = FloatArray(256 * 256 * 3)
        var j = 0
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            result[j++] = (r - 0.5f) / 0.5f
            result[j++] = (g - 0.5f) / 0.5f
            result[j++] = (b - 0.5f) / 0.5f
        }
        return result
    }

    private fun compareRealEmbeddings(reference: FloatArray, candidate: FloatArray): RealPrecisionMetric {
        require(reference.size == candidate.size && reference.isNotEmpty()) { "Size mismatch" }
        var dot = 0.0; var refNorm2 = 0.0; var candNorm2 = 0.0; var absSum = 0.0; var squareSum = 0.0
        for (i in reference.indices) {
            val r = reference[i].toDouble()
            val c = candidate[i].toDouble()
            val d = c - r
            dot += r * c; refNorm2 += r * r; candNorm2 += c * c; absSum += kotlin.math.abs(d); squareSum += d * d
        }
        val denom = kotlin.math.sqrt(refNorm2 * candNorm2)
        val refNorm = kotlin.math.sqrt(refNorm2)
        val count = reference.size.toDouble()
        return RealPrecisionMetric(
            cosine = if (denom > 0.0) dot / denom else 0.0,
            mae = absSum / count,
            rmse = kotlin.math.sqrt(squareSum / count),
            relativeL2 = if (refNorm > 0.0) kotlin.math.sqrt(squareSum) / refNorm else 0.0
        )
    }

    private fun buildRealPrecisionReport(imageCount: Int, int8Rows: List<RealPrecisionRow>, w8a16Rows: List<RealPrecisionRow>, resultPath: String): String {
        fun summary(label: String, rows: List<RealPrecisionRow>): String {
            if (rows.isEmpty()) return "$label\nno results"
            val cosines = rows.map { it.metric.cosine }.sorted()
            val meanCosine = rows.map { it.metric.cosine }.average()
            val meanMae = rows.map { it.metric.mae }.average()
            val meanRmse = rows.map { it.metric.rmse }.average()
            val meanRelL2 = rows.map { it.metric.relativeL2 }.average()
            val p05 = cosines[((cosines.size - 1) * 0.05).toInt()]
            val median = cosines[cosines.size / 2]
            val min = cosines.first()
            val max = cosines.last()

            return """
       $label
       meanCosine       = ${"%.8f".format(Locale.US, meanCosine)}
       p05Cosine        = ${"%.8f".format(Locale.US, p05)}
       medianCosine     = ${"%.8f".format(Locale.US, median)}
       minCosine        = ${"%.8f".format(Locale.US, min)}
       maxCosine        = ${"%.8f".format(Locale.US, max)}
       meanMAE          = ${"%.8f".format(Locale.US, meanMae)}
       meanRMSE         = ${"%.8f".format(Locale.US, meanRmse)}
       meanRelativeL2   = ${"%.8f".format(Locale.US, meanRelL2)}
       """.trimIndent()
        }

        return """
   SigLIP2 真实图片精度测试

   ✅ 测试完成
   imageCount = $imageCount
   reference   = FP16
   candidates  = INT8 + W8A16

   --------------------------------------------------
   ${summary("INT8", int8Rows)}
   --------------------------------------------------
   ${summary("W8A16", w8a16Rows)}
   --------------------------------------------------
   结果 CSV：
   $resultPath
   """.trimIndent()
    }

    private fun writeRealPrecisionCsv(content: String): File {
        val dir = getExternalFilesDir(null) ?: filesDir
        if (!dir.exists()) dir.mkdirs()
        val file = File(dir, REAL_PRECISION_CSV)
        file.writeText(content, Charsets.UTF_8)
        return file
    }

    private fun metricCsv(metric: RealPrecisionMetric): String {
        return "%.8f,%.8f,%.8f,%.8f".format(Locale.US, metric.cosine, metric.mae, metric.rmse, metric.relativeL2)
    }

    private fun String.csvEscape(): String {
        return if (contains(',') || contains('"') || contains('\n') || contains('\r')) {
            "\"${replace("\"", "\"\"")}\""
        } else {
            this
        }
    }

    private data class RealPrecisionMetric(val cosine: Double, val mae: Double, val rmse: Double, val relativeL2: Double)
    private data class RealPrecisionRow(val imageName: String, val metric: RealPrecisionMetric)

    private fun disableAllControls() {
        btnRun.isEnabled = false
        btnQnnDepth.isEnabled = false
        btnQnnSiglip.isEnabled = false
        btnSiglipPrecision.isEnabled = false
        btnXFeat256Precision.isEnabled = false
        btnXFeat384.isEnabled = false
        btnXFeat480.isEnabled = false
        btnXFeat512.isEnabled = false
        btnXFeat640.isEnabled = false
        btnXFeatInt8.isEnabled = false
        btnClose.isEnabled = false
    }

    private fun enableAllControls() {
        btnRun.isEnabled = true
        btnQnnDepth.isEnabled = true
        btnQnnSiglip.isEnabled = true
        btnSiglipPrecision.isEnabled = true
        btnXFeat256Precision.isEnabled = true
        btnXFeat384.isEnabled = true
        btnXFeat480.isEnabled = true
        btnXFeat512.isEnabled = true
        btnXFeat640.isEnabled = true
        btnXFeatInt8.isEnabled = true
        btnClose.isEnabled = true
        btnRun.text = "重新验证"
    }

    override fun onDestroy() {
        try { depthQnnRunner?.close() } catch (_: Throwable) {}
        depthQnnRunner = null
        try { siglipQnnRunner?.close() } catch (_: Throwable) {}
        siglipQnnRunner = null
        try { xfeatW8A16Runner?.close() } catch (_: Throwable) {}
        xfeatW8A16Runner = null
        try { xfeatInt8Runner?.close() } catch (_: Throwable) {}
        xfeatInt8Runner = null
        controller.shutdown()
        super.onDestroy()
    }

    private fun Int.dp(): Int = (this * resources.displayMetrics.density).toInt()

    companion object {
        private const val TAG_DEPTH_QNN = "DepthQnnCtxBin"
        private const val TAG_SIGLIP_QNN = "SiglipQnnCtxBin"
        private const val TAG_XFEAT_QNN = "XFeatQnnCtxBin"
        private const val TAG_XFEAT256_PRECISION = "XFeat256Precision"
        private const val TAG_REAL_PRECISION = "SiglipRealPrecision"
        private const val REAL_IMAGE_ASSET_DIR = "siglip_precision"
        private const val REAL_PRECISION_CSV = "siglip_real_precision.csv"
    }
}