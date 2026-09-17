package com.example.aiobs.core

import android.content.Context
import android.content.res.AssetManager
import java.util.Locale

/**
 * Discovers AI model assets at runtime.
 *
 * No hard-coded list is required for newly added files: placing another model
 * under app/src/main/assets (including a subdirectory) makes it visible after
 * the next app build/install.
 *
 * This catalog intentionally separates:
 *   1) discovery (what files exist),
 *   2) classification (family / format / quantization), and
 *   3) current adapter support.
 *
 * That prevents the UI from claiming a model is executable merely because its
 * filename looks familiar. Unsupported graph contracts are shown as
 * "需适配" and cannot be selected as the live model.
 */
object ModelAssetCatalog {

    enum class Family {
        YOLO_SEG,
        OSTRACK,
        XFEAT,
        DINOV2, // 🌟 新增 DINOv2 家族
        OTHER
    }

    enum class Format {
        TFLITE,
        QNN_CONTEXT_BIN,
        OTHER
    }

    enum class Quantization {
        FP32,
        FP16,
        W8A16,
        W8A8,
        INT8,
        UINT8,
        UNKNOWN
    }

    data class ModelAsset(
        val path: String,
        val name: String,
        val family: Family,
        val format: Format,
        val quantization: Quantization,
        val supportedByCurrentAdapter: Boolean,
        val supportNote: String
    )

    fun discover(context: Context): List<ModelAsset> {
        val result = ArrayList<ModelAsset>()
        val assets = context.applicationContext.assets
        scan(assets, "", result)
        return result
            .filter { it.family != Family.OTHER }
            .sortedWith(compareBy<ModelAsset>({ it.family.ordinal }, { it.name.lowercase(Locale.US) }))
    }

    fun forFamily(context: Context, family: Family): List<ModelAsset> =
        discover(context).filter { it.family == family }

    fun resolveSelected(
        context: Context,
        family: Family,
        requested: String,
        fallback: String
    ): String {
        val models = forFamily(context, family)
        if (models.any { it.path == requested && it.supportedByCurrentAdapter }) return requested
        if (models.any { it.path == fallback && it.supportedByCurrentAdapter }) return fallback
        return models.firstOrNull { it.supportedByCurrentAdapter }?.path ?: fallback
    }

    fun quantizationLabel(value: Quantization): String = when (value) {
        Quantization.FP32 -> "FP32"
        Quantization.FP16 -> "FP16"
        Quantization.W8A16 -> "W8A16"
        Quantization.W8A8 -> "W8A8"
        Quantization.INT8 -> "INT8"
        Quantization.UINT8 -> "UINT8"
        Quantization.UNKNOWN -> "Unknown"
    }

    fun formatLabel(value: Format): String = when (value) {
        Format.TFLITE -> "TFLite"
        Format.QNN_CONTEXT_BIN -> "QNN Context"
        Format.OTHER -> "Other"
    }

    fun familyLabel(value: Family): String = when (value) {
        Family.YOLO_SEG -> "YOLO-Seg"
        Family.OSTRACK -> "OSTrack"
        Family.XFEAT -> "XFeat"
        Family.DINOV2 -> "DINOv2" // 🌟  UI 显示标签
        Family.OTHER -> "Other"
    }

    private fun scan(
        assets: AssetManager,
        directory: String,
        result: MutableList<ModelAsset>
    ) {
        val children = try {
            assets.list(directory).orEmpty()
        } catch (_: Throwable) {
            emptyArray()
        }

        for (child in children) {
            val path = if (directory.isEmpty()) child else "$directory/$child"
            val nested = try { assets.list(path).orEmpty() } catch (_: Throwable) { emptyArray() }
            if (nested.isNotEmpty()) {
                scan(assets, path, result)
                continue
            }

            val fileName = child.lowercase(Locale.US)
            if (!isModelFile(fileName)) continue
            val family = detectFamily(fileName)
            val format = detectFormat(fileName)
            val quantization = detectQuantization(fileName, format)
            val support = supportFor(family, format, fileName)

            result += ModelAsset(
                path = path,
                name = path.substringAfterLast('/'),
                family = family,
                format = format,
                quantization = quantization,
                supportedByCurrentAdapter = support.first,
                supportNote = support.second
            )
        }
    }

    private fun isModelFile(name: String): Boolean =
        name.endsWith(".tflite") ||
                name.endsWith(".bin") ||
                name.endsWith(".serialized")

    private fun detectFamily(name: String): Family = when {
        name.contains("dinov2") -> Family.DINOV2 // 🌟 动态识别包含 dinov2 名字的模型
        name.contains("yolo") && name.contains("seg") -> Family.YOLO_SEG
        name.contains("ostrack") -> Family.OSTRACK
        name.contains("xfeat") -> Family.XFEAT
        else -> Family.OTHER
    }

    private fun detectFormat(name: String): Format = when {
        name.endsWith(".tflite") -> Format.TFLITE
        name.endsWith(".bin") || name.endsWith(".serialized") -> Format.QNN_CONTEXT_BIN
        else -> Format.OTHER
    }

    private fun detectQuantization(name: String, format: Format): Quantization = when {
        name.contains("w8a16") -> Quantization.W8A16
        name.contains("w8a8") -> Quantization.W8A8
        name.contains("int8") -> Quantization.INT8
        name.contains("uint8") -> Quantization.UINT8
        name.contains("fp16") -> Quantization.FP16
        format == Format.TFLITE && name.contains("float") -> Quantization.FP32
        else -> Quantization.UNKNOWN
    }

    private fun supportFor(
        family: Family,
        format: Format,
        name: String
    ): Pair<Boolean, String> = when (family) {
        Family.YOLO_SEG -> if (format == Format.TFLITE) {
            true to "当前 YOLO11-Seg TFLite 适配器"
        } else {
            false to "当前 YOLO-Seg 运行时仅接入 TFLite"
        }

        Family.OSTRACK -> if (
            format == Format.QNN_CONTEXT_BIN &&
            Regex("ostrack_256_.*fp16", RegexOption.IGNORE_CASE).containsMatchIn(name)
        ) {
            true to "当前 OSTrack-256 QNN 3-output 适配器"
        } else {
            false to "当前 OSTrack 适配器固定为 256 / 3-output QNN Context；其他尺寸或量化需对应 adapter"
        }

        Family.XFEAT -> if (
            format == Format.QNN_CONTEXT_BIN &&
            Regex("xfeat_(384|480|512|640)_w8a16", RegexOption.IGNORE_CASE).containsMatchIn(name)
        ) {
            true to "当前 XFeat W8A16 QNN 适配器"
        } else {
            false to "当前 XFeat live adapter 为 W8A16 QNN；其他量化/格式需对应 adapter"
        }

        // 🌟 DINOv2 的可用性判定
        Family.DINOV2 -> if (format == Format.QNN_CONTEXT_BIN) {
            true to "已支持任意分辨率的 DINOv2 QNN Context"
        } else {
            false to "DINOv2 目前专属优化了 QNN Context (.bin) 格式"
        }

        Family.OTHER -> false to "未识别模型族"
    }
}