package com.example.aiobs.core

import android.content.Context

/**
 * Persistent runtime configuration for the AI models and the runtime
 * status/report panel.
 *
 * The selected backend is applied when the AI runtime is (re)created.
 */
object ModelRuntimeConfig {

    enum class Backend {
        QNN,
        GPU,
        CPU
    }

    data class Snapshot(
        val yoloEnabled: Boolean = true,
        val yoloBackend: Backend = Backend.QNN,
        val yoloModelAsset: String = DEFAULT_YOLO_ASSET,
        val xfeatEnabled: Boolean = true,
        val xfeatModelAsset: String = DEFAULT_XFEAT_ASSET,
        val ostrackModelAsset: String = DEFAULT_OSTRACK_ASSET,
        val reportEnabled: Boolean = true,
        val dinov2Enabled: Boolean = true,
        val dinov2ModelAsset: String = DEFAULT_DINOV2_ASSET
    )

    private const val PREFS = "ai_model_runtime_config"
    private const val KEY_YOLO_ENABLED = "yolo_enabled"
    private const val KEY_YOLO_BACKEND = "yolo_backend"
    private const val KEY_YOLO_MODEL = "yolo_model_asset"
    private const val KEY_XFEAT_ENABLED = "xfeat_enabled"
    private const val KEY_XFEAT_MODEL = "xfeat_model_asset"
    private const val KEY_OSTRACK_MODEL = "ostrack_model_asset"
    private const val KEY_REPORT_ENABLED = "report_enabled"
    private const val KEY_DINOV2_ENABLED = "dinov2_enabled"
    private const val KEY_DINOV2_MODEL = "dinov2_model_asset"

    const val DEFAULT_YOLO_ASSET = "yolo11n-seg.tflite"
    const val DEFAULT_XFEAT_ASSET = "xfeat_480_w8a16_sm8550.bin.SM8550.bin"
    const val DEFAULT_OSTRACK_ASSET = "ostrack_256_fp16_sm8550_3out.serialized.SM8550.bin"
    const val DEFAULT_DINOV2_ASSET = "dinov2_s14_224_w8a16.bin"

    fun resolveAssets(context: Context, snapshot: Snapshot): Snapshot = snapshot.copy(
        yoloModelAsset = ModelAssetCatalog.resolveSelected(
            context,
            ModelAssetCatalog.Family.YOLO_SEG,
            snapshot.yoloModelAsset,
            DEFAULT_YOLO_ASSET
        ),
        ostrackModelAsset = ModelAssetCatalog.resolveSelected(
            context,
            ModelAssetCatalog.Family.OSTRACK,
            snapshot.ostrackModelAsset,
            DEFAULT_OSTRACK_ASSET
        ),
        xfeatModelAsset = ModelAssetCatalog.resolveSelected(
            context,
            ModelAssetCatalog.Family.XFEAT,
            snapshot.xfeatModelAsset,
            DEFAULT_XFEAT_ASSET
        ),
        dinov2ModelAsset = ModelAssetCatalog.resolveSelected(
            context,
            ModelAssetCatalog.Family.DINOV2,
            snapshot.dinov2ModelAsset,
            DEFAULT_DINOV2_ASSET
        )
    )

    fun load(context: Context): Snapshot {
        val p = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return Snapshot(
            yoloEnabled = p.getBoolean(KEY_YOLO_ENABLED, true),
            yoloBackend = readBackend(p.getString(KEY_YOLO_BACKEND, Backend.QNN.name)),
            yoloModelAsset = p.getString(KEY_YOLO_MODEL, DEFAULT_YOLO_ASSET) ?: DEFAULT_YOLO_ASSET,
            xfeatEnabled = p.getBoolean(KEY_XFEAT_ENABLED, true),
            xfeatModelAsset = p.getString(KEY_XFEAT_MODEL, DEFAULT_XFEAT_ASSET) ?: DEFAULT_XFEAT_ASSET,
            ostrackModelAsset = p.getString(KEY_OSTRACK_MODEL, DEFAULT_OSTRACK_ASSET) ?: DEFAULT_OSTRACK_ASSET,
            reportEnabled = p.getBoolean(KEY_REPORT_ENABLED, true),
            dinov2Enabled = p.getBoolean(KEY_DINOV2_ENABLED, true),
            dinov2ModelAsset = p.getString(KEY_DINOV2_MODEL, DEFAULT_DINOV2_ASSET) ?: DEFAULT_DINOV2_ASSET
        )
    }

    fun save(context: Context, snapshot: Snapshot) {
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_YOLO_ENABLED, snapshot.yoloEnabled)
            .putString(KEY_YOLO_BACKEND, snapshot.yoloBackend.name)
            .putString(KEY_YOLO_MODEL, snapshot.yoloModelAsset)
            .putBoolean(KEY_XFEAT_ENABLED, snapshot.xfeatEnabled)
            .putString(KEY_XFEAT_MODEL, snapshot.xfeatModelAsset)
            .putString(KEY_OSTRACK_MODEL, snapshot.ostrackModelAsset)
            .putBoolean(KEY_REPORT_ENABLED, snapshot.reportEnabled)
            .putBoolean(KEY_DINOV2_ENABLED, snapshot.dinov2Enabled)
            .putString(KEY_DINOV2_MODEL, snapshot.dinov2ModelAsset)
            .apply()
    }

    fun cycleBackend(current: Backend): Backend = when (current) {
        Backend.QNN -> Backend.GPU
        Backend.GPU -> Backend.CPU
        Backend.CPU -> Backend.QNN
    }

    private fun readBackend(value: String?): Backend = try {
        Backend.valueOf(value ?: Backend.QNN.name)
    } catch (_: Throwable) {
        Backend.QNN
    }
}