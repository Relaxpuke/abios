package com.example.aiobs.core

import android.content.Context

/**
 * Runtime tuning knobs for the current YOLO-Seg + OSTrack + XFeat pipeline.
 * Values are persisted separately from model/backend enablement.
 *
 * XFeat uses one business-level identity threshold. Matching/geometry code only
 * enforces mathematical and data-validity requirements.
 */
data class RuntimeTuningConfig(
    val yoloConfidenceThreshold: Float = 0.35f,
    val yoloSelectedClassIds: Set<Int> = YoloClassCatalog.DEFAULT_CLASS_IDS,
    val ostrackMinScore: Float = 0.50f,
    val xfeatIdentityThreshold: Float = 0.50f,
    val lockedYoloRefreshIntervalMs: Long = 10_000L,
    val lockedYoloFirstRefreshDelayMs: Long = 1_000L
) {
    fun normalized(): RuntimeTuningConfig = copy(
        yoloSelectedClassIds = yoloSelectedClassIds.filter { it in 0 until YoloClassCatalog.COUNT }.toSet().ifEmpty { YoloClassCatalog.DEFAULT_CLASS_IDS },
        yoloConfidenceThreshold = yoloConfidenceThreshold.coerceIn(0.05f, 0.95f),
        ostrackMinScore = ostrackMinScore.coerceIn(0.20f, 0.95f),
        xfeatIdentityThreshold = xfeatIdentityThreshold.coerceIn(0.05f, 0.95f),
        lockedYoloRefreshIntervalMs = lockedYoloRefreshIntervalMs.coerceIn(1_000L, 60_000L),
        lockedYoloFirstRefreshDelayMs = lockedYoloFirstRefreshDelayMs.coerceIn(0L, 10_000L)
    )

    companion object {
        private const val PREFS = "ai_runtime_tuning"
        private const val K_YOLO = "yolo_conf_threshold"
        private const val K_YOLO_CLASSES = "yolo_selected_class_ids"
        private const val K_OSTRACK = "ostrack_min_score"
        private const val K_OSTRACK_LEGACY = "ostrack_lost_threshold"
        private const val K_IDENTITY = "xfeat_identity_threshold"
        private const val K_IDENTITY_LEGACY_TEMPORAL = "xfeat_temporal_min_score"
        private const val K_LOCKED_INTERVAL = "locked_yolo_refresh_interval_ms"
        private const val K_FIRST_DELAY = "locked_yolo_first_refresh_delay_ms"

        fun load(context: Context): RuntimeTuningConfig {
            val p = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val storedClasses = p.getString(K_YOLO_CLASSES, null)
                ?.split(",")
                ?.mapNotNull { it.trim().toIntOrNull() }
                ?.toSet()
                ?: YoloClassCatalog.DEFAULT_CLASS_IDS
            return RuntimeTuningConfig(
                yoloConfidenceThreshold = p.getFloat(K_YOLO, 0.35f),
                yoloSelectedClassIds = storedClasses,
                ostrackMinScore = p.getFloat(
                    K_OSTRACK,
                    p.getFloat(K_OSTRACK_LEGACY, 0.50f)
                ),
                xfeatIdentityThreshold = p.getFloat(
                    K_IDENTITY,
                    p.getFloat(K_IDENTITY_LEGACY_TEMPORAL, 0.50f)
                ),
                lockedYoloRefreshIntervalMs = p.getLong(K_LOCKED_INTERVAL, 10_000L),
                lockedYoloFirstRefreshDelayMs = p.getLong(K_FIRST_DELAY, 1_000L)
            ).normalized()
        }

        fun save(context: Context, value: RuntimeTuningConfig) {
            val v = value.normalized()
            context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit()
                .putFloat(K_YOLO, v.yoloConfidenceThreshold)
                .putString(K_YOLO_CLASSES, v.yoloSelectedClassIds.sorted().joinToString(","))
                .putFloat(K_OSTRACK, v.ostrackMinScore)
                .putFloat(K_IDENTITY, v.xfeatIdentityThreshold)
                .putLong(K_LOCKED_INTERVAL, v.lockedYoloRefreshIntervalMs)
                .putLong(K_FIRST_DELAY, v.lockedYoloFirstRefreshDelayMs)
                .apply()
        }

        fun defaults(): RuntimeTuningConfig = RuntimeTuningConfig()
    }
}
