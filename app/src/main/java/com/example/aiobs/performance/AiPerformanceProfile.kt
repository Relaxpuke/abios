package com.example.aiobs.performance

import android.os.Build

/**
 * V3 设备级 AI 性能策略。
 *
 * 针对当前两台测试机：
 * - Snapdragon 865 / SM8250：优先 CPU，避免 GPU delegate 在老平台上的额外调度/驱动开销。
 * - Snapdragon 8 Gen 2 / SM8550：V3.2 暂时固定 CPU，规避已实测的 GPU 空检测异常。
 *
 * 给 Radar 提供按 SoC 选择的初始节奏和阈值。
 */
data class AiPerformanceProfile(
    val socModel: String,
    val profileName: String,
    val targetInferenceFps: Int,
    val captureIntervalMs: Long,
    val scoreThreshold: Float
)

object AiPerformanceProfileFactory {

    fun create(): AiPerformanceProfile {
        val soc = getSocModel().uppercase()

        return when {
            soc.contains("SM8550") -> {
                // V3.2: S23 / Snapdragon 8 Gen 2 的 Google vision GPU 路径已实测
                // 出现“推理耗时很低但持续 0 框”的异常，暂时禁用 GPU。
                // 等模型/vision 版本重新验证后再重新开放 GPU。
                AiPerformanceProfile(
                    socModel = soc,
                    profileName = "Snapdragon 8 Gen 2 (CPU stable)",
                    targetInferenceFps = 12,
                    captureIntervalMs = 83L,
                    scoreThreshold = 0.35f
                )
            }

            soc.contains("SM8250") -> {
                AiPerformanceProfile(
                    socModel = soc,
                    profileName = "Snapdragon 865",
                    targetInferenceFps = 10,
                    captureIntervalMs = 100L,
                    scoreThreshold = 0.35f
                )
            }

            else -> {
                AiPerformanceProfile(
                    socModel = soc.ifBlank { "UNKNOWN" },
                    profileName = "Generic Android",
                    targetInferenceFps = 10,
                    captureIntervalMs = 100L,
                    scoreThreshold = 0.35f
                )
            }
        }
    }

    private fun getSocModel(): String {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL.ifBlank { Build.HARDWARE }
        } else {
            Build.HARDWARE
        }
    }
}
