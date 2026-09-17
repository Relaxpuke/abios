package com.example.aiobs.validation

/**

 * Immutable result of an on-device model validation pass.
 *
 * The report contains independent validation results for:
 *
 * * YOLO11n-seg
 * * XFeat
 * * Depth Anything V2
 * * Standard OSNet x1.0 FP16
 *
 * OSNet is optional so older callers remain source-compatible.
 */
data class ModelValidationReport(
    val generatedAtMs: Long,
    val deviceSummary: String,
    val yolo11nSeg: ModelCheck,
    val xfeat: ModelCheck,
    val depth: ModelCheck,
    val overallPassed: Boolean,
    val notes: List<String> = emptyList(),
    val osnet: ModelCheck? = null
) {

    data class ModelCheck(
        val name: String,
        val assetName: String,
        val assetPresent: Boolean,
        val assetSizeBytes: Long,
        val loadMs: Long,
        val allocationMs: Long,
        val warmupCount: Int,
        val benchmarkCount: Int,
        val meanMs: Double,
        val medianMs: Double,
        val minMs: Double,
        val maxMs: Double,
        val inputSummary: String,
        val outputSummary: String,
        val sanitySummary: String,
        val contractSummary: String,
        val passed: Boolean,
        val warnings: List<String> = emptyList(),
        val error: String? = null,
        val backend: String = "CPU"
    )

    fun formatForUi(): String = buildString {

        appendLine("V11.8 MODEL VALIDATION")
        appendLine("======================")

        appendLine(deviceSummary)
        appendLine()

        appendModel(
            title = "YOLO11n-seg",
            check = yolo11nSeg
        )

        appendLine()

        appendModel(
            title = "XFeat",
            check = xfeat
        )

        appendLine()

        appendModel(
            title = "Depth Anything V2",
            check = depth
        )

        osnet?.let {

            appendLine()

            appendModel(
                title = "OSNet x1.0 FP16",
                check = it
            )
        }

        if (notes.isNotEmpty()) {

            appendLine()
            appendLine("NOTES")

            notes.forEach {
                appendLine("- $it")
            }
        }

        appendLine()
        appendLine(
            "OVERALL: ${if (overallPassed) "PASS" else "FAIL"}"
        )


    }

    private fun StringBuilder.appendModel(
        title: String,
        check: ModelCheck
    ) {

        appendLine(
            "$title: ${if (check.passed) "PASS" else "FAIL"}"
        )

        appendLine(
            "Asset: ${check.assetName} | ${formatBytes(check.assetSizeBytes)}"
        )

        if (!check.assetPresent) {

            appendLine(
                "Asset: MISSING"
            )

            check.error?.let {
                appendLine(
                    "ERROR: $it"
                )
            }

            return
        }

        appendLine(
            "Backend: ${check.backend}"
        )

        appendLine(
            "Load: ${check.loadMs} ms | " +
                    "Allocate: ${check.allocationMs} ms"
        )

        appendLine(
            "Input: ${check.inputSummary}"
        )

        appendLine(
            "Output: ${check.outputSummary}"
        )

        appendLine(
            "Benchmark: " +
                    "${check.warmupCount} warmup + " +
                    "${check.benchmarkCount} runs | " +
                    "avg=${formatMs(check.meanMs)} | " +
                    "median=${formatMs(check.medianMs)} | " +
                    "min=${formatMs(check.minMs)} | " +
                    "max=${formatMs(check.maxMs)}"
        )

        appendLine(
            "Sanity: ${check.sanitySummary}"
        )

        appendLine(
            "Contract: ${check.contractSummary}"
        )

        check.warnings.forEach {
            appendLine(
                "WARN: $it"
            )
        }

        check.error?.let {
            appendLine(
                "ERROR: $it"
            )
        }

    }

    companion object {

        private fun formatMs(
            ms: Double
        ): String =
            "%.2f ms".format(ms)

        private fun formatBytes(
            bytes: Long
        ): String {

            if (bytes <= 0L) {
                return "0 B"
            }

            val mb =
                bytes /
                        (
                                1024.0 *
                                        1024.0
                                )

            return "%.2f MB".format(
                mb
            )
        }
    }
}
