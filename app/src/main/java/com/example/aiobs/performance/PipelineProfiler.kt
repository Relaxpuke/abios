package com.example.aiobs.performance

/**
 * V4：单帧 Camera -> AI 生命周期的轻量性能剖析器。
 *
 * 时间定义：
 * CAMERA = ImageReader onImageAvailable() 进入到 YUV snapshot copy 完成。
 * QUEUE  = snapshot handoff 完成到 AI worker 真正开始执行。
 * PREP   = YUV -> Bitmap 准备。
 * IMAGE  = 预留的输入图像构建阶段。
 * INFER  = 当前帧实际执行的模型推理耗时。
 * PARSE  = 当前帧检测结果解析耗时。
 * TRACK  = 当前帧最终跟踪/状态收尾耗时。
 * E2E    = ImageReader callback 进入到当前帧 tracker/publish 完成。
 *
 * 额外的模型阶段统计只在该帧真正执行时更新，不会被其它状态的“last value”覆盖。
 */
class PipelineProfiler {

    data class FrameToken(
        val cameraAvailableNs: Long,
        val cameraHandoffNs: Long
    )

    data class Snapshot(
        val captureMs: Float,
        val queueMs: Float,
        val prepareMs: Float,
        val imageBuildMs: Float,
        val inferenceMs: Float,
        val parseMs: Float,
        val trackerMs: Float,
        val endToEndMs: Float,
        val yoloTotalMs: Float,
        val yoloInferMs: Float,
        val ostrackMs: Float
    ) {
        companion object {
            fun empty() = Snapshot(
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f
            )
        }
    }

    data class FrameTimings(
        val token: FrameToken,
        val workerStartNs: Long,
        val prepareStartNs: Long,
        val prepareEndNs: Long,
        val trackerStartNs: Long,
        val imageBuildMs: Float,
        val inferenceMs: Float,
        val parseMs: Float,
        val yoloTotalMs: Float?,
        val yoloInferMs: Float?,
        val ostrackMs: Float?
    )

    @Volatile
    private var snapshot = Snapshot.empty()

    fun newToken(
        cameraAvailableNs: Long,
        cameraHandoffNs: Long
    ): FrameToken = FrameToken(cameraAvailableNs, cameraHandoffNs)

    fun buildTimings(
        token: FrameToken,
        workerStartNs: Long,
        prepareStartNs: Long,
        prepareEndNs: Long,
        trackerStartNs: Long,
        imageBuildMs: Float,
        inferenceMs: Float,
        parseMs: Float,
        yoloTotalMs: Float?,
        yoloInferMs: Float?,
        ostrackMs: Float?
    ): FrameTimings {
        return FrameTimings(
            token = token,
            workerStartNs = workerStartNs,
            prepareStartNs = prepareStartNs,
            prepareEndNs = prepareEndNs,
            trackerStartNs = trackerStartNs,
            imageBuildMs = imageBuildMs,
            inferenceMs = inferenceMs,
            parseMs = parseMs,
            yoloTotalMs = yoloTotalMs,
            yoloInferMs = yoloInferMs,
            ostrackMs = ostrackMs
        )
    }

    fun finish(frame: FrameTimings, trackerEndNs: Long): Snapshot {
        val captureMs = elapsedMs(frame.token.cameraAvailableNs, frame.token.cameraHandoffNs)
        val queueMs = elapsedMs(frame.token.cameraHandoffNs, frame.workerStartNs)
        val prepareMs = elapsedMs(frame.prepareStartNs, frame.prepareEndNs)
        val trackerMs = elapsedMs(frame.trackerStartNs, trackerEndNs)
        val endToEndMs = elapsedMs(frame.token.cameraAvailableNs, trackerEndNs)

        val previous = snapshot
        snapshot = Snapshot(
            captureMs = ema(previous.captureMs, captureMs),
            queueMs = ema(previous.queueMs, queueMs),
            prepareMs = ema(previous.prepareMs, prepareMs),
            imageBuildMs = ema(previous.imageBuildMs, frame.imageBuildMs),
            inferenceMs = ema(previous.inferenceMs, frame.inferenceMs),
            parseMs = ema(previous.parseMs, frame.parseMs),
            trackerMs = ema(previous.trackerMs, trackerMs),
            endToEndMs = ema(previous.endToEndMs, endToEndMs),
            yoloTotalMs = emaIfPresent(previous.yoloTotalMs, frame.yoloTotalMs),
            yoloInferMs = emaIfPresent(previous.yoloInferMs, frame.yoloInferMs),
            ostrackMs = emaIfPresent(previous.ostrackMs, frame.ostrackMs)
        )

        return snapshot
    }

    fun snapshot(): Snapshot = snapshot

    fun reset() {
        snapshot = Snapshot.empty()
    }

    private fun elapsedMs(startNs: Long, endNs: Long): Float {
        if (startNs <= 0L || endNs <= startNs) return 0f
        return (endNs - startNs) / 1_000_000f
    }

    private fun ema(previous: Float, current: Float): Float {
        return if (previous <= 0f) current else previous * 0.85f + current * 0.15f
    }

    private fun emaIfPresent(previous: Float, current: Float?): Float {
        return current?.takeIf { it >= 0f }?.let { ema(previous, it) } ?: previous
    }
}
