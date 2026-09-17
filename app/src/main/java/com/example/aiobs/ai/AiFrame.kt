package com.example.aiobs.ai

/**
 * Camera frame transport. ImageReader data is copied to a safe YUV snapshot
 * before this object crosses into the AI worker thread.
 */
sealed class AiFrame {
    /** ImageReader callback entry timestamp. */
    abstract val availableStartNs: Long
    /** Timestamp immediately before handing the frame to the AI worker queue. */
    abstract val availableEndNs: Long

    data class CameraImage(
        val frame: Yuv420FrameSnapshot,
        override val availableStartNs: Long,
        override val availableEndNs: Long
    ) : AiFrame()
}

/** Safe cross-thread snapshot of a Camera2 YUV_420_888 frame. */
data class Yuv420FrameSnapshot(
    val width: Int,
    val height: Int,
    val yData: ByteArray,
    val uData: ByteArray,
    val vData: ByteArray,
    val yRowStride: Int,
    val yPixelStride: Int,
    val uRowStride: Int,
    val uPixelStride: Int,
    val vRowStride: Int,
    val vPixelStride: Int
)
