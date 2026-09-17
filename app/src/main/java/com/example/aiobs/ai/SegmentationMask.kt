package com.example.aiobs.ai

/**
 * Compact YOLO11-Seg instance mask stored in the detection-local ROI.
 * pixels are 0/1 values in row-major order.
 */
data class SegmentationMask(
    val width: Int,
    val height: Int,
    val pixels: ByteArray,
    val area: Int,
    val threshold: Float = 0.5f
) {
    fun contains(x: Int, y: Int): Boolean {
        if (x !in 0 until width || y !in 0 until height) return false
        return pixels[y * width + x].toInt() != 0
    }

    fun copyMask(): SegmentationMask = SegmentationMask(
        width = width,
        height = height,
        pixels = pixels.clone(),
        area = area,
        threshold = threshold
    )
}
