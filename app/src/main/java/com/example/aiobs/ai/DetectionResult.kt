package com.example.aiobs.ai

/**
 * Detection produced by YOLO11n-Seg and consumed by the live tracker.
 * Coordinates are normalized to the current frame.
 */
data class DetectionResult(
    val label: String,
    val confidence: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    /** YOLO11n-seg instance mask in a compact detection-local ROI. */
    val segmentationMask: SegmentationMask? = null
)
