package com.example.aiobs.core

/** Standard COCO-80 class catalog for the current YOLO11n-Seg 80-class output contract. */
object YoloClassCatalog {
    const val COUNT = 80
    val NAMES: List<String> = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )
    val DEFAULT_CLASS_IDS: Set<Int> = setOf(0, 2)
    fun nameOf(id: Int): String = NAMES.getOrNull(id) ?: "class_$id"
    fun formatSelected(ids: Set<Int>): String {
        val names = ids.filter { it in 0 until COUNT }.sorted().map(::nameOf)
        return when {
            names.isEmpty() -> "无"
            names.size <= 4 -> names.joinToString(", ")
            else -> "${names.take(4).joinToString(", ")} +${names.size - 4}"
        }
    }
}
