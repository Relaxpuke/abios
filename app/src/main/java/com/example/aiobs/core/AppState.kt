package com.example.aiobs.core

/**
 * 全局运行状态的轻量模型。UI 不再直接充当业务状态仓库。
 */
data class AppState(
    var yoloEnabled: Boolean = true,
    var xfeatEnabled: Boolean = true,
    var reportEnabled: Boolean = true,
    var yoloBackend: ModelRuntimeConfig.Backend = ModelRuntimeConfig.Backend.QNN,
    var proCameraEnabled: Boolean = false,
    var videoWidth: Int = 1920,
    var videoHeight: Int = 1080,
    var videoFps: Int = 60,
    var videoBitrate: Int = 3500 * 1024,
    var resolutionMode: Int = 0,
    var portrait: Boolean = true,
    var zoomLevel: Float = 1.0f,
    var frontCamera: Boolean = false,
    var userStoppingStream: Boolean = true
)

fun AppState.apply1080p() {
    videoWidth = 1920
    videoHeight = 1080
    videoBitrate = 3500 * 1024
    resolutionMode = 0
}

fun AppState.apply720p() {
    videoWidth = 1280
    videoHeight = 720
    videoBitrate = 2000 * 1024
    resolutionMode = 1
}
