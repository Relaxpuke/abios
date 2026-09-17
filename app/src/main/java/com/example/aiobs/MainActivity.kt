package com.example.aiobs

import android.Manifest
import android.content.Context
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.transition.TransitionManager
import android.util.Log
import android.view.Surface
import android.view.View
import android.view.WindowManager
import android.view.OrientationEventListener
import android.graphics.Color
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.SeekBar
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.aiobs.ai.TfliteAcceleration
import com.example.aiobs.ai.VisualRadarController
import com.example.aiobs.camera.CameraCapabilities
import com.example.aiobs.camera.CameraCapabilitiesReader
import com.example.aiobs.camera.CameraController
import com.example.aiobs.camera.CameraSettingsManager
import com.example.aiobs.camera.CameraSettingsStore
import com.example.aiobs.core.AppState
import com.example.aiobs.core.ModelRuntimeConfig
import com.example.aiobs.core.ModelAssetCatalog
import com.example.aiobs.core.RuntimeTuningConfig
import com.example.aiobs.core.YoloClassCatalog
import com.example.aiobs.core.apply1080p
import com.example.aiobs.core.apply720p
import com.example.aiobs.device.DeviceStatusController
import com.example.aiobs.stream.StreamController
import com.example.aiobs.utils.SrtProfileManager
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import java.util.Locale

class MainActivity : AppCompatActivity(), StreamController.Listener {

    private lateinit var state: AppState
    private lateinit var surfaceView: com.pedro.library.view.OpenGlView
    private lateinit var aiOverlay: AiOverlayView

    // HUD 悬浮控件
    private lateinit var hudStatusDot: View
    private lateinit var hudStatusText: TextView
    private lateinit var hudTempText: TextView
    private lateinit var hudBatteryText: TextView
    private lateinit var hudYoloTime: TextView
    private lateinit var hudOsTrackTime: TextView
    private lateinit var hudXFeatTime: TextView
    private lateinit var btnOpenDashboard: TextView
    private lateinit var dashboardContainer: View
    private lateinit var hudStreamInfoPill: View
    private lateinit var hudStreamInfoText: TextView

    // Dashboard 选项卡控件
    private lateinit var tabCamera: TextView
    private lateinit var tabAi: TextView
    private lateinit var tabRuntime: TextView
    private lateinit var tabReport: TextView
    private lateinit var pageCamera: View
    private lateinit var pageAi: View
    private lateinit var pageRuntime: View
    private lateinit var pageReport: View

    private lateinit var hudFpsText: TextView
    private lateinit var btnCloseDashboard: TextView
    private lateinit var hudStreamPill: View
    private lateinit var hudStreamDot: View
    private lateinit var hudStreamText: TextView
    private lateinit var hudNetworkText: TextView
    private lateinit var hudDinov2Time: TextView // 🌟 新增

    // 推流设置控件
    private lateinit var btnStream: Button
    private lateinit var etHost: EditText
    private lateinit var etPort: EditText
    private lateinit var etStreamName: EditText

    private lateinit var streamController: StreamController
    private lateinit var cameraController: CameraController
    private lateinit var radarController: VisualRadarController
    private lateinit var deviceStatusController: DeviceStatusController
    private var runtimeConfig = ModelRuntimeConfig.Snapshot()
    private var runtimeTuning = RuntimeTuningConfig.defaults()
    private var cameraAdvanced = CameraSettingsManager.Advanced()

    private var manualOrientationLocked = false

    // Physical device orientation is tracked independently from the Activity UI lock.
    // This is required so AI inference can stay upright even when the UI is locked
    // to portrait/landscape and the user physically rotates the phone.
    @Volatile
    private var physicalDisplayRotation = Surface.ROTATION_0

    private var orientationListener: OrientationEventListener? = null
    private val statusHandler = Handler(Looper.getMainLooper())
    private val statusRunnable = object : Runnable {
        override fun run() {
            renderStreamStatus()
            statusHandler.postDelayed(this, STATUS_INTERVAL_MS)
        }
    }

    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.READ_PHONE_STATE,
        Manifest.permission.ACCESS_FINE_LOCATION
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_YES)

        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        orientationListener = object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                if (orientation == ORIENTATION_UNKNOWN) return
                val newRotation = when {
                    orientation >= 315 || orientation < 45 -> Surface.ROTATION_0
                    orientation < 135 -> Surface.ROTATION_90
                    orientation < 225 -> Surface.ROTATION_180
                    else -> Surface.ROTATION_270
                }
                if (newRotation != physicalDisplayRotation) {
                    physicalDisplayRotation = newRotation
                    if (manualOrientationLocked && ::aiOverlay.isInitialized) {
                        syncOverlayGeometry()
                    }
                    // Physical posture is tracked for diagnostics/UI-only behavior.
                    // AI rotation itself intentionally stays tied to the locked display
                    // orientation, so holding a locked screen sideways does not rotate
                    // the model input into a different coordinate system.
                }
            }
        }
        physicalDisplayRotation = currentDisplayRotation()

        manualOrientationLocked = savedInstanceState?.getBoolean(KEY_MANUAL_ORIENTATION, false)
            ?: CameraSettingsStore.loadOrientationLocked(this, false)
        runtimeConfig = ModelRuntimeConfig.resolveAssets(this, ModelRuntimeConfig.load(this))
        ModelRuntimeConfig.save(this, runtimeConfig)
        runtimeTuning = RuntimeTuningConfig.load(this)
        TfliteAcceleration.initialize(this)
        state = AppState(
            yoloEnabled = runtimeConfig.yoloEnabled,
            xfeatEnabled = runtimeConfig.xfeatEnabled,
            reportEnabled = runtimeConfig.reportEnabled,
            yoloBackend = runtimeConfig.yoloBackend,
        )

        cameraAdvanced = CameraSettingsStore.load(this, state)
        state.proCameraEnabled = cameraAdvanced.afMode == android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_OFF ||
                cameraAdvanced.iso != null || cameraAdvanced.exposureTimeNs != null

        physicalDisplayRotation = currentDisplayRotation()

        if (manualOrientationLocked) {
            state.portrait = resources.configuration.orientation == Configuration.ORIENTATION_PORTRAIT
            requestedOrientation = if (state.portrait) ActivityInfo.SCREEN_ORIENTATION_PORTRAIT else ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        } else {
            requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_FULL_SENSOR
            state.portrait = resources.configuration.orientation == Configuration.ORIENTATION_PORTRAIT
        }

        bindViews()
        loadProfile()
        initControllers()
        normalizeLoadedCameraSettings()
        setupControls()
        requestPermissionsIfNeeded()
        statusHandler.post(statusRunnable)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)

        if (!manualOrientationLocked) {
            state.portrait = newConfig.orientation == Configuration.ORIENTATION_PORTRAIT
        }

        physicalDisplayRotation = currentDisplayRotation()

        // Keep the display geometry in sync without throwing away the AI lock.
        // The camera preview is rebuilt, while VisualRadarController preserves the
        // logical lock and only refreshes model templates for the new AI frame space.
        if (::state.isInitialized && ::tabCamera.isInitialized) {
            renderCameraControls()
            if (::streamController.isInitialized && ::cameraController.isInitialized && ::aiOverlay.isInitialized) {
                syncOverlayGeometry()
                if (cameraController.isPreviewing()) {
                    Log.i(TAG, "Orientation changed: rebuilding camera preview for previewRotation=${computePreviewRotation()} aiRotation=${computeCurrentRotation()}")
                    rebuildPreviewAfterConfiguration()
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        orientationListener?.enable()
    }

    override fun onPause() {
        orientationListener?.disable()
        super.onPause()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean(KEY_MANUAL_ORIENTATION, manualOrientationLocked)
        persistCameraSettings()
        super.onSaveInstanceState(outState)
    }

    private fun bindViews() {
        surfaceView = findViewById(R.id.surfaceView)
        aiOverlay = findViewById(R.id.aiOverlay)

        // HUD 绑定
        hudStatusDot = findViewById(R.id.hudStatusDot)
        hudStatusText = findViewById(R.id.hudStatusText)
        hudTempText = findViewById(R.id.hudTempText)
        hudBatteryText = findViewById(R.id.hudBatteryText)
        hudYoloTime = findViewById(R.id.hudYoloTime)
        hudOsTrackTime = findViewById(R.id.hudOsTrackTime)
        hudXFeatTime = findViewById(R.id.hudXFeatTime)
        hudDinov2Time = findViewById(R.id.hudDinov2Time) // 🌟 新增
        btnOpenDashboard = findViewById(R.id.btnOpenDashboard)
        dashboardContainer = findViewById(R.id.dashboardContainer)
        hudFpsText = findViewById(R.id.hudFpsText)
        hudStreamPill = findViewById(R.id.hudStreamPill)
        hudStreamDot = findViewById(R.id.hudStreamDot)
        hudStreamText = findViewById(R.id.hudStreamText)
        hudNetworkText = findViewById(R.id.hudNetworkText)
        hudStreamInfoPill = findViewById(R.id.hudStreamInfoPill)
        hudStreamInfoText = findViewById(R.id.hudStreamInfoText)


        // Dashboard Tabs 绑定
        tabCamera = findViewById(R.id.tabCamera)
        tabAi = findViewById(R.id.tabAi)
        tabRuntime = findViewById(R.id.tabRuntime)
        tabReport = findViewById(R.id.tabReport)

        pageCamera = findViewById(R.id.pageCamera)
        pageAi = findViewById(R.id.pageAi)
        pageRuntime = findViewById(R.id.pageRuntime)
        pageReport = findViewById(R.id.pageReport)

        // 推流绑定
        btnStream = findViewById(R.id.btnStream)
        etHost = findViewById(R.id.etHost)
        etPort = findViewById(R.id.etPort)
        etStreamName = findViewById(R.id.etStreamName)
        btnCloseDashboard = findViewById(R.id.btnCloseDashboard)
    }

    private fun loadProfile() {
        etHost.setText(SrtProfileManager.getHost(this))
        etPort.setText(SrtProfileManager.getPort(this))
        etStreamName.setText(SrtProfileManager.getStreamName(this))
    }

    private fun setRowValue(rowId: Int, value: String, highlight: Boolean = false) {
        val row = findViewById<LinearLayout>(rowId) ?: return
        var valueView = row.findViewWithTag<TextView>("value_text")
        if (valueView == null) {
            valueView = TextView(this).apply {
                tag = "value_text"
                textSize = 14f
                setTypeface(null, android.graphics.Typeface.BOLD)
            }
            row.addView(valueView)
        }
        valueView.text = value
        valueView.setTextColor(if (highlight) Color.parseColor("#7EC8FF") else Color.WHITE)
        valueView.alpha = if (value.contains("关") || value.contains("OFF")) 0.5f else 1.0f
    }

    private fun currentDisplayRotation(): Int {
        val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        return wm.defaultDisplay.rotation
    }

    private fun physicalRotationFromSensor(): Int = physicalDisplayRotation

    private fun computeCurrentRotation(): Int {
        // IMPORTANT:
        // AI input must be normalized to the CURRENT UI/display orientation,
        // not to the physical handset posture.
        //
        // In manual-lock mode the display rotation intentionally stays fixed while
        // the user may hold the phone sideways. Using the physical orientation here
        // would make the AI bitmap switch to a sideways landscape/portrait coordinate
        // system even though the preview and overlay remain locked. That is exactly
        // what caused YOLO to receive a non-canonical frame when the locked device
        // was held horizontally.
        //
        // In auto mode currentDisplayRotation() tracks the actual display orientation,
        // so the same formula remains correct and automatically adapts at 90/270°.
        val displayRotation = currentDisplayRotation()
        val degrees = when (displayRotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        val sensorOrientation = cameraController.getSensorOrientation()
        return if (state.frontCamera) {
            // Mirror-aware Camera2 relative rotation for the front camera.
            (sensorOrientation + degrees) % 360
        } else {
            (sensorOrientation - degrees + 360) % 360
        }
    }

    private fun computePreviewRotation(): Int {
        // The displayed preview follows the UI orientation. In locked mode this must
        // stay fixed even if the phone is physically rotated; AI uses computeCurrentRotation()
        // and AiOverlayView maps AI coordinates back into this preview coordinate space.
        if (manualOrientationLocked) return if (state.portrait) 90 else 0
        return computeCurrentRotation()
    }

    private fun syncOverlayGeometry() {
        val aiRotation = computeCurrentRotation()
        val isPortraitMode = if (manualOrientationLocked) {
            state.portrait
        } else {
            aiRotation % 180 != 0
        }

        val realWidth = streamController.camera().streamWidth
        val realHeight = streamController.camera().streamHeight

        val w = if (realWidth > 0) realWidth else state.videoWidth
        val h = if (realHeight > 0) realHeight else state.videoHeight

        val longEdge = max(w, h)
        val shortEdge = min(w, h)

        val previewRotation = computePreviewRotation()
        aiOverlay.setSourceGeometry(
            if (isPortraitMode) shortEdge else longEdge,
            if (isPortraitMode) longEdge else shortEdge
        )
        aiOverlay.setCoordinateRotations(aiRotation, previewRotation)
    }

    private fun initControllers() {
        deviceStatusController = DeviceStatusController(this)
        streamController = StreamController(
            surfaceView = surfaceView,
            state = state,
            callback = this,
            rotationProvider = { computeCurrentRotation() },
            previewRotationProvider = { computePreviewRotation() }
        )
        cameraController = CameraController(this, streamController.camera(), surfaceView, state)

        radarController = createRadarController()

        cameraController.attachSurfaceCallbacks(
            onReady = {
                startCameraPipeline()
            },
            onDestroyed = {
                // A Surface recreation is not an AI-session reset. Keep the logical
                // lock/tracker alive; the next Camera frame will reconcile orientation.
                aiOverlay.clearTargets()
            }
        )
    }

    private fun createRadarController(): VisualRadarController {
        return VisualRadarController(
            frameSource = cameraController.aiFrameSource(),
            context = this,
            runtimeConfig = runtimeConfig,
            tuningConfig = runtimeTuning,
            rotationProvider = { computeCurrentRotation() }
        ) { targets ->
            aiOverlay.updateTargets(targets)
            aiOverlay.setLockedTargetId(radarController.getLockedTargetId())
        }
    }

    private fun saveRuntimeConfigAndRecreateAi() {
        runtimeConfig = ModelRuntimeConfig.Snapshot(
            yoloEnabled = state.yoloEnabled,
            yoloBackend = state.yoloBackend,
            yoloModelAsset = runtimeConfig.yoloModelAsset,
            xfeatEnabled = state.xfeatEnabled,
            xfeatModelAsset = runtimeConfig.xfeatModelAsset,
            ostrackModelAsset = runtimeConfig.ostrackModelAsset,
            dinov2Enabled = runtimeConfig.dinov2Enabled,   // 🌟 保存 DINOv2 开关状态
            dinov2ModelAsset = runtimeConfig.dinov2ModelAsset, // 🌟 保存当前 DINOv2 模型
            reportEnabled = state.reportEnabled
        )
        ModelRuntimeConfig.save(this, runtimeConfig)

        val shouldRun = state.yoloEnabled && cameraController.isPreviewing()
        try { radarController.destroy() } catch (_: Throwable) {}
        radarController = createRadarController()
        aiOverlay.clearTargets()
        aiOverlay.setLockedTargetId(null)
        if (shouldRun) {
            try { radarController.start() } catch (t: Throwable) {
                Log.e(TAG, "AI runtime reconfigure start failed", t)
                state.yoloEnabled = false
            }
        }
        renderStreamStatus()
    }

    private fun startCameraPipeline() {
        if (!hasAllRequiredPermissions()) return

        if (!streamController.preparePreview()) {
            toast("相机参数准备失败")
            return
        }

        if (!cameraController.prepareAiFrameSource()) {
            toast("AI视频输入初始化失败")
            return
        }

        if (!cameraController.isPreviewing()) {
            cameraController.startPreview()
        }

        cameraController.applySettings(cameraAdvanced)
        cameraController.applySettingsDelayed()
        syncOverlayGeometry()

        if (state.yoloEnabled) {
            try {
                radarController.start()
            } catch (t: Throwable) {
                Log.e(TAG, "YOLO start failed", t)
            }
        }
    }

    private fun rebuildPreviewAfterConfiguration() {
        if (streamController.isStreaming()) {
            // A live SRT stream cannot safely reconfigure the encoder rotation in-place.
            // Keep the active stream untouched; the orientation selector also blocks
            // manual changes while streaming.
            toast("推流中无法切换画面方向")
            return
        }
        // Rebuild only the camera/preview. Do NOT stop VisualRadarController here:
        // doing so resets the tracker and clears a user lock during rotation.
        cameraController.rebuildPreview(
            prepare = {
                streamController.preparePreview()
            },
            afterRestart = {
                syncOverlayGeometry()
                if (state.yoloEnabled) {
                    try { radarController.start() } catch (t: Throwable) { }
                }
                cameraController.applySettingsDelayed()
            },
            delayMs = 250L
        )
    }

    private fun normalizeLoadedCameraSettings() {
        val caps = runCatching { cameraController.getCameraCapabilities(forceRefresh = true) }.getOrNull()
            ?: return

        if (caps.outputSizes.isNotEmpty() &&
            caps.outputSizes.none { it.width == state.videoWidth && it.height == state.videoHeight }) {
            val preferred = caps.outputSizes.firstOrNull { it.width == 1920 && it.height == 1080 }
                ?: caps.outputSizes.firstOrNull { it.width == 1280 && it.height == 720 }
                ?: caps.outputSizes.first()
            state.videoWidth = preferred.width
            state.videoHeight = preferred.height
            state.videoBitrate = estimateBitrate(preferred.width, preferred.height)
        }

        if (!caps.fpsRanges.any { state.videoFps in it.lower..it.upper }) {
            val preferredFps = listOf(30, 25, 24, 60, 20, 15).firstOrNull { fps ->
                caps.fpsRanges.any { range -> fps in range.lower..range.upper }
            }
            state.videoFps = preferredFps ?: caps.fpsRanges.firstOrNull()?.upper ?: 30
        }

        state.zoomLevel = state.zoomLevel.coerceIn(1f, caps.maxDigitalZoom.coerceAtLeast(1f))
        state.videoBitrate = if (state.videoBitrate > 0) state.videoBitrate else estimateBitrate(state.videoWidth, state.videoHeight)

        cameraAdvanced = CameraSettingsStore.normalizeAdvanced(cameraAdvanced, caps)
        state.proCameraEnabled = cameraAdvanced.afMode == android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_OFF ||
                cameraAdvanced.iso != null || cameraAdvanced.exposureTimeNs != null

        CameraSettingsStore.save(this, state, cameraAdvanced, manualOrientationLocked)
    }

    private fun persistCameraSettings() {
        CameraSettingsStore.save(this, state, cameraAdvanced, manualOrientationLocked)
    }

    private fun resetCameraSettingsToDefaults() {
        val caps = activeCameraCapabilities()
        state.zoomLevel = 1f
        state.videoFps = caps?.fpsRanges?.firstOrNull { 30 in it.lower..it.upper }?.upper
            ?: caps?.fpsRanges?.firstOrNull()?.upper
                    ?: 30
        val preferred = caps?.outputSizes?.firstOrNull { it.width == 1920 && it.height == 1080 }
            ?: caps?.outputSizes?.firstOrNull { it.width == 1280 && it.height == 720 }
            ?: caps?.outputSizes?.firstOrNull()
        preferred?.let {
            state.videoWidth = it.width
            state.videoHeight = it.height
            state.videoBitrate = estimateBitrate(it.width, it.height)
        }
        cameraAdvanced = CameraSettingsManager.Advanced()
        state.proCameraEnabled = false
        manualOrientationLocked = false
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_FULL_SENSOR
        state.portrait = resources.configuration.orientation == Configuration.ORIENTATION_PORTRAIT

        persistCameraSettings()
        renderCameraControls()
        if (cameraController.isPreviewing() || streamController.isStreaming()) {
            cameraController.applySettings(cameraAdvanced)
        }
    }

    private fun setupControls() {
        // AI Overlay 点击锁定逻辑
        aiOverlay.setLockCallbacks(
            onLockToggle = { id -> radarController.toggleLock(id); aiOverlay.setLockedTargetId(radarController.getLockedTargetId()) },
            onClearLock = { radarController.clearLock(); aiOverlay.setLockedTargetId(null) }
        )

        // 绑定 Dashboard 容器的展开与收起
        fun toggleDashboard() {
            TransitionManager.beginDelayedTransition(findViewById(android.R.id.content))
            val isVisible = dashboardContainer.visibility == View.VISIBLE
            dashboardContainer.visibility = if (isVisible) View.GONE else View.VISIBLE
        }
        btnOpenDashboard.setOnClickListener { toggleDashboard() }
        btnCloseDashboard.setOnClickListener { toggleDashboard() }

        // 绑定 Tab 切换
        val tabs = listOf(tabCamera, tabAi, tabRuntime, tabReport)
        val pages = listOf(pageCamera, pageAi, pageRuntime, pageReport)
        fun selectTab(selectedIndex: Int) {
            for (i in tabs.indices) {
                val isSelected = (i == selectedIndex)
                tabs[i].setTextColor(if (isSelected) Color.WHITE else Color.parseColor("#888888"))
                pages[i].visibility = if (isSelected) View.VISIBLE else View.GONE
            }
        }
        tabCamera.setOnClickListener { selectTab(0) }
        tabAi.setOnClickListener { selectTab(1) }
        tabRuntime.setOnClickListener { selectTab(2) }
        tabReport.setOnClickListener { selectTab(3) }
        selectTab(0) // 默认显示 Camera Tab

        // --- Camera 控制组 ---
        findViewById<View>(R.id.tvToggleCamera)?.setOnClickListener { showCameraCapabilitiesDialog() }
        findViewById<View>(R.id.tvToggleLens)?.setOnClickListener { showLensSelectionDialog() }
        findViewById<View>(R.id.tvToggleRes)?.setOnClickListener { showResolutionDialog() }
        findViewById<View>(R.id.tvPerfFps)?.setOnClickListener { showFpsDialog() }
        findViewById<View>(R.id.tvToggleZoom)?.setOnClickListener { showZoomDialog() }
        findViewById<View>(R.id.tvToggleOri)?.setOnClickListener {
            if (streamController.isStreaming()) { toast("推流中无法切换方向"); return@setOnClickListener }
            showOrientationDialog()
        }

        // --- AI Models 控制组 ---
        findViewById<View>(R.id.tvAiModelYolo)?.setOnClickListener {
            showModelSelectionDialog(
                ModelAssetCatalog.Family.YOLO_SEG,
                runtimeConfig.yoloModelAsset
            ) { path ->
                runtimeConfig = runtimeConfig.copy(yoloModelAsset = path)
                ModelRuntimeConfig.save(this, runtimeConfig)
                renderAiModelControls()
                saveRuntimeConfigAndRecreateAi()
            }
        }
        findViewById<View>(R.id.tvAiModelOstrack)?.setOnClickListener {
            showModelSelectionDialog(
                ModelAssetCatalog.Family.OSTRACK,
                runtimeConfig.ostrackModelAsset
            ) { path ->
                runtimeConfig = runtimeConfig.copy(ostrackModelAsset = path)
                ModelRuntimeConfig.save(this, runtimeConfig)
                renderAiModelControls()
                saveRuntimeConfigAndRecreateAi()
            }
        }
        findViewById<View>(R.id.tvAiModelXFeat)?.setOnClickListener {
            showModelSelectionDialog(
                ModelAssetCatalog.Family.XFEAT,
                runtimeConfig.xfeatModelAsset
            ) { path ->
                runtimeConfig = runtimeConfig.copy(xfeatModelAsset = path)
                ModelRuntimeConfig.save(this, runtimeConfig)
                renderAiModelControls()
                saveRuntimeConfigAndRecreateAi()
            }
        }

        // 🌟 新增 XFeat 生死开关绑定
        findViewById<View>(R.id.tvToggleXFeatEnabled)?.setOnClickListener {
            val newState = !runtimeConfig.xfeatEnabled
            runtimeConfig = runtimeConfig.copy(xfeatEnabled = newState)
            ModelRuntimeConfig.save(this, runtimeConfig)

            // 实时更新 AppState，让 pipeline 不再跑 XFeat
            state.xfeatEnabled = newState

            renderAiModelControls()
            saveRuntimeConfigAndRecreateAi()
            toast(if (newState) "XFeat 已开启 (双重核验模式)" else "XFeat 已关闭 (纯 DINOv2 极速模式)")
        }

        // 🌟 新增 DINOv2 的选择菜单 🌟
        findViewById<View>(R.id.tvAiModelDinov2)?.setOnClickListener {
            showModelSelectionDialog(
                ModelAssetCatalog.Family.DINOV2,
                runtimeConfig.dinov2ModelAsset
            ) { path ->
                runtimeConfig = runtimeConfig.copy(dinov2ModelAsset = path)
                ModelRuntimeConfig.save(this, runtimeConfig)
                renderAiModelControls()
                saveRuntimeConfigAndRecreateAi()
            }
        }

        findViewById<View>(R.id.tvAiTuning)?.setOnClickListener { showRuntimeTuningDialog() }
        findViewById<View>(R.id.tvAiValidation)?.setOnClickListener {
            startActivity(android.content.Intent(this, ModelValidationActivity::class.java))
        }

        // --- Runtime 控制组 ---
        findViewById<View>(R.id.tvRuntimeYoloBackend)?.setOnClickListener {
            state.yoloBackend = ModelRuntimeConfig.cycleBackend(state.yoloBackend)
            renderRuntimeControls()
            saveRuntimeConfigAndRecreateAi()
        }
        findViewById<View>(R.id.tvRuntimeReport)?.setOnClickListener {
            state.reportEnabled = !state.reportEnabled
            renderRuntimeControls()
            runtimeConfig = runtimeConfig.copy(reportEnabled = state.reportEnabled)
            ModelRuntimeConfig.save(this, runtimeConfig)
        }

        // --- QNN 调度模式切换 ---
        findViewById<View>(R.id.tvToggleFace)?.setOnClickListener {
            val currentMode = com.example.aiobs.ai.TfliteAcceleration.qnnPerformanceMode
            val nextMode = when (currentMode) {
                com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.DEFAULT -> com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.BURST
                com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.BURST -> com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.SUSTAINED_HIGH_PERFORMANCE
                com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.SUSTAINED_HIGH_PERFORMANCE -> com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.POWER_SAVER
                com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.POWER_SAVER -> com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.DEFAULT
            }
            TfliteAcceleration.setQnnPerformanceMode(this, nextMode)
            renderRuntimeControls()
            saveRuntimeConfigAndRecreateAi()
            val toastMsg = when (nextMode) {
                com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.DEFAULT -> "已恢复默认调度"
                TfliteAcceleration.QnnPerformanceMode.BURST -> "已切换至: 爆发性能模式"
                TfliteAcceleration.QnnPerformanceMode.SUSTAINED_HIGH_PERFORMANCE -> "已切换至: 持续高性能模式"
                TfliteAcceleration.QnnPerformanceMode.POWER_SAVER -> "已切换至: 省电策略"
            }
            toast(toastMsg)
        }

        // --- Debug 与工具 ---
        findViewById<View>(R.id.tvDebugFreeze)?.setOnClickListener {
            val frozen = radarController.toggleAiFrozen()
            setRowValue(R.id.tvDebugFreeze, if (frozen) "FROZEN" else "RUNNING", frozen)
        }
        findViewById<View>(R.id.tvModelValidation)?.setOnClickListener { startActivity(android.content.Intent(this, ModelValidationActivity::class.java)) }

        // --- 推流触发 ---
        btnStream.setOnClickListener {
            if (streamController.isStreaming()) stopStreamingByUser() else startStreamingByUser()
        }

        // 初始化加载状态值
        renderAiModelControls()
        renderRuntimeControls()
        renderCameraControls()
    }

    private fun renderAiModelControls() {
        setRowValue(R.id.tvAiModelYolo, shortModelName(runtimeConfig.yoloModelAsset))
        setRowValue(R.id.tvAiModelOstrack, shortModelName(runtimeConfig.ostrackModelAsset))
        setRowValue(R.id.tvAiModelXFeat, shortModelName(runtimeConfig.xfeatModelAsset))
        setRowValue(R.id.tvAiModelDinov2, shortModelName(runtimeConfig.dinov2ModelAsset))
        // 🌟 显示开关状态
        setRowValue(R.id.tvToggleXFeatEnabled, if (runtimeConfig.xfeatEnabled) "ON" else "OFF", runtimeConfig.xfeatEnabled)
    }

    private fun renderRuntimeControls() {
        setRowValue(R.id.tvRuntimeYoloBackend, state.yoloBackend.name)
        setRowValue(R.id.tvRuntimeReport, if (state.reportEnabled) "ON" else "OFF", state.reportEnabled)
        val modeStr = when (com.example.aiobs.ai.TfliteAcceleration.qnnPerformanceMode) {
            com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.DEFAULT -> "Default Auto"
            com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.BURST -> "Burst"
            com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.SUSTAINED_HIGH_PERFORMANCE -> "Sustained High"
            com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.POWER_SAVER -> "Power Saver"
        }
        setRowValue(R.id.tvToggleFace, modeStr, com.example.aiobs.ai.TfliteAcceleration.qnnPerformanceMode != com.example.aiobs.ai.TfliteAcceleration.QnnPerformanceMode.DEFAULT)
    }

    private fun renderCameraControls() {
        setRowValue(R.id.tvToggleRes, resolutionShort(state.videoWidth, state.videoHeight))
        setRowValue(R.id.tvPerfFps, "${state.videoFps} FPS")
        setRowValue(R.id.tvToggleZoom, formatZoom(state.zoomLevel))
        setRowValue(R.id.tvToggleOri, if (manualOrientationLocked) if (state.portrait) "Portrait" else "Landscape" else "Auto")
        setRowValue(R.id.tvToggleLens, if (state.frontCamera) "Front" else "Rear")
    }

    private fun startStreamingByUser() {
        val host = etHost.text.toString().trim()
        val port = etPort.text.toString().trim()
        val stream = etStreamName.text.toString().trim()
        if (host.isBlank() || port.isBlank() || stream.isBlank()) { toast("推流地址信息不完整"); return }

        SrtProfileManager.saveProfile(this, host, port, stream)
        state.userStoppingStream = false

        if (!cameraController.isPreviewing()) {
            startCameraPipeline()
        }

        if (!streamController.start(SrtProfileManager.getFullUrl(this))) {
            state.userStoppingStream = true
        }
    }

    private fun stopStreamingByUser() {
        state.userStoppingStream = true
        streamController.stop(userInitiated = true)
        renderStreamButton(false, false)
    }

    override fun onStreamStateChanged(streaming: Boolean, reconnecting: Boolean) {
        runOnUiThread { renderStreamButton(streaming, reconnecting) }
    }

    override fun onBitrateChanged(bitrate: Long) = Unit

    override fun onStreamError(reason: String) {
        runOnUiThread {
            if (state.userStoppingStream) {
                renderStreamButton(false, false)
                return@runOnUiThread
            }
            if (!streamController.shouldAutoReconnect(reason)) {
                renderStreamButton(false, false)
                toast(reason.removePrefix("STREAM_INIT_ERROR:"))
                state.userStoppingStream = true
                return@runOnUiThread
            }
            streamController.stop(userInitiated = false)
            streamController.scheduleReconnect(urlProvider = { SrtProfileManager.getFullUrl(this@MainActivity) })
        }
    }

    private fun activeCameraCapabilities(): CameraCapabilities? = cameraController.getCameraCapabilities()

    private fun showCameraCapabilitiesDialog() {
        val caps = activeCameraCapabilities()
        if (caps == null) {
            toast("无法读取当前摄像头能力")
            return
        }

        val bottomSheet = com.google.android.material.bottomsheet.BottomSheetDialog(this)
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(android.graphics.Color.parseColor("#121212")) // 极客深黑背景
            setPadding(48, 48, 48, 48)
        }

        val titleView = TextView(this).apply {
            text = "Camera ${caps.cameraId} Capabilities"
            textSize = 20f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(android.graphics.Color.WHITE)
            setPadding(0, 0, 0, 32)
        }
        container.addView(titleView)

        fun addSettingRow(label: String, value: String, onClick: (() -> Unit)? = null) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, 36, 0, 36)
                gravity = android.view.Gravity.CENTER_VERTICAL

                addView(TextView(this@MainActivity).apply {
                    text = label
                    textSize = 15f
                    setTextColor(android.graphics.Color.parseColor("#888888"))
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                })

                addView(TextView(this@MainActivity).apply {
                    text = value
                    textSize = 15f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                    setTextColor(if (onClick != null) android.graphics.Color.parseColor("#7EC8FF") else android.graphics.Color.WHITE)
                })

                if (onClick != null) {
                    isClickable = true
                    val outValue = android.util.TypedValue()
                    theme.resolveAttribute(android.R.attr.selectableItemBackground, outValue, true)
                    setBackgroundResource(outValue.resourceId)
                    setOnClickListener { onClick() }
                }
            }
            container.addView(row)

            val divider = View(this).apply {
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 1).apply { setMargins(0, 0, 0, 0) }
                setBackgroundColor(android.graphics.Color.parseColor("#1AFFFFFF"))
            }
            container.addView(divider)
        }

        addSettingRow("镜头 (Lens)", "${CameraCapabilitiesReader.lensLabel(caps.lensFacing)} · ID ${caps.cameraId}")
        addSettingRow("分辨率 (Resolution)", "${state.videoWidth} × ${state.videoHeight}") { showResolutionDialog(); bottomSheet.dismiss() }
        addSettingRow("帧率 (FPS)", "${state.videoFps} FPS") { showFpsDialog(); bottomSheet.dismiss() }
        addSettingRow("数码变焦 (Zoom)", "${formatZoom(state.zoomLevel)} / ${formatZoom(caps.maxDigitalZoom)}") { showZoomDialog(); bottomSheet.dismiss() }
        addSettingRow("对焦模式 (Focus)", focusSummary(caps)) { showFocusDialog(); bottomSheet.dismiss() }

        if (caps.isoRange != null) {
            addSettingRow("感光度 (ISO)", cameraAdvanced.iso?.toString() ?: "自动 (Auto)") { showIsoDialog(caps); bottomSheet.dismiss() }
        }
        if (caps.exposureTimeRangeNs != null) {
            addSettingRow("快门速度 (Shutter)", formatExposureNs(cameraAdvanced.exposureTimeNs ?: 0L)) { showExposureDialog(caps); bottomSheet.dismiss() }
        }
        if (caps.exposureCompensationRange != null) {
            addSettingRow("曝光补偿 (EV)", "${cameraAdvanced.exposureCompensation ?: 0}") { showExposureCompensationDialog(caps); bottomSheet.dismiss() }
        }
        if (caps.awbModes.isNotEmpty()) {
            addSettingRow("白平衡 (AWB)", awbSummary(cameraAdvanced.awbMode)) { showAwbDialog(caps); bottomSheet.dismiss() }
        }

        val resetBtn = com.google.android.material.button.MaterialButton(this).apply {
            text = "恢复默认 (Reset)"
            setBackgroundColor(android.graphics.Color.parseColor("#222222"))
            setTextColor(android.graphics.Color.parseColor("#FF6B6B")) // 危险操作用红色预警
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { setMargins(0, 48, 0, 0) }
            setOnClickListener {
                resetCameraSettingsToDefaults()
                toast("相机参数已恢复默认")
                bottomSheet.dismiss()
            }
        }
        container.addView(resetBtn)

        val scrollView = android.widget.ScrollView(this).apply { addView(container) }
        bottomSheet.setContentView(scrollView)
        bottomSheet.show()
    }

    private fun showLensSelectionDialog() {
        if (streamController.isStreaming()) {
            toast("推流中无法切换镜头")
            return
        }

        val availableCameras = streamController.camera().camerasAvailable
        if (availableCameras.isNullOrEmpty()) {
            toast("未读取到可用镜头")
            return
        }

        val labels = availableCameras.map { id ->
            when (id) {
                "0" -> "ID 0 (主摄通道 Main)"
                "1" -> "ID 1 (前置通道 Front)"
                else -> "ID $id (附加物理/逻辑镜头)"
            }
        }.toTypedArray()

        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("选择物理镜头 (Camera ID)")
            .setItems(labels) { _, which ->
                val wantedId = availableCameras[which]
                val currentId = streamController.camera().currentCameraId

                if (currentId != wantedId) {
                    state.zoomLevel = 1f
                    radarController.stop(clearLock = false)

                    cameraController.stopPreview()
                    streamController.camera().startPreview(wantedId)

                    persistCameraSettings()
                    cameraController.rebuildPreview(
                        prepare = { streamController.preparePreview() },
                        afterRestart = {
                            syncOverlayGeometry()
                            renderCameraControls()
                            if (state.yoloEnabled) { try { radarController.start() } catch (_: Throwable) {} }
                        },
                        delayMs = 250L
                    )
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun showResolutionDialog() {
        if (streamController.isStreaming()) { toast("推流中无法切换画质"); return }
        val caps = activeCameraCapabilities() ?: run { toast("无法读取摄像头能力"); return }
        val sizes = caps.outputSizes
        if (sizes.isEmpty()) { toast("当前摄像头未返回可用分辨率"); return }
        val labels = sizes.map(CameraCapabilitiesReader::formatSize).toTypedArray()
        val selected = sizes.indexOfFirst { it.width == state.videoWidth && it.height == state.videoHeight }.coerceAtLeast(0)
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("分辨率 · 支持 ${sizes.size} 档")
            .setSingleChoiceItems(labels, selected) { dialog, which ->
                val size = sizes[which]
                state.videoWidth = size.width
                state.videoHeight = size.height
                state.videoBitrate = estimateBitrate(size.width, size.height)
                persistCameraSettings()
                renderCameraControls()
                rebuildPreviewAfterConfiguration()
                dialog.dismiss()
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun showFpsDialog() {
        val caps = activeCameraCapabilities() ?: run { toast("无法读取摄像头能力"); return }
        val ranges = caps.fpsRanges
        if (ranges.isEmpty()) { toast("当前摄像头未返回 FPS 能力"); return }
        val labels = ranges.map(CameraCapabilitiesReader::formatFps).toTypedArray()
        val selected = ranges.indexOfFirst { state.videoFps in it.lower..it.upper }.coerceAtLeast(0)
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("帧率 · Camera2 支持范围")
            .setSingleChoiceItems(labels, selected) { dialog, which ->
                state.videoFps = ranges[which].upper
                persistCameraSettings()
                renderCameraControls()
                cameraController.applySettings(cameraAdvanced)
                dialog.dismiss()
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun showZoomDialog() {
        val caps = activeCameraCapabilities() ?: run { toast("无法读取摄像头能力"); return }
        val maxZoom = caps.maxDigitalZoom.coerceAtLeast(1f)
        val content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(36, 12, 36, 8) }
        val value = TextView(this).apply { text = formatZoom(state.zoomLevel); textSize = 18f }
        val bar = SeekBar(this).apply {
            max = 1000
            progress = (((state.zoomLevel - 1f) / (maxZoom - 1f).coerceAtLeast(0.001f)) * max).toInt().coerceIn(0, max)
        }
        bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                val z = 1f + (maxZoom - 1f) * progress / seekBar.max.toFloat()
                state.zoomLevel = z.coerceIn(1f, maxZoom)
                value.text = formatZoom(state.zoomLevel)
                persistCameraSettings()
                renderCameraControls()
                cameraController.applySettings(cameraAdvanced)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })
        content.addView(value); content.addView(bar)
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("Zoom · 1×–${formatZoom(maxZoom)}")
            .setView(content)
            .setNegativeButton("关闭", null)
            .show()
    }

    private fun showFocusDialog() {
        val caps = activeCameraCapabilities() ?: return
        val labels = mutableListOf<String>(); val modes = mutableListOf<Int>()
        fun add(mode: Int, label: String) { if (caps.supportsAfMode(mode)) { modes += mode; labels += label } }
        add(android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO, "连续视频对焦")
        add(android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE, "连续拍照对焦")
        add(android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_AUTO, "自动对焦")
        add(android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_OFF, "手动 / 锁焦")
        if (labels.isEmpty()) { toast("当前摄像头没有可用 AF 模式"); return }
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("对焦模式 · 最小对焦距离 ${caps.minFocusDistance}")
            .setSingleChoiceItems(labels.toTypedArray(), -1) { dialog, which ->
                val mode = modes[which]
                cameraAdvanced = cameraAdvanced.copy(
                    afMode = mode,
                    focusDistance = if (mode == android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_OFF)
                        cameraAdvanced.focusDistance else null
                )
                state.proCameraEnabled = mode == android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_OFF ||
                        cameraAdvanced.iso != null || cameraAdvanced.exposureTimeNs != null
                persistCameraSettings()
                cameraController.applySettings(cameraAdvanced)
                dialog.dismiss()
                if (mode == android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_OFF && caps.supportsManualFocus()) showManualFocusDistanceDialog(caps)
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun showManualFocusDistanceDialog(caps: CameraCapabilities) {
        val maxD = caps.minFocusDistance.coerceAtLeast(0.01f)
        val content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(36, 8, 36, 8) }
        val value = TextView(this).apply { text = "${cameraAdvanced.focusDistance ?: 1f} D"; textSize = 16f }
        val bar = SeekBar(this).apply { max = 1000; progress = (((cameraAdvanced.focusDistance ?: 1f) / maxD) * max).toInt().coerceIn(0, max) }
        bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                val d = maxD * progress / seekBar.max.toFloat()
                cameraAdvanced = cameraAdvanced.copy(focusDistance = d)
                value.text = "${"%.2f".format(d)} D"
                persistCameraSettings()
                cameraController.applySettings(cameraAdvanced)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })
        content.addView(value); content.addView(bar)
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this).setTitle("手动对焦").setView(content).setNegativeButton("关闭", null).show()
    }

    private fun showIsoDialog(caps: CameraCapabilities) {
        val r = caps.isoRange ?: return
        val values = listOf(r.lower, r.upper, ((r.lower + r.upper) / 2), 100, 200, 400, 800, 1600, 3200, 6400)
            .filter { it in r.lower..r.upper }.distinct().sorted()
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this).setTitle("ISO · ${r.lower}–${r.upper}")
            .setItems(values.map { it.toString() }.toTypedArray()) { _, which ->
                cameraAdvanced = cameraAdvanced.copy(iso = values[which])
                state.proCameraEnabled = true
                persistCameraSettings()
                cameraController.applySettings(cameraAdvanced)
            }.setNegativeButton("自动") { _, _ ->
                cameraAdvanced = cameraAdvanced.copy(iso = null)
                state.proCameraEnabled = cameraAdvanced.exposureTimeNs != null ||
                        cameraAdvanced.afMode == android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_OFF
                persistCameraSettings()
                cameraController.applySettings(cameraAdvanced)
            }.show()
    }

    private fun showExposureDialog(caps: CameraCapabilities) {
        val r = caps.exposureTimeRangeNs ?: return
        val ONE_SEC = 1_000_000_000L
        val standardSpeeds = listOf(
            r.lower,
            ONE_SEC / 8000, ONE_SEC / 4000, ONE_SEC / 2000, ONE_SEC / 1000,
            ONE_SEC / 500,  ONE_SEC / 250,  ONE_SEC / 125,
            ONE_SEC / 100,  ONE_SEC / 90,   ONE_SEC / 60,
            ONE_SEC / 50,   ONE_SEC / 30,   ONE_SEC / 15,
            ONE_SEC / 8,    ONE_SEC / 4,    ONE_SEC / 2,
            r.upper
        )

        val values = standardSpeeds.filter { it in r.lower..r.upper }.distinct().sorted()

        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("曝光时间 (Shutter Speed)")
            .setItems(values.map(::formatExposureNs).toTypedArray()) { _, which ->
                cameraAdvanced = cameraAdvanced.copy(exposureTimeNs = values[which])
                state.proCameraEnabled = true
                persistCameraSettings()
                cameraController.applySettings(cameraAdvanced)
            }
            .setNegativeButton("自动 (Auto)") { _, _ ->
                cameraAdvanced = cameraAdvanced.copy(exposureTimeNs = null)
                state.proCameraEnabled = cameraAdvanced.iso != null || cameraAdvanced.afMode == android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_OFF
                persistCameraSettings()
                cameraController.applySettings(cameraAdvanced)
            }
            .show()
    }

    private fun showExposureCompensationDialog(caps: CameraCapabilities) {
        val r = caps.exposureCompensationRange ?: return
        val values = (r.lower..r.upper).toList()
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this).setTitle("曝光补偿 EV").setItems(values.map { if (it >= 0) "+$it" else it.toString() }.toTypedArray()) { _, which ->
            cameraAdvanced = cameraAdvanced.copy(exposureCompensation = values[which])
            persistCameraSettings()
            cameraController.applySettings(cameraAdvanced)
        }.setNegativeButton("0") { _, _ ->
            cameraAdvanced = cameraAdvanced.copy(exposureCompensation = 0)
            persistCameraSettings()
            cameraController.applySettings(cameraAdvanced)
        }.show()
    }

    private fun showAwbDialog(caps: CameraCapabilities) {
        val modes = caps.awbModes
        val labels = modes.map { awbLabel(it) }.toTypedArray()
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this).setTitle("白平衡").setItems(labels) { _, which ->
            cameraAdvanced = cameraAdvanced.copy(awbMode = modes[which])
            persistCameraSettings()
            cameraController.applySettings(cameraAdvanced)
        }.setNegativeButton("自动") { _, _ ->
            cameraAdvanced = cameraAdvanced.copy(awbMode = null)
            persistCameraSettings()
            cameraController.applySettings(cameraAdvanced)
        }.show()
    }

    private fun showOrientationDialog() {
        if (streamController.isStreaming()) {
            toast("推流中无法切换方向")
            return
        }
        val labels = arrayOf("自动旋转", "竖屏", "横屏")
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this).setTitle("画面方向").setItems(labels) { _, which ->
            when (which) {
                0 -> {
                    manualOrientationLocked = false
                    requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_FULL_SENSOR
                    state.portrait = resources.configuration.orientation == Configuration.ORIENTATION_PORTRAIT
                }
                1 -> {
                    manualOrientationLocked = true
                    state.portrait = true
                    requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
                }
                2 -> {
                    manualOrientationLocked = true
                    state.portrait = false
                    requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
                }
            }
            persistCameraSettings()
            renderCameraControls()
            if (::cameraController.isInitialized && cameraController.isPreviewing()) {
                syncOverlayGeometry()
                rebuildPreviewAfterConfiguration()
            }
        }.setNegativeButton("取消", null).show()
    }

    private fun focusSummary(caps: CameraCapabilities): String = when {
        cameraAdvanced.afMode == android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_OFF -> if (cameraAdvanced.focusDistance != null) "手动 ${cameraAdvanced.focusDistance}D" else "锁焦"
        cameraAdvanced.afMode == android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_AUTO -> "AF AUTO"
        cameraAdvanced.afMode == android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE -> "CONTINUOUS PICTURE"
        else -> "连续视频"
    }

    private fun awbSummary(mode: Int?): String = mode?.let(::awbLabel) ?: "自动"
    private fun awbLabel(mode: Int): String = when (mode) {
        android.hardware.camera2.CameraMetadata.CONTROL_AWB_MODE_AUTO -> "自动"
        android.hardware.camera2.CameraMetadata.CONTROL_AWB_MODE_DAYLIGHT -> "日光"
        android.hardware.camera2.CameraMetadata.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT -> "阴天"
        android.hardware.camera2.CameraMetadata.CONTROL_AWB_MODE_TWILIGHT -> "黄昏"
        android.hardware.camera2.CameraMetadata.CONTROL_AWB_MODE_FLUORESCENT -> "荧光"
        android.hardware.camera2.CameraMetadata.CONTROL_AWB_MODE_INCANDESCENT -> "白炽"
        else -> "模式 $mode"
    }
    private fun formatZoom(z: Float): String = if (z >= 10f) "%.1fX".format(z) else "%.2fX".format(z)
    private fun resolutionShort(w: Int, h: Int): String = when (w to h) { 1920 to 1080 -> "1080P"; 1280 to 720 -> "720P"; else -> "${w}×${h}" }
    private fun estimateBitrate(w: Int, h: Int): Int = ((w.toLong() * h.toLong() * 3L).coerceIn(900_000L, 8_000_000L)).toInt()
    private fun formatExposureNs(ns: Long): String = if (ns <= 0L) "自动" else if (ns >= 1_000_000_000L) "%.2fs".format(ns / 1_000_000_000.0) else "1/${(1_000_000_000.0 / ns).roundToInt().coerceAtLeast(1)}s"

    private fun renderStreamButton(streaming: Boolean, reconnecting: Boolean) {
        when {
            reconnecting -> { btnStream.text = "RECONNECTING..."; btnStream.setBackgroundColor(android.graphics.Color.parseColor("#FF9800")) }
            streaming -> { btnStream.text = "STOP STREAMING"; btnStream.setBackgroundColor(android.graphics.Color.parseColor("#4CAF50")) }
            else -> { btnStream.text = "START STREAMING"; btnStream.setBackgroundColor(android.graphics.Color.parseColor("#D32F2F")) }
        }
    }

    private fun renderStreamStatus() {
        val radarEnabled = state.yoloEnabled
        val ai = radarController.performanceSnapshot()
        val ui = radarController.uiStatusSnapshot()
        val device = deviceStatusController.snapshot()
        val camera = cameraController.cameraStatsSnapshot()

        val isRecovering = ui.state == com.example.aiobs.ai.TrackState.LOST && ui.recoveryActive
        val statusText = when {
            !radarEnabled -> "AI OFF"
            isRecovering -> "RECOVERY"
            ui.state == com.example.aiobs.ai.TrackState.LOST -> "LOST"
            radarController.getLockedTargetId() != null -> "LOCKED"
            ai.processedFps > 0.1f -> "SEARCHING"
            else -> "READY"
        }

        hudStatusText.text = statusText
        val statusColor = when (statusText) {
            "LOCKED" -> Color.parseColor("#55D68A")
            "RECOVERY" -> Color.parseColor("#FFB84D")
            "LOST" -> Color.parseColor("#FF6B6B")
            "SEARCHING" -> Color.parseColor("#FFD166")
            "AI OFF" -> Color.parseColor("#777777")
            else -> Color.parseColor("#69AFFF")
        }
        hudStatusDot.background?.setTint(statusColor)
        hudFpsText.text = "CAM ${"%.1f".format(camera.receivedFps)} | AI ${"%.1f".format(ai.processedFps)}"
        hudTempText.text = if (device.temperatureC > 0f) "${"%.1f".format(device.temperatureC)}°C" else "--°C"
        hudBatteryText.text = "${device.batteryPercent}%"
        hudYoloTime.text = if (ai.yoloTotalMs > 0) "${"%.1f".format(ai.yoloTotalMs)}ms" else "—"
        hudOsTrackTime.text = if (ai.ostrackMs > 0) "${"%.1f".format(ai.ostrackMs)}ms" else "—"
        hudDinov2Time.text = if (ai.dinov2Ms > 0) "${"%.1f".format(ai.dinov2Ms)}ms" else "—"
        val xfeatMs = ai.xfeatExtractMs + ai.xfeatMatchMs
        hudXFeatTime.text = if (xfeatMs > 0) "${"%.1f".format(xfeatMs)}ms" else "—"
        hudNetworkText.text = device.network ?: "NO NET"

        val showHud = state.reportEnabled
        val hudVisibility = if (showHud) View.VISIBLE else View.GONE

        findViewById<View>(R.id.hudStatusPill)?.visibility = hudVisibility
        findViewById<View>(R.id.hudDevicePill)?.visibility = hudVisibility
        findViewById<View>(R.id.hudFpsPill)?.visibility = hudVisibility
        findViewById<View>(R.id.hudPipelinePill)?.visibility = hudVisibility

        val streaming = streamController.isStreaming()
        if (streaming && showHud) {
            hudStreamPill.visibility = View.VISIBLE
            hudStreamInfoPill.visibility = View.VISIBLE

            val actualKbps = streamController.currentRealBitrate / 1024
            val targetKbps = state.videoBitrate / 1024
            hudStreamText.text = "LIVE ${actualKbps}/${targetKbps} Kbps"

            val resStr = resolutionShort(state.videoWidth, state.videoHeight)
            val strmFps = streamController.currentStreamFps
            hudStreamInfoText.text = "$resStr | STRM ${strmFps}FPS"
        } else {
            hudStreamPill.visibility = View.GONE
            hudStreamInfoPill.visibility = View.GONE
        }

        btnOpenDashboard.alpha = if (showHud) 1.0f else 0.3f
    }

    private fun hasAllRequiredPermissions(): Boolean {
        return requiredPermissions.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }
    }

    private fun requestPermissionsIfNeeded() {
        val missing = requiredPermissions.filter { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isNotEmpty()) ActivityCompat.requestPermissions(this, missing.toTypedArray(), PERMISSIONS_REQUEST_CODE)
    }

    private fun toast(message: String) { Toast.makeText(this, message, Toast.LENGTH_SHORT).show() }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSIONS_REQUEST_CODE && hasAllRequiredPermissions()) {
            startCameraPipeline()
        }
    }

    override fun onDestroy() {
        state.userStoppingStream = true
        radarController.stop(clearLock = true)
        radarController.destroy()
        cameraController.releaseAiFrameSource()
        streamController.stop(userInitiated = true)
        cameraController.stopPreview()
        statusHandler.removeCallbacksAndMessages(null)
        super.onDestroy()
    }

    private fun showYoloClassSelectionDialog(
        context: Context,
        current: Set<*>,
        onChanged: (Set<Int>) -> Unit
    ) {
        val selected = current.mapNotNull { (it as? Number)?.toInt() }.filter { it in 0 until YoloClassCatalog.COUNT }.toMutableSet()
        if (selected.isEmpty()) selected += YoloClassCatalog.DEFAULT_CLASS_IDS
        val checked = BooleanArray(YoloClassCatalog.COUNT) { it in selected }

        val dialog = com.google.android.material.dialog.MaterialAlertDialogBuilder(context)
            .setTitle("YOLO 识别类别（${YoloClassCatalog.COUNT} 类）")
            .setMultiChoiceItems(YoloClassCatalog.NAMES.toTypedArray(), checked) { _, which, isChecked ->
                if (isChecked) selected += which else selected -= which
            }
            .setNeutralButton("默认：person + car") { _, _ ->
                onChanged(YoloClassCatalog.DEFAULT_CLASS_IDS)
            }
            .setNegativeButton("取消", null)
            .setPositiveButton("确定", null)
            .create()

        dialog.setOnShowListener {
            dialog.getButton(android.app.AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                if (selected.isEmpty()) {
                    Toast.makeText(context, "至少选择一个 YOLO 类别", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
                onChanged(selected.toSet())
                dialog.dismiss()
            }
        }
        dialog.show()
    }

    private fun shortModelName(path: String): String = path.substringAfterLast('/').take(34)

    private fun currentAssetFor(
        family: ModelAssetCatalog.Family,
        yolo: String,
        ostrack: String,
        xfeat: String,
        dinov2: String
    ): String = when (family) {
        ModelAssetCatalog.Family.YOLO_SEG -> yolo
        ModelAssetCatalog.Family.OSTRACK -> ostrack
        ModelAssetCatalog.Family.XFEAT -> xfeat
        ModelAssetCatalog.Family.DINOV2 -> dinov2 // 🌟
        ModelAssetCatalog.Family.OTHER -> ""
    }

    private fun showModelSelectionDialog(
        family: ModelAssetCatalog.Family,
        initial: String,
        onSelected: (String) -> Unit
    ) {
        val models = ModelAssetCatalog.forFamily(this, family)
        if (models.isEmpty()) {
            Toast.makeText(this, "assets 中没有发现 ${ModelAssetCatalog.familyLabel(family)} 模型", Toast.LENGTH_SHORT).show()
            return
        }
        val selectedIndex = models.indexOfFirst { it.path == initial }.coerceAtLeast(0)
        val entries = models.map { model ->
            buildString {
                append(model.name)
                append("  ·  ")
                append(ModelAssetCatalog.quantizationLabel(model.quantization))
                append(" / ")
                append(ModelAssetCatalog.formatLabel(model.format))
                append(if (model.supportedByCurrentAdapter) "  ✓ 可用" else "  ⚠ 需适配")
            }
        }.toTypedArray()

        val dialog = com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("选择 ${ModelAssetCatalog.familyLabel(family)}")
            .setSingleChoiceItems(entries, selectedIndex) { d, which ->
                val model = models[which]
                if (!model.supportedByCurrentAdapter) {
                    Toast.makeText(this, model.supportNote, Toast.LENGTH_LONG).show()
                    return@setSingleChoiceItems
                }
                onSelected(model.path)
                d.dismiss()
            }
            .setNegativeButton("取消", null)
            .create()
        dialog.show()
    }

    private fun showRuntimeTuningDialog() {
        val bottomSheet = com.google.android.material.bottomsheet.BottomSheetDialog(this)

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(android.graphics.Color.parseColor("#121212")) // 极客深黑背景
            setPadding(48, 48, 48, 48)
        }

        content.addView(TextView(this).apply {
            text = "AI 参数 (Runtime Tuning)"
            textSize = 20f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(android.graphics.Color.WHITE)
            setPadding(0, 0, 0, 32)
        })

        var selectedYoloModel = runtimeConfig.yoloModelAsset
        var selectedOstrackModel = runtimeConfig.ostrackModelAsset
        var selectedXFeatModel = runtimeConfig.xfeatModelAsset
        var selectedDinov2Model = runtimeConfig.dinov2ModelAsset // 🌟

        fun createModernButton(btnText: String, onClick: () -> Unit): com.google.android.material.button.MaterialButton {
            return com.google.android.material.button.MaterialButton(this).apply {
                text = btnText
                setBackgroundColor(android.graphics.Color.parseColor("#2A2A2A")) // 高级深灰
                setTextColor(android.graphics.Color.parseColor("#7EC8FF"))       // 极客亮蓝
                elevation = 0f
                cornerRadius = (12 * resources.displayMetrics.density).toInt()
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(0, 12, 0, 32) }
                setOnClickListener { onClick() }
            }
        }

        fun addModelSelector(
            title: String,
            family: ModelAssetCatalog.Family,
            initial: String,
            onChanged: (String) -> Unit
        ) {
            val label = TextView(this).apply {
                textSize = 13f
                setTextColor(android.graphics.Color.parseColor("#888888"))
                setPadding(0, 10, 0, 8)
                text = "$title：${shortModelName(initial)}"
            }
            val button = createModernButton("选择 $title") {
                showModelSelectionDialog(family, initial = currentAssetFor(family, selectedYoloModel, selectedOstrackModel, selectedXFeatModel, selectedDinov2Model)) { path ->
                    onChanged(path)
                    label.text = "$title：${shortModelName(path)}"
                }
            }
            content.addView(label)
            content.addView(button)
        }

        addModelSelector("YOLO-Seg 模型", ModelAssetCatalog.Family.YOLO_SEG, selectedYoloModel) { selectedYoloModel = it }
        addModelSelector("OSTrack 模型", ModelAssetCatalog.Family.OSTRACK, selectedOstrackModel) { selectedOstrackModel = it }
        addModelSelector("XFeat 模型", ModelAssetCatalog.Family.XFEAT, selectedXFeatModel) { selectedXFeatModel = it }
        addModelSelector("DINOv2 模型", ModelAssetCatalog.Family.DINOV2, selectedDinov2Model) { selectedDinov2Model = it } // 🌟

        data class SliderSpec(
            val title: String,
            val minValue: Float,
            val maxValue: Float,
            val step: Float,
            val initial: Float,
            val formatter: (Float) -> String
        )
        val labels = mutableListOf<TextView>()
        fun addSlider(spec: SliderSpec) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(0, 20, 0, 20) }
                setBackgroundColor(android.graphics.Color.parseColor("#1A1A1A"))
                setPadding(28, 20, 28, 20)
            }

            val headerRow = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
            }

            val title = TextView(this).apply {
                text = spec.title
                textSize = 14f
                setTextColor(android.graphics.Color.WHITE)
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }

            val valIndicator = TextView(this).apply {
                text = spec.formatter(spec.initial)
                textSize = 15f
                typeface = android.graphics.Typeface.MONOSPACE
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setTextColor(android.graphics.Color.parseColor("#7EC8FF"))
            }

            headerRow.addView(title)
            headerRow.addView(valIndicator)
            row.addView(headerRow)

            val seek = SeekBar(this).apply {
                setPadding(16, 24, 16, 24)
            }

            val count = ((spec.maxValue - spec.minValue) / spec.step).roundToInt().coerceAtLeast(1)
            seek.max = count
            seek.progress = ((spec.initial - spec.minValue) / spec.step).roundToInt().coerceIn(0, count)

            seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    val value = (spec.minValue + progress * spec.step).coerceIn(spec.minValue, spec.maxValue)
                    valIndicator.text = spec.formatter(value)
                }
                override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
            })

            title.tag = seek
            labels += title

            row.addView(seek)
            content.addView(row)
        }

        fun f2(v: Float) = String.format(Locale.US, "%.2f", v)

        val classLabel = TextView(this).apply {
            text = "YOLO 识别类别  ${runtimeTuning.yoloSelectedClassIds.size}/${YoloClassCatalog.COUNT}"
            textSize = 13f
            setTextColor(android.graphics.Color.parseColor("#888888"))
            setPadding(0, 12, 0, 8)
        }
        content.addView(classLabel)

        val classButton = createModernButton("选择类别：${YoloClassCatalog.formatSelected(runtimeTuning.yoloSelectedClassIds)}") { }
        classButton.tag = runtimeTuning.yoloSelectedClassIds
        classButton.setOnClickListener {
            showYoloClassSelectionDialog(this@MainActivity, classButton.tag as? Set<*> ?: YoloClassCatalog.DEFAULT_CLASS_IDS) { selected ->
                classButton.tag = selected
                classLabel.text = "YOLO 识别类别  ${selected.size}/${YoloClassCatalog.COUNT}"
                classButton.text = "选择类别：${YoloClassCatalog.formatSelected(selected)}"
            }
        }
        content.addView(classButton)

        addSlider(SliderSpec("YOLO 置信阈值", 0.05f, 0.95f, 0.01f, runtimeTuning.yoloConfidenceThreshold, ::f2))
        addSlider(SliderSpec("OSTrack 最低有效分数", 0.20f, 0.95f, 0.01f, runtimeTuning.ostrackMinScore, ::f2))
        addSlider(SliderSpec("XFeat Identity Threshold", 0.05f, 0.95f, 0.01f, runtimeTuning.xfeatIdentityThreshold, ::f2))
        addSlider(SliderSpec("LOCKED YOLO 刷新周期", 1f, 60f, 1f, runtimeTuning.lockedYoloRefreshIntervalMs / 1000f, { String.format(Locale.US, "%.0fs", it) }))
        addSlider(SliderSpec("首次刷新延迟", 0f, 10f, 0.5f, runtimeTuning.lockedYoloFirstRefreshDelayMs / 1000f, { String.format(Locale.US, "%.1fs", it) }))

        val actionRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                setMargins(0, 48, 0, 0)
            }

            addView(com.google.android.material.button.MaterialButton(this@MainActivity).apply {
                text = "恢复默认"
                setBackgroundColor(android.graphics.Color.parseColor("#222222"))
                setTextColor(android.graphics.Color.parseColor("#FF6B6B"))
                elevation = 0f
                cornerRadius = (12 * resources.displayMetrics.density).toInt()
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { setMargins(0, 0, 16, 0) }

                setOnClickListener {
                    runtimeTuning = RuntimeTuningConfig.defaults()
                    selectedYoloModel = ModelRuntimeConfig.DEFAULT_YOLO_ASSET
                    selectedOstrackModel = ModelRuntimeConfig.DEFAULT_OSTRACK_ASSET
                    selectedXFeatModel = ModelRuntimeConfig.DEFAULT_XFEAT_ASSET
                    selectedDinov2Model = ModelRuntimeConfig.DEFAULT_DINOV2_ASSET // 🌟

                    classButton.tag = runtimeTuning.yoloSelectedClassIds
                    classLabel.text = "YOLO 识别类别  ${runtimeTuning.yoloSelectedClassIds.size}/${YoloClassCatalog.COUNT}"
                    classButton.text = "选择类别：${YoloClassCatalog.formatSelected(runtimeTuning.yoloSelectedClassIds)}"
                    RuntimeTuningConfig.save(this@MainActivity, runtimeTuning)

                    runtimeConfig = runtimeConfig.copy(
                        yoloModelAsset = ModelRuntimeConfig.DEFAULT_YOLO_ASSET,
                        ostrackModelAsset = ModelRuntimeConfig.DEFAULT_OSTRACK_ASSET,
                        xfeatModelAsset = ModelRuntimeConfig.DEFAULT_XFEAT_ASSET,
                        dinov2ModelAsset = ModelRuntimeConfig.DEFAULT_DINOV2_ASSET // 🌟
                    )
                    ModelRuntimeConfig.save(this@MainActivity, runtimeConfig)
                    saveRuntimeConfigAndRecreateAi()
                    renderStreamStatus()
                    toast("AI 参数已恢复默认")
                    bottomSheet.dismiss()
                }
            })

            addView(com.google.android.material.button.MaterialButton(this@MainActivity).apply {
                text = "确定应用"
                setBackgroundColor(android.graphics.Color.parseColor("#0057FF"))
                setTextColor(android.graphics.Color.WHITE)
                elevation = 0f
                cornerRadius = (12 * resources.displayMetrics.density).toInt()
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { setMargins(16, 0, 0, 0) }

                setOnClickListener {
                    fun valueAt(index: Int): Float {
                        val title = labels[index]
                        val seek = title.tag as SeekBar
                        return when (index) {
                            0 -> 0.05f + seek.progress * 0.01f
                            1 -> 0.20f + seek.progress * 0.01f
                            2 -> 0.05f + seek.progress * 0.01f
                            3 -> 1f + seek.progress * 1f
                            4 -> seek.progress * 0.5f
                            else -> 0f
                        }
                    }
                    val selectedClasses = (classButton.tag as? Set<*>)
                        ?.mapNotNull { (it as? Number)?.toInt() }
                        ?.filter { it in 0 until YoloClassCatalog.COUNT }
                        ?.toSet()
                        ?.ifEmpty { YoloClassCatalog.DEFAULT_CLASS_IDS }
                        ?: YoloClassCatalog.DEFAULT_CLASS_IDS

                    runtimeConfig = runtimeConfig.copy(
                        yoloModelAsset = selectedYoloModel,
                        ostrackModelAsset = selectedOstrackModel,
                        xfeatModelAsset = selectedXFeatModel,
                        dinov2ModelAsset = selectedDinov2Model // 🌟 应用最新的 DINOv2 模型
                    )

                    runtimeTuning = RuntimeTuningConfig(
                        yoloConfidenceThreshold = valueAt(0),
                        yoloSelectedClassIds = selectedClasses,
                        ostrackMinScore = valueAt(1),
                        xfeatIdentityThreshold = valueAt(2),
                        lockedYoloRefreshIntervalMs = (valueAt(3) * 1000L).toLong(),
                        lockedYoloFirstRefreshDelayMs = (valueAt(4) * 1000L).toLong()
                    ).normalized()

                    RuntimeTuningConfig.save(this@MainActivity, runtimeTuning)
                    ModelRuntimeConfig.save(this@MainActivity, runtimeConfig)
                    saveRuntimeConfigAndRecreateAi()
                    renderStreamStatus()
                    toast("AI 参数已应用")
                    bottomSheet.dismiss()
                }
            })
        }
        content.addView(actionRow)

        val scroll = android.widget.ScrollView(this).apply { addView(content) }
        bottomSheet.setContentView(scroll)
        bottomSheet.show()
    }

    companion object {
        private const val TAG = "MainActivityV22"
        private const val PERMISSIONS_REQUEST_CODE = 1001
        private const val STATUS_INTERVAL_MS = 1000L
        private const val KEY_MANUAL_ORIENTATION = "manual_orientation_locked"
    }
}