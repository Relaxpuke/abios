package com.example.aiobs

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.DashPathEffect
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import com.example.aiobs.ai.TrackState
import com.example.aiobs.ai.TrackedObject

class AiOverlayView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private val defaultBoxPaint = Paint().apply { color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 6f }
    private val defaultTextPaint = Paint().apply { color = Color.WHITE; textSize = 45f; isFakeBoldText = true; setShadowLayer(8f, 0f, 0f, Color.BLACK) }
    private val dimBoxPaint = Paint().apply { color = Color.parseColor("#4400FFFF"); style = Paint.Style.STROKE; strokeWidth = 3f }
    private val dimTextPaint = Paint().apply { color = Color.parseColor("#88FFFFFF"); textSize = 35f; setShadowLayer(4f, 0f, 0f, Color.BLACK) }
    private val lockedBoxPaint = Paint().apply { color = Color.parseColor("#FF3333"); style = Paint.Style.STROKE; strokeWidth = 8f }
    private val lockedTextPaint = Paint().apply { color = Color.parseColor("#FF3333"); textSize = 50f; isFakeBoldText = true; setShadowLayer(8f, 0f, 0f, Color.WHITE) }
    private val lostLockedPaint = Paint(lockedBoxPaint).apply { pathEffect = DashPathEffect(floatArrayOf(24f, 18f), 0f); strokeWidth = 6f }

    private var trackedObjects: List<TrackedObject> = emptyList()

    /**
     * Visual-only smoothing state. The AI pipeline always keeps the original
     * TrackedObject coordinates; these values are used only when drawing/touching
     * the overlay. Coordinates are normalized to the source image (0..1).
     */
    private data class DisplayBox(
        var cx: Float,
        var cy: Float,
        var w: Float,
        var h: Float
    )

    private val displayBoxes = mutableMapOf<Int, DisplayBox>()

    private companion object {
        private const val POSITION_DEADBAND = 0.0012f
        private const val SIZE_DEADBAND = 0.006f
        private const val EMA_ALPHA_SLOW = 0.22f
        private const val EMA_ALPHA_NORMAL = 0.38f
        private const val EMA_ALPHA_FAST = 0.78f
        private const val FAST_MOTION_THRESHOLD = 0.03f
        private const val SNAP_THRESHOLD = 0.12f
    }
    private var lockedTargetId: Int? = null
    private var lockListener: ((Int) -> Unit)? = null
    private var unlockListener: (() -> Unit)? = null

    private var sourceWidth = 1920
    private var sourceHeight = 1080

    // Coordinates emitted by AI may follow physical device rotation even when the
    // Activity/preview is locked. These rotations let the overlay map AI-space boxes
    // back into the currently displayed preview-space without changing model outputs.
    @Volatile private var aiRotationDegrees = 0
    @Volatile private var previewRotationDegrees = 0

    fun setCoordinateRotations(aiRotation: Int, previewRotation: Int) {
        aiRotationDegrees = ((aiRotation % 360) + 360) % 360
        previewRotationDegrees = ((previewRotation % 360) + 360) % 360
        invalidate()
    }

    fun setSourceGeometry(width: Int, height: Int) {
        sourceWidth = width.coerceAtLeast(1)
        sourceHeight = height.coerceAtLeast(1)
        invalidate()
    }

    fun setLockCallbacks(onLockToggle: (Int) -> Unit, onClearLock: () -> Unit) {
        lockListener = onLockToggle
        unlockListener = onClearLock
    }

    fun setLockedTargetId(id: Int?) { lockedTargetId = id; invalidate() }

    fun updateTargets(targets: List<TrackedObject>) {
        trackedObjects = targets
        updateDisplayBoxes(targets)
        invalidate()
    }

    fun clearTargets() {
        trackedObjects = emptyList()
        displayBoxes.clear()
        invalidate()
    }

    private fun contentRect(): RectF {
        val vw = width.toFloat().coerceAtLeast(1f)
        val vh = height.toFloat().coerceAtLeast(1f)
        val sw = sourceWidth.toFloat().coerceAtLeast(1f)
        val sh = sourceHeight.toFloat().coerceAtLeast(1f)

        // 🌟 修复：使用 minOf (Fit Center) 完美对齐相机底层的 AspectRatioMode.ADJUST
        val scale = minOf(vw / sw, vh / sh)
        val contentW = sw * scale
        val contentH = sh * scale
        val left = (vw - contentW) / 2f
        val top = (vh - contentH) / 2f
        return RectF(left, top, left + contentW, top + contentH)
    }

    private fun boxForTarget(target: TrackedObject): RectF {
        val r = contentRect()
        val display = displayBoxes[target.id]
        val cxNorm = (display?.cx ?: target.cx).coerceIn(0f, 1f)
        val cyNorm = (display?.cy ?: target.cy).coerceIn(0f, 1f)
        val wNorm = (display?.w ?: target.w).coerceIn(0f, 1f)
        val hNorm = (display?.h ?: target.h).coerceIn(0f, 1f)

        val left = (cxNorm - wNorm * 0.5f).coerceIn(0f, 1f)
        val top = (cyNorm - hNorm * 0.5f).coerceIn(0f, 1f)
        val right = (cxNorm + wNorm * 0.5f).coerceIn(0f, 1f)
        val bottom = (cyNorm + hNorm * 0.5f).coerceIn(0f, 1f)

        val mapped = mapBoxBetweenRotations(RectF(left, top, right, bottom))
        val cx = r.left + mapped.centerX() * r.width()
        val cy = r.top + mapped.centerY() * r.height()
        val bw = mapped.width() * r.width()
        val bh = mapped.height() * r.height()
        return RectF(cx - bw / 2f, cy - bh / 2f, cx + bw / 2f, cy + bh / 2f)
    }

    private fun rotateNormPoint(x: Float, y: Float, clockwiseDegrees: Int): Pair<Float, Float> = when ((clockwiseDegrees + 360) % 360) {
        90 -> (1f - y) to x
        180 -> (1f - x) to (1f - y)
        270 -> y to (1f - x)
        else -> x to y
    }

    private fun mapBoxBetweenRotations(box: RectF): RectF {
        val delta = ((previewRotationDegrees - aiRotationDegrees) % 360 + 360) % 360
        if (delta == 0) return box
        val points = arrayOf(
            rotateNormPoint(box.left, box.top, delta),
            rotateNormPoint(box.right, box.top, delta),
            rotateNormPoint(box.right, box.bottom, delta),
            rotateNormPoint(box.left, box.bottom, delta)
        )
        val minX = points.minOf { it.first }.coerceIn(0f, 1f)
        val minY = points.minOf { it.second }.coerceIn(0f, 1f)
        val maxX = points.maxOf { it.first }.coerceIn(0f, 1f)
        val maxY = points.maxOf { it.second }.coerceIn(0f, 1f)
        return RectF(minX, minY, maxX, maxY)
    }

    private fun updateDisplayBoxes(targets: List<TrackedObject>) {
        val liveIds = targets.asSequence().map { it.id }.toSet()
        displayBoxes.keys.retainAll(liveIds)

        targets.forEach { target ->
            val existing = displayBoxes[target.id]
            if (existing == null) {
                displayBoxes[target.id] = DisplayBox(target.cx, target.cy, target.w, target.h)
                return@forEach
            }

            val dx = target.cx - existing.cx
            val dy = target.cy - existing.cy
            val dw = target.w - existing.w
            val dh = target.h - existing.h
            val centerDelta = kotlin.math.hypot(dx.toDouble(), dy.toDouble()).toFloat()
            val sizeDelta = maxOf(kotlin.math.abs(dw), kotlin.math.abs(dh))

            // Large movement is intentional motion or a new/recovered location;
            // snapping avoids visible lag while small changes are smoothed.
            if (centerDelta >= SNAP_THRESHOLD || sizeDelta >= 0.20f) {
                existing.cx = target.cx
                existing.cy = target.cy
                existing.w = target.w
                existing.h = target.h
                return@forEach
            }

            val motion = centerDelta + sizeDelta * 0.35f
            val alpha = when {
                motion >= FAST_MOTION_THRESHOLD -> EMA_ALPHA_FAST
                motion <= POSITION_DEADBAND && sizeDelta <= SIZE_DEADBAND -> EMA_ALPHA_SLOW
                else -> EMA_ALPHA_NORMAL
            }

            existing.cx = if (kotlin.math.abs(dx) <= POSITION_DEADBAND) existing.cx
            else existing.cx + dx * alpha
            existing.cy = if (kotlin.math.abs(dy) <= POSITION_DEADBAND) existing.cy
            else existing.cy + dy * alpha
            existing.w = if (kotlin.math.abs(dw) <= SIZE_DEADBAND) existing.w
            else existing.w + dw * alpha
            existing.h = if (kotlin.math.abs(dh) <= SIZE_DEADBAND) existing.h
            else existing.h + dh * alpha
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_DOWN) return super.onTouchEvent(event)
        val clicked = trackedObjects
            .asSequence()
            .filter { it.state != TrackState.LOST || it.id == lockedTargetId }
            .mapNotNull { target ->
                val box = boxForTarget(target)
                if (box.contains(event.x, event.y)) target else null
            }
            .minByOrNull { target ->
                val box = boxForTarget(target)
                val dx = box.centerX() - event.x
                val dy = box.centerY() - event.y
                dx * dx + dy * dy
            }
        if (clicked != null) lockListener?.invoke(clicked.id) else unlockListener?.invoke()
        performClick()
        return true
    }

    override fun performClick(): Boolean = super.performClick()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val anyLocked = lockedTargetId != null
        trackedObjects.forEach { target ->
            val box = boxForTarget(target)
            val centerX = box.centerX()
            val centerY = box.centerY()
            val isLocked = target.id == lockedTargetId

            val boxPaint: Paint
            val textPaint: Paint
            val label: String

            when {
                isLocked && target.state == TrackState.LOST -> {
                    boxPaint = lostLockedPaint
                    textPaint = lockedTextPaint
                    label = if (target.recoveryConfirmCount > 0) {
                        "RECOVERING ${target.recoveryConfirmCount}/2 [${target.label.uppercase()}] ID:${target.id}"
                    } else {
                        "LOST / RECOVERY [${target.label.uppercase()}] ID:${target.id}"
                    }
                }
                isLocked -> {
                    boxPaint = lockedBoxPaint
                    textPaint = lockedTextPaint
                    label = "LOCKED [${target.label.uppercase()}] ID:${target.id}"
                }
                anyLocked -> {
                    boxPaint = dimBoxPaint
                    textPaint = dimTextPaint
                    label = "${target.label} ID:${target.id}"
                }
                else -> {
                    boxPaint = defaultBoxPaint
                    textPaint = defaultTextPaint
                    label = "${target.label} ID:${target.id}"
                }
            }

            canvas.drawRect(box, boxPaint)
            canvas.drawText(label, box.left, (box.top - 15f).coerceAtLeast(45f), textPaint)
            if (isLocked) {
                val crossSize = 30f
                canvas.drawLine(centerX - crossSize, centerY, centerX + crossSize, centerY, lockedBoxPaint)
                canvas.drawLine(centerX, centerY - crossSize, centerX, centerY + crossSize, lockedBoxPaint)
            }
        }
    }
}