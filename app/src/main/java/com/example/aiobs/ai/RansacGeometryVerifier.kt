package com.example.aiobs.ai

import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point

/** Homography/RANSAC verification for local-feature matches. */
object RansacGeometryVerifier {
    data class Result(
        val passed: Boolean,
        val inliers: Int,
        val totalMatches: Int,
        val inlierRatio: Float,
        val inlierMask: BooleanArray
    )

    fun verify(
        srcPoints: FloatArray,
        dstPoints: FloatArray,
        srcWidth: Int,
        srcHeight: Int,
        reprojectionError: Double = 5.0
    ): Result {
        val total = srcPoints.size / 2
        if (total < 4 || dstPoints.size < 8 || dstPoints.size != srcPoints.size ||
            srcPoints.any { !it.isFinite() } || dstPoints.any { !it.isFinite() }) {
            return Result(false, 0, total, 0f, BooleanArray(0))
        }

        val src = MatOfPoint2f()
        val dst = MatOfPoint2f()
        val mask = Mat()
        var h: Mat? = null
        return try {
            val srcArray = Array(total) { i -> Point(srcPoints[i * 2].toDouble(), srcPoints[i * 2 + 1].toDouble()) }
            val dstArray = Array(total) { i -> Point(dstPoints[i * 2].toDouble(), dstPoints[i * 2 + 1].toDouble()) }
            src.fromArray(*srcArray)
            dst.fromArray(*dstArray)

            h = Calib3d.findHomography(
                src,
                dst,
                Calib3d.RANSAC,
                reprojectionError,
                mask,
                2000,
                0.995
            )
            val homography = h ?: return Result(false, 0, total, 0f, BooleanArray(0))
            if (homography.empty() || mask.empty()) {
                return Result(false, 0, total, 0f, BooleanArray(0))
            }

            val maskData = ByteArray((mask.total() * mask.channels()).toInt())
            mask.get(0, 0, maskData)
            var inliers = 0
            for (v in maskData) if (v.toInt() != 0) inliers++
            val ratio = inliers.toFloat() / total.coerceAtLeast(1)

            val hData = DoubleArray((homography.total() * homography.channels()).toInt())
            homography.get(0, 0, hData)
            val homographyFinite = hData.all { it.isFinite() }
            val inlierMask = BooleanArray(total) { i -> maskData.getOrNull(i)?.toInt() != 0 }
            // A homography requires at least four geometrically consistent correspondences.
            // This is mathematical validity, not a tunable business confidence gate.
            val mathematicallyValid =
                homography.rows() == 3 &&
                        homography.cols() == 3 &&
                        homographyFinite &&
                        inliers >= 4
            Result(mathematicallyValid, inliers, total, ratio, inlierMask)
        } catch (_: Throwable) {
            Result(false, 0, total, 0f, BooleanArray(0))
        } finally {
            try { mask.release() } catch (_: Throwable) {}
            try { h?.release() } catch (_: Throwable) {}
            try { src.release() } catch (_: Throwable) {}
            try { dst.release() } catch (_: Throwable) {}
        }
    }

}
