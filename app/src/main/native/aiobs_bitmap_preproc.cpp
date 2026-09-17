#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace {
constexpr char kTag[] = "NativeBitmapPreproc";
inline uint8_t toGray(const uint8_t* p) {
    const int r = static_cast<int>(p[0]);
    const int g = static_cast<int>(p[1]);
    const int b = static_cast<int>(p[2]);
    const float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    return static_cast<uint8_t>(std::max(0, std::min(255, static_cast<int>(gray))));
}
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_aiobs_ai_NativeBitmapPreprocessor_nativeExtractGrayPatch(
        JNIEnv* env,
        jclass,
        jobject bitmap,
        jint rectLeft,
        jint rectTop,
        jint rectWidth,
        jint rectHeight,
        jbyteArray maskPixels,
        jint maskWidth,
        jint maskHeight,
        jfloat rawLeft,
        jfloat rawTop,
        jfloat rawRight,
        jfloat rawBottom,
        jbyteArray output) {
    if (!bitmap || !output || rectWidth <= 0 || rectHeight <= 0) return JNI_FALSE;
    const jsize outLen = env->GetArrayLength(output);
    if (outLen < rectWidth * rectHeight) return JNI_FALSE;

    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return JNI_FALSE;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return JNI_FALSE;
    if (rectLeft < 0 || rectTop < 0 || rectLeft + rectWidth > static_cast<jint>(info.width) || rectTop + rectHeight > static_cast<jint>(info.height)) return JNI_FALSE;

    jboolean maskCopy = JNI_FALSE;
    jbyte* mask = nullptr;
    const bool hasMask = maskPixels != nullptr && maskWidth > 0 && maskHeight > 0 && env->GetArrayLength(maskPixels) >= maskWidth * maskHeight;
    if (hasMask) {
        mask = env->GetByteArrayElements(maskPixels, &maskCopy);
        if (!mask) return JNI_FALSE;
    }

    jboolean outCopy = JNI_FALSE;
    jbyte* out = env->GetByteArrayElements(output, &outCopy);
    if (!out) {
        if (mask) env->ReleaseByteArrayElements(maskPixels, mask, JNI_ABORT);
        return JNI_FALSE;
    }

    void* pixelsVoid = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixelsVoid) != ANDROID_BITMAP_RESULT_SUCCESS || !pixelsVoid) {
        env->ReleaseByteArrayElements(output, out, 0);
        if (mask) env->ReleaseByteArrayElements(maskPixels, mask, JNI_ABORT);
        return JNI_FALSE;
    }

    const auto* pixels = reinterpret_cast<const uint8_t*>(pixelsVoid);
    const size_t strideBytes = static_cast<size_t>(info.stride);
    const float rawW = std::max(1e-6f, rawRight - rawLeft);
    const float rawH = std::max(1e-6f, rawBottom - rawTop);
    const bool maskUsable = hasMask && rawW > 0.0f && rawH > 0.0f;

    for (int y = 0; y < rectHeight; ++y) {
        const int srcY = rectTop + y;
        const float sourceY = static_cast<float>(srcY) + 0.5f;
        int my = -1;
        if (maskUsable && sourceY >= rawTop && sourceY < rawBottom) {
            my = std::clamp(static_cast<int>(((sourceY - rawTop) / rawH) * maskHeight), 0, maskHeight - 1);
        }
        for (int x = 0; x < rectWidth; ++x) {
            const int srcX = rectLeft + x;
            const float sourceX = static_cast<float>(srcX) + 0.5f;
            bool keep = true;
            if (maskUsable) {
                if (my < 0 || sourceX < rawLeft || sourceX >= rawRight) {
                    keep = false;
                } else {
                    const int mx = std::clamp(static_cast<int>(((sourceX - rawLeft) / rawW) * maskWidth), 0, maskWidth - 1);
                    keep = (static_cast<uint8_t>(mask[my * maskWidth + mx]) != 0);
                }
            }
            out[y * rectWidth + x] = static_cast<jbyte>(keep ? toGray(pixels + static_cast<size_t>(srcY) * strideBytes + static_cast<size_t>(srcX) * 4u) : 0);
        }
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    env->ReleaseByteArrayElements(output, out, 0);
    if (mask) env->ReleaseByteArrayElements(maskPixels, mask, JNI_ABORT);
    return JNI_TRUE;
}

