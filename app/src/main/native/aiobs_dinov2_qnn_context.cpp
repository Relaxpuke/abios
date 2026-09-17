#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace {

    constexpr char kTag[] = "Dinov2QnnCtx";

    void logi(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        __android_log_vprint(ANDROID_LOG_INFO, kTag, fmt, args);
        va_end(args);
    }

    void loge(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        __android_log_vprint(ANDROID_LOG_ERROR, kTag, fmt, args);
        va_end(args);
    }

    std::string dlError() {
        const char* e = dlerror();
        return e ? std::string(e) : std::string("unknown dlerror");
    }

    // =========================================================================
    // 基础结构体与辅助函数 (完全拷贝自 ostrack/xfeat，解决编译报错)
    // =========================================================================

    using QnnInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(
            const QnnInterface_t*** providerList,
            uint32_t* numProviders);

    using QnnSystemInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(
            const QnnSystemInterface_t*** providerList,
            uint32_t* numProviders);

    const QnnInterface_t* chooseQnnProvider(
            const QnnInterface_t** providers,
            uint32_t count) {
        const QnnInterface_t* best = nullptr;
        for (uint32_t i = 0; i < count; ++i) {
            const auto* p = providers[i];
            if (!p) continue;
            if (!best ||
                p->apiVersion.coreApiVersion.major > best->apiVersion.coreApiVersion.major ||
                (p->apiVersion.coreApiVersion.major == best->apiVersion.coreApiVersion.major &&
                 p->apiVersion.coreApiVersion.minor > best->apiVersion.coreApiVersion.minor) ||
                (p->apiVersion.coreApiVersion.major == best->apiVersion.coreApiVersion.major &&
                 p->apiVersion.coreApiVersion.minor == best->apiVersion.coreApiVersion.minor &&
                 p->apiVersion.coreApiVersion.patch > best->apiVersion.coreApiVersion.patch)) {
                best = p;
            }
        }
        return best;
    }

    const QnnSystemInterface_t* chooseSystemProvider(
            const QnnSystemInterface_t** providers,
            uint32_t count) {
        const QnnSystemInterface_t* best = nullptr;
        for (uint32_t i = 0; i < count; ++i) {
            const auto* p = providers[i];
            if (!p) continue;
            if (!best ||
                p->systemApiVersion.major > best->systemApiVersion.major ||
                (p->systemApiVersion.major == best->systemApiVersion.major &&
                 p->systemApiVersion.minor > best->systemApiVersion.minor) ||
                (p->systemApiVersion.major == best->systemApiVersion.major &&
                 p->systemApiVersion.minor == best->systemApiVersion.minor &&
                 p->systemApiVersion.patch > best->systemApiVersion.patch)) {
                best = p;
            }
        }
        return best;
    }

    struct TensorMeta {
        const char* name = nullptr;
        Qnn_TensorType_t type = QNN_TENSOR_TYPE_UNDEFINED;
        Qnn_DataType_t dataType = QNN_DATATYPE_UNDEFINED;
        Qnn_TensorMemType_t memType = QNN_TENSORMEMTYPE_UNDEFINED;
        Qnn_TensorDataFormat_t dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        Qnn_QuantizeParams_t quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
        uint32_t rank = 0;
        const uint32_t* dims = nullptr;
    };

    bool tensorMetadata(const Qnn_Tensor_t& tensor, TensorMeta* meta) {
        if (!meta) return false;
        *meta = TensorMeta{};
        switch (tensor.version) {
            case QNN_TENSOR_VERSION_1:
                meta->name = tensor.v1.name;
                meta->type = tensor.v1.type;
                meta->dataType = tensor.v1.dataType;
                meta->memType = tensor.v1.memType;
                meta->dataFormat = tensor.v1.dataFormat;
                meta->quantizeParams = tensor.v1.quantizeParams;
                meta->rank = tensor.v1.rank;
                meta->dims = tensor.v1.dimensions;
                return true;
            case QNN_TENSOR_VERSION_2:
                meta->name = tensor.v2.name;
                meta->type = tensor.v2.type;
                meta->dataType = tensor.v2.dataType;
                meta->memType = tensor.v2.memType;
                meta->dataFormat = tensor.v2.dataFormat;
                meta->quantizeParams = tensor.v2.quantizeParams;
                meta->rank = tensor.v2.rank;
                meta->dims = tensor.v2.dimensions;
                return true;
            default:
                return false;
        }
    }

    size_t tensorElementCount(const Qnn_Tensor_t& tensor) {
        TensorMeta meta;
        if (!tensorMetadata(tensor, &meta)) return 0;
        if (meta.rank > 0 && meta.dims == nullptr) return 0;
        size_t count = 1;
        for (uint32_t i = 0; i < meta.rank; ++i) {
            if (meta.dims[i] == 0) return 0;
            count *= static_cast<size_t>(meta.dims[i]);
        }
        return count;
    }

    const char* dataTypeName(Qnn_DataType_t dt) {
        switch (dt) {
            case QNN_DATATYPE_FLOAT_16: return "FLOAT16";
            case QNN_DATATYPE_FLOAT_32: return "FLOAT32";
            case QNN_DATATYPE_SFIXED_POINT_8: return "SFIXED_POINT_8";
            case QNN_DATATYPE_UFIXED_POINT_8: return "UFIXED_POINT_8";
            case QNN_DATATYPE_INT_8: return "INT8";
            case QNN_DATATYPE_UINT_8: return "UINT8";
            case QNN_DATATYPE_SFIXED_POINT_16: return "SFIXED_POINT_16";
            case QNN_DATATYPE_UFIXED_POINT_16: return "UFIXED_POINT_16";
            case QNN_DATATYPE_INT_16: return "INT16";
            case QNN_DATATYPE_UINT_16: return "UINT16";
            case QNN_DATATYPE_SFIXED_POINT_32: return "SFIXED_POINT_32";
            case QNN_DATATYPE_UFIXED_POINT_32: return "UFIXED_POINT_32";
            case QNN_DATATYPE_INT_32: return "INT32";
            case QNN_DATATYPE_UINT_32: return "UINT32";
            default: return "OTHER";
        }
    }

    size_t dataTypeBytes(Qnn_DataType_t dt) {
        switch (dt) {
            case QNN_DATATYPE_FLOAT_32: return 4;
            case QNN_DATATYPE_FLOAT_16: return 2;
            case QNN_DATATYPE_SFIXED_POINT_8:
            case QNN_DATATYPE_UFIXED_POINT_8:
            case QNN_DATATYPE_INT_8:
            case QNN_DATATYPE_UINT_8: return 1;
            case QNN_DATATYPE_SFIXED_POINT_16:
            case QNN_DATATYPE_UFIXED_POINT_16:
            case QNN_DATATYPE_INT_16:
            case QNN_DATATYPE_UINT_16: return 2;
            case QNN_DATATYPE_SFIXED_POINT_32:
            case QNN_DATATYPE_UFIXED_POINT_32:
            case QNN_DATATYPE_INT_32:
            case QNN_DATATYPE_UINT_32: return 4;
            default: return 0;
        }
    }

    bool isFloat16(Qnn_DataType_t dt) { return dt == QNN_DATATYPE_FLOAT_16; }
    bool isFloat32(Qnn_DataType_t dt) { return dt == QNN_DATATYPE_FLOAT_32; }
    bool isSigned8(Qnn_DataType_t dt) { return dt == QNN_DATATYPE_INT_8 || dt == QNN_DATATYPE_SFIXED_POINT_8; }
    bool isUnsigned8(Qnn_DataType_t dt) { return dt == QNN_DATATYPE_UINT_8 || dt == QNN_DATATYPE_UFIXED_POINT_8; }
    bool isSigned16(Qnn_DataType_t dt) { return dt == QNN_DATATYPE_INT_16 || dt == QNN_DATATYPE_SFIXED_POINT_16; }
    bool isUnsigned16(Qnn_DataType_t dt) { return dt == QNN_DATATYPE_UINT_16 || dt == QNN_DATATYPE_UFIXED_POINT_16; }
    bool isQuantized8(Qnn_DataType_t dt) { return isSigned8(dt) || isUnsigned8(dt); }
    bool isQuantized16(Qnn_DataType_t dt) { return isSigned16(dt) || isUnsigned16(dt); }
    bool isQuantized(Qnn_DataType_t dt) { return isQuantized8(dt) || isQuantized16(dt); }

    bool hasScaleOffsetQuant(const TensorMeta& meta, float* scale, int32_t* offset) {
        if (!scale || !offset) return false;
        if (meta.quantizeParams.encodingDefinition != QNN_DEFINITION_DEFINED) return false;
        if (meta.quantizeParams.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) return false;
        const float s = meta.quantizeParams.scaleOffsetEncoding.scale;
        const int32_t o = meta.quantizeParams.scaleOffsetEncoding.offset;
        if (!std::isfinite(s) || s <= 0.0f) return false;
        *scale = s;
        *offset = o;
        return true;
    }

    std::string quantSummary(const TensorMeta& meta) {
        std::ostringstream oss;
        oss << "quantDef=" << static_cast<int>(meta.quantizeParams.encodingDefinition)
            << " encoding=" << static_cast<int>(meta.quantizeParams.quantizationEncoding);
        float scale = 0.0f;
        int32_t offset = 0;
        if (hasScaleOffsetQuant(meta, &scale, &offset)) {
            oss << " scale=" << scale << " offset=" << offset;
        }
        return oss.str();
    }

    std::string tensorSummary(const Qnn_Tensor_t& tensor) {
        TensorMeta m;
        if (!tensorMetadata(tensor, &m)) return "<invalid>";
        std::ostringstream oss;
        oss << "name=" << (m.name ? m.name : "<null>")
            << " type=" << static_cast<int>(m.type)
            << " dataType=" << dataTypeName(m.dataType)
            << " rank=" << m.rank << " shape=[";
        for (uint32_t i = 0; i < m.rank; ++i) {
            if (i) oss << ',';
            oss << (m.dims ? m.dims[i] : 0);
        }
        oss << "] memType=" << static_cast<int>(m.memType) << " " << quantSummary(m);
        return oss.str();
    }

    bool setTensorIoBuffer(Qnn_Tensor_t* tensor, bool input, void* data, uint32_t bytes) {
        if (!tensor) return false;
        Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;
        clientBuffer.data = data;
        clientBuffer.dataSize = bytes;
        const auto type = input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_APP_READ;
        switch (tensor->version) {
            case QNN_TENSOR_VERSION_1:
                tensor->v1.type = type;
                tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;
                tensor->v1.clientBuf = clientBuffer;
                return true;
            case QNN_TENSOR_VERSION_2:
                tensor->v2.type = type;
                tensor->v2.memType = QNN_TENSORMEMTYPE_RAW;
                tensor->v2.clientBuf = clientBuffer;
                return true;
            default:
                return false;
        }
    }

    uint16_t floatToHalf(float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t sign = (bits >> 16) & 0x8000u;
        const uint32_t exp = (bits >> 23) & 0xffu;
        const uint32_t mantissa = bits & 0x7fffffu;
        if (exp == 0xffu) {
            if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7c00u);
            uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
            if (!payload) payload = 1;
            return static_cast<uint16_t>(sign | 0x7c00u | payload);
        }
        if (exp == 0) return static_cast<uint16_t>(sign);
        const int32_t halfExp = static_cast<int32_t>(exp) - 127 + 15;
        if (halfExp <= 0) {
            if (halfExp < -10) return static_cast<uint16_t>(sign);
            const uint32_t mant = mantissa | 0x800000u;
            const int shift = 14 - halfExp;
            uint32_t halfMant = mant >> shift;
            const uint32_t roundBit = (mant >> (shift - 1)) & 1u;
            if (roundBit) ++halfMant;
            return static_cast<uint16_t>(sign | halfMant);
        }
        if (halfExp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
        uint32_t halfMant = mantissa >> 13;
        const uint32_t roundBits = mantissa & 0x1fffu;
        if (roundBits > 0x1000u || (roundBits == 0x1000u && (halfMant & 1u))) {
            ++halfMant;
            if (halfMant == 0x400u) {
                halfMant = 0;
                const int32_t newExp = halfExp + 1;
                if (newExp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
                return static_cast<uint16_t>(sign | (static_cast<uint32_t>(newExp) << 10));
            }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(halfExp) << 10) | halfMant);
    }

    float halfToFloat(uint16_t half) {
        const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
        const uint32_t exp = (static_cast<uint32_t>(half) >> 10) & 0x1fu;
        const uint32_t mantissa = static_cast<uint32_t>(half & 0x03ffu);
        uint32_t bits = 0;
        if (exp == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                uint32_t mant = mantissa;
                int exponent = -14;
                while ((mant & 0x400u) == 0) {
                    mant <<= 1;
                    --exponent;
                }
                mant &= 0x3ffu;
                bits = sign | (static_cast<uint32_t>(exponent + 127) << 23) | (mant << 13);
            }
        } else if (exp == 0x1fu) {
            bits = sign | 0x7f800000u | (mantissa << 13);
        } else {
            bits = sign | ((exp + 112u) << 23) | (mantissa << 13);
        }
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    bool quantizeInt8(float realValue, float scale, int32_t offset, bool signedType, int8_t* signedOut, uint8_t* unsignedOut) {
        if (!std::isfinite(realValue) || !std::isfinite(scale) || scale <= 0.0f) return false;
        const float qFloat = std::round(realValue / scale - static_cast<float>(offset));
        if (signedType) {
            const float qClamped = std::max(-128.0f, std::min(127.0f, qFloat));
            if (signedOut) *signedOut = static_cast<int8_t>(static_cast<int32_t>(qClamped));
        } else {
            const float qClamped = std::max(0.0f, std::min(255.0f, qFloat));
            if (unsignedOut) *unsignedOut = static_cast<uint8_t>(static_cast<int32_t>(qClamped));
        }
        return true;
    }

    float dequantizeInt8(int32_t q, float scale, int32_t offset) {
        return (static_cast<float>(q) + static_cast<float>(offset)) * scale;
    }

    bool fillLogicalInput(const Qnn_Tensor_t& tensor, const float* logicalNchw, size_t logicalElements, std::vector<uint8_t>* storage) {
        if (!logicalNchw || !storage) return false;
        TensorMeta m;
        if (!tensorMetadata(tensor, &m) || logicalElements != tensorElementCount(tensor) || m.rank != 4 || !m.dims) return false;

        const uint32_t d0 = m.dims[0];
        const uint32_t d1 = m.dims[1];
        const uint32_t d2 = m.dims[2];
        const uint32_t d3 = m.dims[3];
        if (d0 != 1 || d1 * d2 * d3 != logicalElements) return false;

        const bool nchw = d1 == 3 && d2 * d3 == logicalElements / 3u;
        const bool nhwc = d3 == 3 && d1 * d2 == logicalElements / 3u;
        if (!nchw && !nhwc) return false;

        const size_t bytesPerElement = dataTypeBytes(m.dataType);
        if (bytesPerElement == 0) return false;
        storage->resize(logicalElements * bytesPerElement);

        if (isFloat16(m.dataType)) {
            auto* dst = reinterpret_cast<uint16_t*>(storage->data());
            if (nchw) {
                for (size_t i = 0; i < logicalElements; ++i) dst[i] = floatToHalf(logicalNchw[i]);
            } else {
                const size_t plane = static_cast<size_t>(d1) * d2;
                for (uint32_t y = 0; y < d1; ++y) {
                    for (uint32_t x = 0; x < d2; ++x) {
                        const size_t pix = static_cast<size_t>(y) * d2 + x;
                        dst[pix * 3u + 0u] = floatToHalf(logicalNchw[pix]);
                        dst[pix * 3u + 1u] = floatToHalf(logicalNchw[plane + pix]);
                        dst[pix * 3u + 2u] = floatToHalf(logicalNchw[2u * plane + pix]);
                    }
                }
            }
            return true;
        }

        if (isFloat32(m.dataType)) {
            auto* dst = reinterpret_cast<float*>(storage->data());
            if (nchw) {
                std::memcpy(dst, logicalNchw, logicalElements * sizeof(float));
            } else {
                const size_t plane = static_cast<size_t>(d1) * d2;
                for (uint32_t y = 0; y < d1; ++y) {
                    for (uint32_t x = 0; x < d2; ++x) {
                        const size_t pix = static_cast<size_t>(y) * d2 + x;
                        dst[pix * 3u + 0u] = logicalNchw[pix];
                        dst[pix * 3u + 1u] = logicalNchw[plane + pix];
                        dst[pix * 3u + 2u] = logicalNchw[2u * plane + pix];
                    }
                }
            }
            return true;
        }

        if (isQuantized8(m.dataType)) {
            float scale = 0.0f;
            int32_t offset = 0;
            if (!hasScaleOffsetQuant(m, &scale, &offset)) return false;
            const bool signedType = isSigned8(m.dataType);
            auto writeOne = [&](size_t dstIndex, float realValue) {
                if (signedType) {
                    int8_t q = 0;
                    if (!quantizeInt8(realValue, scale, offset, true, &q, nullptr)) return false;
                    reinterpret_cast<int8_t*>(storage->data())[dstIndex] = q;
                } else {
                    uint8_t q = 0;
                    if (!quantizeInt8(realValue, scale, offset, false, nullptr, &q)) return false;
                    reinterpret_cast<uint8_t*>(storage->data())[dstIndex] = q;
                }
                return true;
            };

            if (nchw) {
                for (size_t i = 0; i < logicalElements; ++i) if (!writeOne(i, logicalNchw[i])) return false;
            } else {
                const size_t plane = static_cast<size_t>(d1) * d2;
                for (uint32_t y = 0; y < d1; ++y) {
                    for (uint32_t x = 0; x < d2; ++x) {
                        const size_t pix = static_cast<size_t>(y) * d2 + x;
                        if (!writeOne(pix * 3u + 0u, logicalNchw[pix])) return false;
                        if (!writeOne(pix * 3u + 1u, logicalNchw[plane + pix])) return false;
                        if (!writeOne(pix * 3u + 2u, logicalNchw[2u * plane + pix])) return false;
                    }
                }
            }
            return true;
        }

        return false;
    }

    bool decodeOutputTensor(const Qnn_Tensor_t& tensor, const std::vector<uint8_t>& storage, std::vector<float>* out) {
        if (!out || storage.empty()) return false;
        TensorMeta m;
        if (!tensorMetadata(tensor, &m)) return false;
        const size_t elements = tensorElementCount(tensor);
        if (elements == 0) return false;
        const size_t expectedBytes = elements * dataTypeBytes(m.dataType);
        if (expectedBytes == 0 || storage.size() < expectedBytes) return false;
        out->resize(elements);

        if (isFloat16(m.dataType)) {
            const auto* src = reinterpret_cast<const uint16_t*>(storage.data());
            for (size_t i = 0; i < elements; ++i) (*out)[i] = halfToFloat(src[i]);
            return true;
        }

        if (isFloat32(m.dataType)) {
            const auto* src = reinterpret_cast<const float*>(storage.data());
            std::memcpy(out->data(), src, elements * sizeof(float));
            return true;
        }

        if (isQuantized8(m.dataType)) {
            float scale = 0.0f;
            int32_t offset = 0;
            if (!hasScaleOffsetQuant(m, &scale, &offset)) return false;
            if (isSigned8(m.dataType)) {
                const auto* src = reinterpret_cast<const int8_t*>(storage.data());
                for (size_t i = 0; i < elements; ++i) (*out)[i] = dequantizeInt8(static_cast<int32_t>(src[i]), scale, offset);
            } else {
                const auto* src = reinterpret_cast<const uint8_t*>(storage.data());
                for (size_t i = 0; i < elements; ++i) (*out)[i] = dequantizeInt8(static_cast<int32_t>(src[i]), scale, offset);
            }
            return true;
        }
        return false;
    }

    // =========================================================================
    // Runner 定义及解析加载逻辑
    // =========================================================================

    struct Runner {
        void* htpLib = nullptr;
        void* systemLib = nullptr;
        QNN_INTERFACE_VER_TYPE qnn{};
        QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystem{};
        Qnn_BackendHandle_t backend = nullptr;
        Qnn_DeviceHandle_t device = nullptr;
        Qnn_ContextHandle_t context = nullptr;
        Qnn_GraphHandle_t graph = nullptr;
        QnnSystemContext_Handle_t systemContext = nullptr;

        std::vector<uint8_t> binary;
        std::vector<Qnn_Tensor_t> inputs;
        std::vector<Qnn_Tensor_t> outputs;

        std::string graphName;
        std::string assetName;
        bool initialized = false;

        // 🌟 DINOv2 动态维度嗅探属性 🌟
        uint32_t inputWidth = 0;
        uint32_t inputHeight = 0;
        uint32_t embeddingDim = 0; // 取 384 或 768
        size_t inputElements = 0;
        size_t outputElements = 0;

        ~Runner() { close(); }
        void close() {
            initialized = false;
            graph = nullptr;
            if (context && qnn.contextFree) qnn.contextFree(context, nullptr);
            context = nullptr;
            if (systemContext && qnnSystem.systemContextFree) qnnSystem.systemContextFree(systemContext);
            systemContext = nullptr;
            if (device && qnn.deviceFree) qnn.deviceFree(device);
            device = nullptr;
            if (backend && qnn.backendFree) qnn.backendFree(backend);
            backend = nullptr;
            if (systemLib) { dlclose(systemLib); systemLib = nullptr; }
            if (htpLib) { dlclose(htpLib); htpLib = nullptr; }
        }
    };

    template <typename GraphInfo>
    bool captureGraph(Runner* r, const GraphInfo& g) {
        if (!r || !g.graphName) return false;
        r->graphName = g.graphName;
        r->inputs.clear();
        r->outputs.clear();
        if (g.numGraphInputs && g.graphInputs) {
            r->inputs.assign(g.graphInputs, g.graphInputs + g.numGraphInputs);
        }
        if (g.numGraphOutputs && g.graphOutputs) {
            r->outputs.assign(g.graphOutputs, g.graphOutputs + g.numGraphOutputs);
        }
        return !r->inputs.empty() && !r->outputs.empty();
    }

    bool readGraphInfo(Runner* r) {
        const QnnSystemContext_BinaryInfo_t* info = nullptr;
        Qnn_ContextBinarySize_t infoSize = 0;
        if (!r || !r->systemContext || !r->qnnSystem.systemContextGetBinaryInfo) return false;
        const auto rc = r->qnnSystem.systemContextGetBinaryInfo(
                r->systemContext, r->binary.data(), static_cast<Qnn_ContextBinarySize_t>(r->binary.size()), &info, &infoSize);
        if (rc != QNN_SUCCESS || !info) return false;

        switch (info->version) {
            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
                const auto& v = info->contextBinaryInfoV1;
                for (uint32_t i = 0; i < v.numGraphs; ++i) {
                    const auto& g = v.graphs[i];
                    if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1 && captureGraph(r, g.graphInfoV1)) return true;
                }
                break;
            }
            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {
                const auto& v = info->contextBinaryInfoV2;
                for (uint32_t i = 0; i < v.numGraphs; ++i) {
                    const auto& g = v.graphs[i];
                    if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 && captureGraph(r, g.graphInfoV2)) return true;
                }
                break;
            }
#if (QNN_API_VERSION_MINOR >= 21)
                case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {
                const auto& v = info->contextBinaryInfoV3;
                for (uint32_t i = 0; i < v.numGraphs; ++i) {
                    const auto& g = v.graphs[i];
                    if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3 && captureGraph(r, g.graphInfoV3)) return true;
                }
                break;
            }
#endif
            default: break;
        }
        return false;
    }

    bool validateContract(Runner* r) {
        if (!r || r->inputs.size() != 1 || r->outputs.size() != 1) {
            loge("DINOv2 requires exactly 1 input and 1 output");
            return false;
        }

        // 🌟 1. 动态解析输入维度 (NCHW)
        TensorMeta inMeta;
        if (!tensorMetadata(r->inputs[0], &inMeta) || inMeta.rank != 4 || !inMeta.dims) return false;

        // 我们强制验证通道数必须为 3 (C=3)
        if (inMeta.dims[1] != 3) {
            loge("DINOv2 input must have 3 channels (NCHW), got %d", inMeta.dims[1]);
            return false;
        }
        r->inputHeight = inMeta.dims[2]; // e.g. 224, 336, 448, 512
        r->inputWidth = inMeta.dims[3];
        r->inputElements = tensorElementCount(r->inputs[0]);

        // 🌟 2. 动态解析输出维度，完美兼容 [1, 384], [1, 768], [1, 257, 384] 等一切妖魔鬼怪
        TensorMeta outMeta;
        if (!tensorMetadata(r->outputs[0], &outMeta) || outMeta.rank == 0 || !outMeta.dims) return false;

        r->outputElements = tensorElementCount(r->outputs[0]);
        // 灵魂维度永远是张量的最后一维！(无论是 s14 的 384 还是 b14 的 768)
        r->embeddingDim = outMeta.dims[outMeta.rank - 1];

        logi("DINOv2 Contract PASSED! Model=%s, Resolution=%dx%d, EmbeddingDim=%d, TotalOutElements=%zu",
             r->assetName.c_str(), r->inputWidth, r->inputHeight, r->embeddingDim, r->outputElements);
        return true;
    }

    std::unique_ptr<Runner> createRunner(AAssetManager* manager, const std::string& assetName, const std::string& nativeLibDir) {
        if (!manager || nativeLibDir.empty()) return nullptr;
        setenv("ADSP_LIBRARY_PATH", nativeLibDir.c_str(), 1);

        AAsset* asset = AAssetManager_open(manager, assetName.c_str(), AASSET_MODE_BUFFER);
        if (!asset) { loge("Asset open failed: %s", assetName.c_str()); return nullptr; }

        const size_t length = static_cast<size_t>(AAsset_getLength(asset));
        const void* data = AAsset_getBuffer(asset);
        if (!data || !length) { AAsset_close(asset); return nullptr; }

        auto r = std::make_unique<Runner>();
        r->assetName = assetName;
        r->binary.resize(length);
        std::memcpy(r->binary.data(), data, length);
        AAsset_close(asset);

        const std::string htpPath = nativeLibDir + "/libQnnHtp.so";
        const std::string sysPath = nativeLibDir + "/libQnnSystem.so";

        r->htpLib = dlopen(htpPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!r->htpLib) { loge("dlopen libQnnHtp.so failed: %s", dlError().c_str()); return nullptr; }
        r->systemLib = dlopen(sysPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!r->systemLib) { loge("dlopen libQnnSystem.so failed: %s", dlError().c_str()); return nullptr; }

        auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn>(dlsym(r->htpLib, "QnnInterface_getProviders"));
        auto getSystemProviders = reinterpret_cast<QnnSystemInterfaceGetProvidersFn>(dlsym(r->systemLib, "QnnSystemInterface_getProviders"));
        if (!getProviders || !getSystemProviders) { loge("QNN provider symbols not found"); return nullptr; }

        const QnnInterface_t** providers = nullptr;
        uint32_t providerCount = 0;
        Qnn_ErrorHandle_t rc = getProviders(&providers, &providerCount);
        if (rc != QNN_SUCCESS || !providers || !providerCount) { loge("QnnInterface_getProviders failed rc=%u", static_cast<unsigned>(rc)); return nullptr; }

        const auto* provider = chooseQnnProvider(providers, providerCount);
        if (!provider) return nullptr;
        r->qnn = provider->QNN_INTERFACE_VER_NAME;

        const QnnSystemInterface_t** sysProviders = nullptr;
        uint32_t sysCount = 0;
        rc = getSystemProviders(&sysProviders, &sysCount);
        if (rc != QNN_SUCCESS || !sysProviders || !sysCount) return nullptr;
        const auto* sysProvider = chooseSystemProvider(sysProviders, sysCount);
        if (!sysProvider) return nullptr;
        r->qnnSystem = sysProvider->QNN_SYSTEM_INTERFACE_VER_NAME;

        if (r->qnn.backendCreate(nullptr, nullptr, &r->backend) != QNN_SUCCESS || !r->backend) return nullptr;
        if (r->qnn.deviceCreate(nullptr, nullptr, &r->device) != QNN_SUCCESS || !r->device) return nullptr;
        if (r->qnnSystem.systemContextCreate(&r->systemContext) != QNN_SUCCESS || !r->systemContext) return nullptr;
        if (!readGraphInfo(r.get())) return nullptr;
        if (!validateContract(r.get())) return nullptr;

        if (r->qnn.contextValidateBinary) {
            rc = r->qnn.contextValidateBinary(r->backend, r->device, nullptr, r->binary.data(), static_cast<Qnn_ContextBinarySize_t>(r->binary.size()));
            if (rc != QNN_SUCCESS) return nullptr;
        }

        rc = r->qnn.contextCreateFromBinary(r->backend, r->device, nullptr, r->binary.data(), static_cast<Qnn_ContextBinarySize_t>(r->binary.size()), &r->context, nullptr);
        if (rc != QNN_SUCCESS || !r->context) return nullptr;

        rc = r->qnn.graphRetrieve(r->context, r->graphName.c_str(), &r->graph);
        if (rc != QNN_SUCCESS || !r->graph) return nullptr;

        r->initialized = true;
        return r;
    }

    bool executeOnce(Runner* r, const float* logicalNchw, std::vector<float>* output) {
        if (!r || !r->initialized || !output) return false;

        std::vector<uint8_t> inputStorage;
        if (!fillLogicalInput(r->inputs[0], logicalNchw, r->inputElements, &inputStorage)) return false;

        Qnn_Tensor_t inTensor = r->inputs[0];
        if (!setTensorIoBuffer(&inTensor, true, inputStorage.data(), inputStorage.size())) return false;

        std::vector<uint8_t> outputStorage;
        Qnn_Tensor_t outTensor = r->outputs[0];

        // Setup Output buffer
        TensorMeta outMeta;
        if (!tensorMetadata(outTensor, &outMeta)) return false;
        size_t expectedOutBytes = r->outputElements * dataTypeBytes(outMeta.dataType);
        outputStorage.assign(expectedOutBytes, 0);
        if (!setTensorIoBuffer(&outTensor, false, outputStorage.data(), outputStorage.size())) return false;

        // ----------------------------------------------------
        // 执行 QNN Graph
        // ----------------------------------------------------
        if (r->qnn.graphExecute(r->graph, &inTensor, 1, &outTensor, 1, nullptr, nullptr) != QNN_SUCCESS) {
            loge("DINOv2 graphExecute failed");
            return false;
        }

        // ----------------------------------------------------
        // 🌟 3. 剥离 CLS Token 魔法 🌟
        // ----------------------------------------------------
        std::vector<float> decodedRaw;
        if (!decodeOutputTensor(outTensor, outputStorage, &decodedRaw)) return false;

        // decodedRaw 里面可能包含多达 257*384 个浮点数
        // 但 DINOv2 的 CLS 灵魂向量永远在数组的最前面 (前 embeddingDim 个数)
        output->clear();
        output->reserve(r->embeddingDim);
        output->insert(output->end(), decodedRaw.begin(), decodedRaw.begin() + r->embeddingDim);

        return true;
    }
} // namespace

// ============================================================================
// JNI 接口
// ============================================================================
extern "C" JNIEXPORT jlong JNICALL
Java_com_example_aiobs_ai_Dinov2QnnContextBinaryRunner_nativeCreate(
        JNIEnv* env, jclass, jobject assetManager, jstring assetName, jstring nativeLibDir) {
    if (!env || !assetManager || !assetName || !nativeLibDir) return 0;
    AAssetManager* manager = AAssetManager_fromJava(env, assetManager);
    if (!manager) return 0;

    const char* a = env->GetStringUTFChars(assetName, nullptr);
    const char* d = env->GetStringUTFChars(nativeLibDir, nullptr);
    if (!a || !d) {
        if (a) env->ReleaseStringUTFChars(assetName, a);
        if (d) env->ReleaseStringUTFChars(nativeLibDir, d);
        return 0;
    }

    std::string asset(a);
    std::string libdir(d);
    env->ReleaseStringUTFChars(assetName, a);
    env->ReleaseStringUTFChars(nativeLibDir, d);

    auto r = createRunner(manager, asset, libdir);
    if (!r) return 0;
    return reinterpret_cast<jlong>(r.release());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_aiobs_ai_Dinov2QnnContextBinaryRunner_nativeGetInputWidth(
        JNIEnv*, jclass, jlong handle) {
    auto* r = reinterpret_cast<Runner*>(handle);
    return r ? r->inputWidth : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_aiobs_ai_Dinov2QnnContextBinaryRunner_nativeGetInputHeight(
        JNIEnv*, jclass, jlong handle) {
    auto* r = reinterpret_cast<Runner*>(handle);
    return r ? r->inputHeight : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_aiobs_ai_Dinov2QnnContextBinaryRunner_nativeGetEmbeddingDim(
        JNIEnv*, jclass, jlong handle) {
    auto* r = reinterpret_cast<Runner*>(handle);
    return r ? r->embeddingDim : 0;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_aiobs_ai_Dinov2QnnContextBinaryRunner_nativeRun(
        JNIEnv* env, jclass, jlong handle, jfloatArray inputArray) {
    auto* r = reinterpret_cast<Runner*>(handle);
    if (!r || !r->initialized) return nullptr;

    if (env->GetArrayLength(inputArray) != static_cast<jsize>(r->inputElements)) {
        loge("DINOv2 invalid input length: expected %zu, got %d", r->inputElements, env->GetArrayLength(inputArray));
        return nullptr;
    }

    jboolean isCopy = JNI_FALSE;
    jfloat* input = env->GetFloatArrayElements(inputArray, &isCopy);
    if (!input) return nullptr;

    std::vector<float> output;
    const bool ok = executeOnce(r, input, &output);
    env->ReleaseFloatArrayElements(inputArray, input, JNI_ABORT);

    if (!ok) return nullptr;

    jfloatArray result = env->NewFloatArray(output.size());
    env->SetFloatArrayRegion(result, 0, output.size(), output.data());
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_aiobs_ai_Dinov2QnnContextBinaryRunner_nativeClose(
        JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<Runner*>(handle);
}