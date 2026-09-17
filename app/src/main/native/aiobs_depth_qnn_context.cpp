#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace {

    constexpr char kTag[] = "DepthQnnCtxBin";

    constexpr uint32_t kExpectedInputElements =
            3u * 518u * 518u;

    constexpr uint32_t kExpectedOutputElements =
            518u * 518u;

// ============================================================================
// Logging
// ============================================================================

    void logi(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);

        __android_log_vprint(
                ANDROID_LOG_INFO,
                kTag,
                fmt,
                args);

        va_end(args);
    }

    void loge(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);

        __android_log_vprint(
                ANDROID_LOG_ERROR,
                kTag,
                fmt,
                args);

        va_end(args);
    }

// ============================================================================
// Dynamic loading helpers
// ============================================================================

    std::string dlError() {
        const char* e = dlerror();

        return e
               ? std::string(e)
               : std::string("unknown dlerror");
    }

// ============================================================================
// Tensor helpers
//
// QNN 2.49 defines:
//
//   Qnn_Tensor_t {
//       Qnn_TensorVersion_t version;
//       union {
//           Qnn_TensorV1_t v1;
//           Qnn_TensorV2_t v2;
//       };
//   }
//
// QNN 2.49 headers used by this project do NOT provide
// QNN_TENSOR_GET_* / QNN_TENSOR_SET_* helper macros.
//
// Therefore all access is explicitly dispatched through tensor.version.
//
// IMPORTANT:
// Do not overwrite tensor.version.
// The Context Binary used by this project provides V2 tensors.
// ============================================================================

    bool tensorGetMetadata(
            const Qnn_Tensor_t& tensor,
            const char** name,
            Qnn_TensorType_t* type,
            Qnn_DataType_t* dataType,
            Qnn_TensorMemType_t* memType,
            uint32_t* rank,
            const uint32_t** dimensions) {

        if (name == nullptr ||
            type == nullptr ||
            dataType == nullptr ||
            memType == nullptr ||
            rank == nullptr ||
            dimensions == nullptr) {

            return false;
        }

        switch (tensor.version) {

            case QNN_TENSOR_VERSION_1:

                *name = tensor.v1.name;
                *type = tensor.v1.type;
                *dataType = tensor.v1.dataType;
                *memType = tensor.v1.memType;
                *rank = tensor.v1.rank;
                *dimensions = tensor.v1.dimensions;

                return true;

            case QNN_TENSOR_VERSION_2:

                *name = tensor.v2.name;
                *type = tensor.v2.type;
                *dataType = tensor.v2.dataType;
                *memType = tensor.v2.memType;
                *rank = tensor.v2.rank;
                *dimensions = tensor.v2.dimensions;

                return true;

            default:

                return false;
        }
    }

    size_t elementCount(
            const Qnn_Tensor_t& tensor) {

        const char* name = nullptr;

        Qnn_TensorType_t type =
                QNN_TENSOR_TYPE_UNDEFINED;

        Qnn_DataType_t dataType =
                QNN_DATATYPE_UNDEFINED;

        Qnn_TensorMemType_t memType =
                QNN_TENSORMEMTYPE_UNDEFINED;

        uint32_t rank = 0;

        const uint32_t* dims = nullptr;

        if (!tensorGetMetadata(
                tensor,
                &name,
                &type,
                &dataType,
                &memType,
                &rank,
                &dims)) {

            return 0;
        }

        if (rank > 0 &&
            dims == nullptr) {

            return 0;
        }

        size_t count = 1;

        for (uint32_t i = 0;
             i < rank;
             ++i) {

            count *=
                    static_cast<size_t>(
                            dims[i]);
        }

        return count;
    }

    std::string tensorSummary(
            const Qnn_Tensor_t& tensor) {

        std::ostringstream oss;

        const char* name = nullptr;

        Qnn_TensorType_t type =
                QNN_TENSOR_TYPE_UNDEFINED;

        Qnn_DataType_t dataType =
                QNN_DATATYPE_UNDEFINED;

        Qnn_TensorMemType_t memType =
                QNN_TENSORMEMTYPE_UNDEFINED;

        uint32_t rank = 0;

        const uint32_t* dims = nullptr;

        if (!tensorGetMetadata(
                tensor,
                &name,
                &type,
                &dataType,
                &memType,
                &rank,
                &dims)) {

            oss << "<invalid-tensor-version>"
                << " version="
                << static_cast<int>(
                        tensor.version);

            return oss.str();
        }

        oss << (name
                ? name
                : "<unnamed>")
            << " version="
            << static_cast<int>(
                    tensor.version)
            << " type="
            << static_cast<int>(type)
            << " dataType="
            << static_cast<int>(dataType)
            << " rank="
            << rank
            << " shape=[";

        for (uint32_t i = 0;
             i < rank;
             ++i) {

            if (i != 0) {
                oss << ',';
            }

            oss << (dims
                    ? dims[i]
                    : 0);
        }

        oss << "] memType="
            << static_cast<int>(
                    memType);

        if (tensor.version ==
            QNN_TENSOR_VERSION_2) {

            oss << " isProduced="
                << static_cast<int>(
                        tensor.v2.isProduced);
        }

        return oss.str();
    }

    bool setTensorIoBuffer(
            Qnn_Tensor_t* tensor,
            bool input,
            void* data,
            uint32_t dataSize) {

        if (tensor == nullptr) {

            loge(
                    "setTensorIoBuffer: tensor == nullptr");

            return false;
        }

        if (data == nullptr &&
            dataSize != 0) {

            loge(
                    "setTensorIoBuffer: null data with dataSize=%u",
                    dataSize);

            return false;
        }

        const Qnn_TensorType_t tensorType =
                input
                ? QNN_TENSOR_TYPE_APP_WRITE
                : QNN_TENSOR_TYPE_APP_READ;

        Qnn_ClientBuffer_t clientBuffer =
                QNN_CLIENT_BUFFER_INIT;

        clientBuffer.data =
                data;

        clientBuffer.dataSize =
                dataSize;

        switch (tensor->version) {

            case QNN_TENSOR_VERSION_1:

                tensor->v1.type =
                        tensorType;

                tensor->v1.memType =
                        QNN_TENSORMEMTYPE_RAW;

                tensor->v1.clientBuf =
                        clientBuffer;

                return true;

            case QNN_TENSOR_VERSION_2:

                tensor->v2.type =
                        tensorType;

                tensor->v2.memType =
                        QNN_TENSORMEMTYPE_RAW;

                tensor->v2.clientBuf =
                        clientBuffer;

                return true;

            default:

                loge(
                        "setTensorIoBuffer: unsupported tensor version=%d",
                        static_cast<int>(
                                tensor->version));

                return false;
        }
    }

// ============================================================================
// Runner
//
// IMPORTANT:
//
// This class intentionally DOES NOT own any HTP performance vote.
//
// HTP performance policy is now global and is owned by:
//   HtpPerformanceManager.kt
//       -> aiobs_htp_performance.cpp
//
// This runner only owns:
//   - QNN HTP provider
//   - QNN backend/device
//   - QNN System context
//   - Context Binary
//   - QNN context
//   - QNN graph
//
// It does NOT own:
//   - QnnHtpDevice_PerfInfrastructure_t
//   - powerConfigId
//   - createPowerConfigId()
//   - setPowerConfig()
//   - destroyPowerConfigId()
// ============================================================================

    struct Runner {

        void* htpLib = nullptr;

        void* systemLib = nullptr;

        QNN_INTERFACE_VER_TYPE qnn{};

        QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystem{};

        Qnn_BackendHandle_t backend = nullptr;

        Qnn_DeviceHandle_t device = nullptr;

        Qnn_ContextHandle_t context = nullptr;

        Qnn_GraphHandle_t graph = nullptr;

        QnnSystemContext_Handle_t systemContext =
                nullptr;

        std::vector<uint8_t> binary;

        std::vector<Qnn_Tensor_t> inputMeta;

        std::vector<Qnn_Tensor_t> outputMeta;

        std::string graphName;

        bool initialized = false;

        ~Runner() {
            close();
        }

        void close() {

            // -----------------------------------------------------------------------
            // NO HTP performance vote cleanup here.
            //
            // Performance ownership belongs to HtpPerformanceManager.
            // -----------------------------------------------------------------------

            if (context != nullptr &&
                qnn.contextFree != nullptr) {

                qnn.contextFree(
                        context,
                        nullptr);
            }

            context = nullptr;

            graph = nullptr;

            if (systemContext != nullptr &&
                qnnSystem.systemContextFree != nullptr) {

                qnnSystem.systemContextFree(
                        systemContext);
            }

            systemContext = nullptr;

            if (device != nullptr &&
                qnn.deviceFree != nullptr) {

                qnn.deviceFree(
                        device);
            }

            device = nullptr;

            if (backend != nullptr &&
                qnn.backendFree != nullptr) {

                qnn.backendFree(
                        backend);
            }

            backend = nullptr;

            if (systemLib != nullptr) {

                dlclose(systemLib);

                systemLib = nullptr;
            }

            if (htpLib != nullptr) {

                dlclose(htpLib);

                htpLib = nullptr;
            }

            initialized = false;
        }
    };

// ============================================================================
// QNN provider helpers
// ============================================================================

    using QnnInterfaceGetProvidersFn =
            Qnn_ErrorHandle_t (*)(
                    const QnnInterface_t*** providerList,
                    uint32_t* numProviders);

    using QnnSystemInterfaceGetProvidersFn =
            Qnn_ErrorHandle_t (*)(
                    const QnnSystemInterface_t*** providerList,
                    uint32_t* numProviders);

    const QnnInterface_t* chooseQnnProvider(
            const QnnInterface_t** providers,
            uint32_t count) {

        const QnnInterface_t* best = nullptr;

        for (uint32_t i = 0;
             i < count;
             ++i) {

            const auto* p =
                    providers[i];

            if (p == nullptr) {
                continue;
            }

            if (best == nullptr ||
                p->apiVersion.coreApiVersion.major >
                best->apiVersion.coreApiVersion.major ||
                (p->apiVersion.coreApiVersion.major ==
                 best->apiVersion.coreApiVersion.major &&
                 p->apiVersion.coreApiVersion.minor >
                 best->apiVersion.coreApiVersion.minor) ||
                (p->apiVersion.coreApiVersion.major ==
                 best->apiVersion.coreApiVersion.major &&
                 p->apiVersion.coreApiVersion.minor ==
                 best->apiVersion.coreApiVersion.minor &&
                 p->apiVersion.coreApiVersion.patch >
                 best->apiVersion.coreApiVersion.patch)) {

                best = p;
            }
        }

        return best;
    }

    const QnnSystemInterface_t* chooseSystemProvider(
            const QnnSystemInterface_t** providers,
            uint32_t count) {

        const QnnSystemInterface_t* best = nullptr;

        for (uint32_t i = 0;
             i < count;
             ++i) {

            const auto* p =
                    providers[i];

            if (p == nullptr) {
                continue;
            }

            if (best == nullptr ||
                p->systemApiVersion.major >
                best->systemApiVersion.major ||
                (p->systemApiVersion.major ==
                 best->systemApiVersion.major &&
                 p->systemApiVersion.minor >
                 best->systemApiVersion.minor) ||
                (p->systemApiVersion.major ==
                 best->systemApiVersion.major &&
                 p->systemApiVersion.minor ==
                 best->systemApiVersion.minor &&
                 p->systemApiVersion.patch >
                 best->systemApiVersion.patch)) {

                best = p;
            }
        }

        return best;
    }

// ============================================================================
// Context Binary metadata
// ============================================================================

    template <typename GraphInfo>
    bool captureGraph(
            Runner* runner,
            const GraphInfo& info) {

        if (runner == nullptr) {
            return false;
        }

        if (info.graphName == nullptr) {
            return false;
        }

        runner->graphName =
                info.graphName;

        if (info.numGraphInputs > 0 &&
            info.graphInputs != nullptr) {

            runner->inputMeta.assign(
                    info.graphInputs,
                    info.graphInputs +
                    info.numGraphInputs);
        }

        if (info.numGraphOutputs > 0 &&
            info.graphOutputs != nullptr) {

            runner->outputMeta.assign(
                    info.graphOutputs,
                    info.graphOutputs +
                    info.numGraphOutputs);
        }

        logi(
                "Captured graph metadata: graph=%s inputs=%u outputs=%u",
                runner->graphName.c_str(),
                info.numGraphInputs,
                info.numGraphOutputs);

        return !runner->inputMeta.empty() &&
               !runner->outputMeta.empty();
    }

    bool readGraphInfo(
            Runner* runner) {

        if (runner == nullptr) {
            return false;
        }

        if (runner->systemContext == nullptr ||
            runner->qnnSystem.systemContextGetBinaryInfo ==
            nullptr) {

            loge(
                    "QNN System interface is unavailable");

            return false;
        }

        const QnnSystemContext_BinaryInfo_t*
                binaryInfo = nullptr;

        Qnn_ContextBinarySize_t binaryInfoSize = 0;

        const Qnn_ErrorHandle_t rc =
                runner->qnnSystem.systemContextGetBinaryInfo(
                        runner->systemContext,
                        runner->binary.data(),
                        static_cast<
                                Qnn_ContextBinarySize_t>(
                                runner->binary.size()),
                        &binaryInfo,
                        &binaryInfoSize);

        if (rc != QNN_SUCCESS ||
            binaryInfo == nullptr) {

            loge(
                    "systemContextGetBinaryInfo failed rc=%u",
                    static_cast<unsigned>(rc));

            return false;
        }

        logi(
                "QNN context binary info version=%d size=%llu binaryBytes=%zu",
                static_cast<int>(
                        binaryInfo->version),
                static_cast<unsigned long long>(
                        binaryInfoSize),
                runner->binary.size());

        switch (binaryInfo->version) {

            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {

                const auto& info =
                        binaryInfo->contextBinaryInfoV1;

                if (info.numGraphs < 1) {

                    loge(
                            "Binary contains no graphs");

                    return false;
                }

                for (uint32_t i = 0;
                     i < info.numGraphs;
                     ++i) {

                    const auto& g =
                            info.graphs[i];

                    if (g.version ==
                        QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {

                        if (captureGraph(
                                runner,
                                g.graphInfoV1)) {

                            return true;
                        }
                    }
                }

                return false;
            }

            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {

                const auto& info =
                        binaryInfo->contextBinaryInfoV2;

                if (info.numGraphs < 1) {

                    loge(
                            "Binary contains no graphs");

                    return false;
                }

                for (uint32_t i = 0;
                     i < info.numGraphs;
                     ++i) {

                    const auto& g =
                            info.graphs[i];

                    if (g.version ==
                        QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {

                        if (captureGraph(
                                runner,
                                g.graphInfoV2)) {

                            return true;
                        }
                    }
                }

                return false;
            }

#if (QNN_API_VERSION_MINOR >= 21)

            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {

                const auto& info =
                        binaryInfo->contextBinaryInfoV3;

                if (info.numGraphs < 1) {

                    loge(
                            "Binary contains no graphs");

                    return false;
                }

                for (uint32_t i = 0;
                     i < info.numGraphs;
                     ++i) {

                    const auto& g =
                            info.graphs[i];

                    if (g.version ==
                        QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {

                        if (captureGraph(
                                runner,
                                g.graphInfoV3)) {

                            return true;
                        }
                    }
                }

                return false;
            }

#endif

            default:

                loge(
                        "Unsupported QNN binary info version=%d",
                        static_cast<int>(
                                binaryInfo->version));

                return false;
        }
    }

// ============================================================================
// Graph validation
// ============================================================================

    bool validateAndPrepare(
            Runner* runner) {

        if (runner == nullptr) {
            return false;
        }

        if (runner->inputMeta.size() != 1 ||
            runner->outputMeta.empty()) {

            loge(
                    "Unexpected IO counts input=%zu output=%zu",
                    runner->inputMeta.size(),
                    runner->outputMeta.size());

            return false;
        }

        const auto& in =
                runner->inputMeta[0];

        const auto& out =
                runner->outputMeta[0];

        logi(
                "Graph=%s",
                runner->graphName.c_str());

        logi(
                "Input: %s",
                tensorSummary(in).c_str());

        logi(
                "Output: %s",
                tensorSummary(out).c_str());

        // -----------------------------------------------------------------------
        // Input tensor
        // -----------------------------------------------------------------------

        const char* inputName = nullptr;

        Qnn_TensorType_t inputTensorType =
                QNN_TENSOR_TYPE_UNDEFINED;

        Qnn_DataType_t inputType =
                QNN_DATATYPE_UNDEFINED;

        Qnn_TensorMemType_t inputMemType =
                QNN_TENSORMEMTYPE_UNDEFINED;

        uint32_t inputRank = 0;

        const uint32_t* inputDims = nullptr;

        if (!tensorGetMetadata(
                in,
                &inputName,
                &inputTensorType,
                &inputType,
                &inputMemType,
                &inputRank,
                &inputDims)) {

            loge(
                    "Unsupported input tensor version=%d",
                    static_cast<int>(
                            in.version));

            return false;
        }

        if (inputRank == 0 ||
            inputDims == nullptr) {

            loge(
                    "Invalid input tensor dimensions");

            return false;
        }

        const size_t inputElements =
                elementCount(in);

        if (inputElements !=
            kExpectedInputElements) {

            loge(
                    "Input element count mismatch: %zu expected=%u",
                    inputElements,
                    kExpectedInputElements);

            return false;
        }

        if (inputType !=
            QNN_DATATYPE_FLOAT_32) {

            loge(
                    "Expected float32 input, got dataType=%d",
                    static_cast<int>(
                            inputType));

            return false;
        }

        if (inputRank != 4) {

            loge(
                    "Expected input rank=4, got rank=%u",
                    inputRank);

            return false;
        }

        if (inputDims[0] != 1 ||
            inputDims[1] != 3 ||
            inputDims[2] != 518 ||
            inputDims[3] != 518) {

            loge(
                    "Unexpected input shape rank=%u shape=[%u,%u,%u,%u]",
                    inputRank,
                    inputDims[0],
                    inputDims[1],
                    inputDims[2],
                    inputDims[3]);

            return false;
        }

        // -----------------------------------------------------------------------
        // Output tensor
        // -----------------------------------------------------------------------

        const char* outputName = nullptr;

        Qnn_TensorType_t outputTensorType =
                QNN_TENSOR_TYPE_UNDEFINED;

        Qnn_DataType_t outputType =
                QNN_DATATYPE_UNDEFINED;

        Qnn_TensorMemType_t outputMemType =
                QNN_TENSORMEMTYPE_UNDEFINED;

        uint32_t outputRank = 0;

        const uint32_t* outputDims = nullptr;

        if (!tensorGetMetadata(
                out,
                &outputName,
                &outputTensorType,
                &outputType,
                &outputMemType,
                &outputRank,
                &outputDims)) {

            loge(
                    "Unsupported output tensor version=%d",
                    static_cast<int>(
                            out.version));

            return false;
        }

        if (outputRank == 0 ||
            outputDims == nullptr) {

            loge(
                    "Invalid output tensor dimensions");

            return false;
        }

        const size_t outputElements =
                elementCount(out);

        if (outputElements !=
            kExpectedOutputElements) {

            loge(
                    "Output element count mismatch: %zu expected=%u",
                    outputElements,
                    kExpectedOutputElements);

            return false;
        }

        if (outputType !=
            QNN_DATATYPE_FLOAT_32) {

            loge(
                    "Expected float32 output, got dataType=%d",
                    static_cast<int>(
                            outputType));

            return false;
        }

        logi(
                "Input contract PASSED: float32 NCHW [1,3,518,518]");

        logi(
                "Output contract PASSED: float32 elements=%zu",
                outputElements);

        logi(
                "Input tensor version=%d type=%d memType=%d",
                static_cast<int>(
                        in.version),
                static_cast<int>(
                        inputTensorType),
                static_cast<int>(
                        inputMemType));

        logi(
                "Output tensor version=%d type=%d memType=%d",
                static_cast<int>(
                        out.version),
                static_cast<int>(
                        outputTensorType),
                static_cast<int>(
                        outputMemType));

        logi(
                "QNN tensor contract validation PASSED");

        return true;
    }

// ============================================================================
// Runner creation
// ============================================================================

    std::unique_ptr<Runner> createRunner(
            AAssetManager* assetManager,
            const std::string& assetName,
            const std::string& nativeLibDir) {

        if (assetManager == nullptr) {

            loge(
                    "AssetManager is null");

            return nullptr;
        }

        if (nativeLibDir.empty()) {

            loge(
                    "Native library directory is empty");

            return nullptr;
        }

        // -----------------------------------------------------------------------
        // HTP DSP library path
        //
        // This is runtime library discovery only.
        // It is NOT a performance vote.
        // -----------------------------------------------------------------------

        setenv(
                "ADSP_LIBRARY_PATH",
                nativeLibDir.c_str(),
                1);

        logi(
                "ADSP_LIBRARY_PATH=%s",
                nativeLibDir.c_str());

        // -----------------------------------------------------------------------
        // Load Context Binary from assets
        // -----------------------------------------------------------------------

        AAsset* asset =
                AAssetManager_open(
                        assetManager,
                        assetName.c_str(),
                        AASSET_MODE_BUFFER);

        if (asset == nullptr) {

            loge(
                    "Failed to open asset %s",
                    assetName.c_str());

            return nullptr;
        }

        const size_t length =
                static_cast<size_t>(
                        AAsset_getLength(asset));

        const void* buffer =
                AAsset_getBuffer(asset);

        if (length == 0 ||
            buffer == nullptr) {

            loge(
                    "Invalid asset %s length=%zu",
                    assetName.c_str(),
                    length);

            AAsset_close(asset);

            return nullptr;
        }

        auto runner =
                std::make_unique<Runner>();

        runner->binary.resize(
                length);

        std::memcpy(
                runner->binary.data(),
                buffer,
                length);

        AAsset_close(asset);

        logi(
                "Loaded Context Binary asset=%s bytes=%zu",
                assetName.c_str(),
                runner->binary.size());

        // -----------------------------------------------------------------------
        // QNN libraries
        // -----------------------------------------------------------------------

        const std::string htpPath =
                nativeLibDir +
                "/libQnnHtp.so";

        const std::string systemPath =
                nativeLibDir +
                "/libQnnSystem.so";

        runner->htpLib =
                dlopen(
                        htpPath.c_str(),
                        RTLD_NOW | RTLD_LOCAL);

        if (runner->htpLib == nullptr) {

            loge(
                    "dlopen HTP failed: %s",
                    dlError().c_str());

            return nullptr;
        }

        runner->systemLib =
                dlopen(
                        systemPath.c_str(),
                        RTLD_NOW | RTLD_LOCAL);

        if (runner->systemLib == nullptr) {

            loge(
                    "dlopen QnnSystem failed: %s",
                    dlError().c_str());

            return nullptr;
        }

        logi(
                "QNN libraries loaded successfully");

        // -----------------------------------------------------------------------
        // HTP provider
        // -----------------------------------------------------------------------

        auto getProviders =
                reinterpret_cast<
                        QnnInterfaceGetProvidersFn>(
                        dlsym(
                                runner->htpLib,
                                "QnnInterface_getProviders"));

        if (getProviders == nullptr) {

            loge(
                    "QnnInterface_getProviders symbol not found");

            return nullptr;
        }

        const QnnInterface_t** providers =
                nullptr;

        uint32_t providerCount = 0;

        Qnn_ErrorHandle_t rc =
                getProviders(
                        &providers,
                        &providerCount);

        if (rc != QNN_SUCCESS ||
            providers == nullptr ||
            providerCount == 0) {

            loge(
                    "QnnInterface_getProviders failed rc=%u count=%u",
                    static_cast<unsigned>(rc),
                    providerCount);

            return nullptr;
        }

        logi(
                "QNN HTP providers=%u",
                providerCount);

        const auto* provider =
                chooseQnnProvider(
                        providers,
                        providerCount);

        if (provider == nullptr) {

            loge(
                    "No QNN provider found");

            return nullptr;
        }

        runner->qnn =
                provider->QNN_INTERFACE_VER_NAME;

        logi(
                "QNN provider=%s core=%u.%u.%u backend=%u.%u.%u",
                provider->providerName
                ? provider->providerName
                : "<null>",
                provider->apiVersion.coreApiVersion.major,
                provider->apiVersion.coreApiVersion.minor,
                provider->apiVersion.coreApiVersion.patch,
                provider->apiVersion.backendApiVersion.major,
                provider->apiVersion.backendApiVersion.minor,
                provider->apiVersion.backendApiVersion.patch);

        // -----------------------------------------------------------------------
        // QNN System provider
        // -----------------------------------------------------------------------

        auto getSystemProviders =
                reinterpret_cast<
                        QnnSystemInterfaceGetProvidersFn>(
                        dlsym(
                                runner->systemLib,
                                "QnnSystemInterface_getProviders"));

        if (getSystemProviders == nullptr) {

            loge(
                    "QnnSystemInterface_getProviders symbol not found");

            return nullptr;
        }

        const QnnSystemInterface_t** systemProviders =
                nullptr;

        uint32_t systemProviderCount = 0;

        rc =
                getSystemProviders(
                        &systemProviders,
                        &systemProviderCount);

        if (rc != QNN_SUCCESS ||
            systemProviders == nullptr ||
            systemProviderCount == 0) {

            loge(
                    "QnnSystemInterface_getProviders failed rc=%u count=%u",
                    static_cast<unsigned>(rc),
                    systemProviderCount);

            return nullptr;
        }

        logi(
                "QNN System providers=%u",
                systemProviderCount);

        const auto* systemProvider =
                chooseSystemProvider(
                        systemProviders,
                        systemProviderCount);

        if (systemProvider == nullptr) {

            loge(
                    "No QNN System provider found");

            return nullptr;
        }

        runner->qnnSystem =
                systemProvider->
                        QNN_SYSTEM_INTERFACE_VER_NAME;

        logi(
                "QNN System core=%u.%u.%u",
                systemProvider->systemApiVersion.major,
                systemProvider->systemApiVersion.minor,
                systemProvider->systemApiVersion.patch);

        // -----------------------------------------------------------------------
        // Required APIs
        //
        // NOTE:
        // deviceGetInfrastructure is deliberately NOT required anymore.
        // The depth runner no longer owns the HTP performance infrastructure.
        // -----------------------------------------------------------------------

        if (runner->qnn.backendCreate == nullptr ||
            runner->qnn.deviceCreate == nullptr ||
            runner->qnn.contextCreateFromBinary ==
            nullptr ||
            runner->qnn.graphRetrieve == nullptr ||
            runner->qnn.graphExecute == nullptr) {

            loge(
                    "Required QNN API entries are missing");

            return nullptr;
        }

        if (runner->qnnSystem.systemContextCreate ==
            nullptr ||
            runner->qnnSystem.systemContextGetBinaryInfo ==
            nullptr) {

            loge(
                    "Required QNN System API entries are missing");

            return nullptr;
        }

        // -----------------------------------------------------------------------
        // Backend
        // -----------------------------------------------------------------------

        logi(
                "backendCreate BEGIN");

        rc =
                runner->qnn.backendCreate(
                        nullptr,
                        nullptr,
                        &runner->backend);

        logi(
                "backendCreate END rc=%u backend=%p",
                static_cast<unsigned>(rc),
                static_cast<void*>(runner->backend));

        if (rc != QNN_SUCCESS ||
            runner->backend == nullptr) {

            loge(
                    "backendCreate failed rc=%u",
                    static_cast<unsigned>(rc));

            return nullptr;
        }

        logi(
                "QNN backend created");

        // -----------------------------------------------------------------------
        // Device
        // -----------------------------------------------------------------------

        logi(
                "deviceCreate BEGIN");

        rc =
                runner->qnn.deviceCreate(
                        nullptr,
                        nullptr,
                        &runner->device);

        logi(
                "deviceCreate END rc=%u device=%p",
                static_cast<unsigned>(rc),
                static_cast<void*>(runner->device));

        if (rc != QNN_SUCCESS ||
            runner->device == nullptr) {

            loge(
                    "deviceCreate failed rc=%u",
                    static_cast<unsigned>(rc));

            return nullptr;
        }

        logi(
                "QNN device created");

        // -----------------------------------------------------------------------
        // IMPORTANT:
        //
        // No call to:
        //
        //   deviceGetInfrastructure()
        //   createPowerConfigId()
        //   setPowerConfig()
        //
        // HTP performance ownership has been removed from this runner.
        // -----------------------------------------------------------------------

        logi(
                "HTP performance vote ownership: NONE");

        // -----------------------------------------------------------------------
        // System context
        // -----------------------------------------------------------------------

        logi(
                "systemContextCreate BEGIN");

        rc =
                runner->qnnSystem.systemContextCreate(
                        &runner->systemContext);

        logi(
                "systemContextCreate END rc=%u context=%p",
                static_cast<unsigned>(rc),
                static_cast<void*>(runner->systemContext));

        if (rc != QNN_SUCCESS ||
            runner->systemContext == nullptr) {

            loge(
                    "systemContextCreate failed rc=%u",
                    static_cast<unsigned>(rc));

            return nullptr;
        }

        logi(
                "QNN System context created");

        // -----------------------------------------------------------------------
        // Read graph metadata BEFORE graphRetrieve().
        // -----------------------------------------------------------------------

        if (!readGraphInfo(
                runner.get())) {

            loge(
                    "Failed to read QNN graph metadata");

            return nullptr;
        }

        if (!validateAndPrepare(
                runner.get())) {

            loge(
                    "Failed to validate QNN graph metadata");

            return nullptr;
        }

        // -----------------------------------------------------------------------
        // Optional context binary validation
        // -----------------------------------------------------------------------

        if (runner->qnn.contextValidateBinary !=
            nullptr) {

            logi(
                    "contextValidateBinary BEGIN");

            rc =
                    runner->qnn.contextValidateBinary(
                            runner->backend,
                            runner->device,
                            nullptr,
                            runner->binary.data(),
                            static_cast<
                                    Qnn_ContextBinarySize_t>(
                                    runner->binary.size()));

            logi(
                    "contextValidateBinary END rc=%u",
                    static_cast<unsigned>(rc));

            if (rc != QNN_SUCCESS) {

                loge(
                        "QNN context binary validation failed rc=%u",
                        static_cast<unsigned>(rc));

                return nullptr;
            }
        }

        // -----------------------------------------------------------------------
        // Create context from binary
        // -----------------------------------------------------------------------

        logi(
                "contextCreateFromBinary BEGIN");

        rc =
                runner->qnn.contextCreateFromBinary(
                        runner->backend,
                        runner->device,
                        nullptr,
                        runner->binary.data(),
                        static_cast<
                                Qnn_ContextBinarySize_t>(
                                runner->binary.size()),
                        &runner->context,
                        nullptr);

        logi(
                "contextCreateFromBinary END rc=%u",
                static_cast<unsigned>(rc));

        if (rc != QNN_SUCCESS ||
            runner->context == nullptr) {

            loge(
                    "contextCreateFromBinary failed rc=%u",
                    static_cast<unsigned>(rc));

            return nullptr;
        }

        logi(
                "QNN Context created successfully");

        // -----------------------------------------------------------------------
        // Graph retrieve
        // -----------------------------------------------------------------------

        logi(
                "graphRetrieve BEGIN graph=%s",
                runner->graphName.c_str());

        rc =
                runner->qnn.graphRetrieve(
                        runner->context,
                        runner->graphName.c_str(),
                        &runner->graph);

        logi(
                "graphRetrieve END rc=%u graph=%p",
                static_cast<unsigned>(rc),
                static_cast<void*>(runner->graph));

        if (rc != QNN_SUCCESS ||
            runner->graph == nullptr) {

            loge(
                    "graphRetrieve failed rc=%u graph=%s",
                    static_cast<unsigned>(rc),
                    runner->graphName.c_str());

            return nullptr;
        }

        logi(
                "QNN graph retrieved successfully graph=%s",
                runner->graphName.c_str());

        runner->initialized = true;

        logi(
                "QNN Context Binary ready graph=%s binaryBytes=%zu",
                runner->graphName.c_str(),
                runner->binary.size());

        return runner;
    }

// ============================================================================
// Execute one inference
// ============================================================================

    bool executeOnce(
            Runner* runner,
            const float* input,
            float* output,
            double* elapsedMs) {

        logi(
                "executeOnce ENTER runner=%p",
                static_cast<void*>(runner));

        if (runner == nullptr ||
            !runner->initialized ||
            runner->graph == nullptr ||
            runner->inputMeta.empty() ||
            runner->outputMeta.empty()) {

            loge(
                    "executeOnce FAIL state runner=%p initialized=%d graph=%p inputs=%zu outputs=%zu",
                    static_cast<void*>(runner),
                    runner
                    ? (runner->initialized ? 1 : 0)
                    : 0,
                    runner
                    ? static_cast<void*>(
                            runner->graph)
                    : nullptr,
                    runner
                    ? runner->inputMeta.size()
                    : 0,
                    runner
                    ? runner->outputMeta.size()
                    : 0);

            return false;
        }

        if (input == nullptr ||
            output == nullptr ||
            elapsedMs == nullptr) {

            loge(
                    "executeOnce FAIL null argument input=%p output=%p elapsedMs=%p",
                    static_cast<const void*>(input),
                    static_cast<void*>(output),
                    static_cast<void*>(elapsedMs));

            return false;
        }

        logi(
                "executeOnce input metadata: %s",
                tensorSummary(
                        runner->inputMeta[0]).c_str());

        logi(
                "executeOnce output metadata: %s",
                tensorSummary(
                        runner->outputMeta[0]).c_str());

        const uint32_t inputBytes =
                kExpectedInputElements *
                sizeof(float);

        const uint32_t outputBytes =
                kExpectedOutputElements *
                sizeof(float);

        Qnn_Tensor_t inputTensor =
                runner->inputMeta[0];

        Qnn_Tensor_t outputTensor =
                runner->outputMeta[0];

        logi(
                "executeOnce copied tensors inputVersion=%d outputVersion=%d",
                static_cast<int>(
                        inputTensor.version),
                static_cast<int>(
                        outputTensor.version));

        if (!setTensorIoBuffer(
                &inputTensor,
                true,
                const_cast<float*>(input),
                inputBytes)) {

            loge(
                    "executeOnce FAIL: set input tensor buffer");

            return false;
        }

        if (!setTensorIoBuffer(
                &outputTensor,
                false,
                output,
                outputBytes)) {

            loge(
                    "executeOnce FAIL: set output tensor buffer");

            return false;
        }

        logi(
                "executeOnce buffers attached inputVersion=%d outputVersion=%d",
                static_cast<int>(
                        inputTensor.version),
                static_cast<int>(
                        outputTensor.version));

        logi(
                "executeOnce GRAPH EXECUTE BEGIN graph=%p",
                static_cast<void*>(runner->graph));

        const auto start =
                std::chrono::steady_clock::now();

        const Qnn_ErrorHandle_t rc =
                runner->qnn.graphExecute(
                        runner->graph,
                        &inputTensor,
                        1,
                        &outputTensor,
                        1,
                        nullptr,
                        nullptr);

        const auto end =
                std::chrono::steady_clock::now();

        logi(
                "executeOnce GRAPH EXECUTE RETURN rc=%u",
                static_cast<unsigned>(rc));

        if (rc != QNN_SUCCESS) {

            loge(
                    "executeOnce GRAPH EXECUTE FAILED rc=%u",
                    static_cast<unsigned>(rc));

            return false;
        }

        *elapsedMs =
                std::chrono::duration<
                        double,
                        std::milli>(
                        end - start).count();

        logi(
                "executeOnce GRAPH EXECUTE END elapsed=%.3f ms",
                *elapsedMs);

        return true;
    }

}  // namespace

// ============================================================================
// JNI: create
// ============================================================================

extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_aiobs_ai_DepthQnnContextBinaryRunner_nativeCreate(
        JNIEnv* env,
        jclass,
        jobject assetManager,
        jstring assetName,
        jstring nativeLibDir) {

    logi(
            "nativeCreate ENTER");

    if (env == nullptr ||
        assetManager == nullptr ||
        assetName == nullptr ||
        nativeLibDir == nullptr) {

        loge(
                "nativeCreate FAIL: null argument");

        return 0;
    }

    AAssetManager* manager =
            AAssetManager_fromJava(
                    env,
                    assetManager);

    if (manager == nullptr) {

        loge(
                "AAssetManager_fromJava failed");

        return 0;
    }

    const char* assetChars =
            env->GetStringUTFChars(
                    assetName,
                    nullptr);

    const char* libDirChars =
            env->GetStringUTFChars(
                    nativeLibDir,
                    nullptr);

    if (assetChars == nullptr ||
        libDirChars == nullptr) {

        loge(
                "nativeCreate FAIL: GetStringUTFChars");

        if (assetChars != nullptr) {

            env->ReleaseStringUTFChars(
                    assetName,
                    assetChars);
        }

        if (libDirChars != nullptr) {

            env->ReleaseStringUTFChars(
                    nativeLibDir,
                    libDirChars);
        }

        return 0;
    }

    std::string assetPath(
            assetChars);

    std::string libDir(
            libDirChars);

    env->ReleaseStringUTFChars(
            assetName,
            assetChars);

    env->ReleaseStringUTFChars(
            nativeLibDir,
            libDirChars);

    logi(
            "nativeCreate creating runner asset=%s libDir=%s",
            assetPath.c_str(),
            libDir.c_str());

    auto runner =
            createRunner(
                    manager,
                    assetPath,
                    libDir);

    if (!runner) {

        loge(
                "nativeCreate FAIL: createRunner returned null");

        return 0;
    }

    auto* rawRunner =
            runner.release();

    logi(
            "nativeCreate SUCCESS handle=%p",
            static_cast<void*>(rawRunner));

    return reinterpret_cast<jlong>(
            rawRunner);
}

// ============================================================================
// JNI: describe
// ============================================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_aiobs_ai_DepthQnnContextBinaryRunner_nativeDescribe(
        JNIEnv* env,
        jclass,
        jlong handle) {

    logi(
            "nativeDescribe ENTER handle=%p",
            reinterpret_cast<void*>(handle));

    if (env == nullptr) {
        return nullptr;
    }

    auto* runner =
            reinterpret_cast<Runner*>(
                    handle);

    if (runner == nullptr ||
        !runner->initialized) {

        loge(
                "nativeDescribe NOT_READY");

        return env->NewStringUTF(
                "NOT_READY");
    }

    std::ostringstream oss;

    oss << "backend=QNN/HTP\n"
        << "graph="
        << runner->graphName
        << "\n"
        << "binaryBytes="
        << runner->binary.size()
        << "\n"
        << "inputs="
        << runner->inputMeta.size()
        << "\n"
        << "outputs="
        << runner->outputMeta.size()
        << "\n"
        << "htpPerformanceVoteOwnership=NONE\n";

    if (!runner->inputMeta.empty()) {

        oss << "input[0]="
            << tensorSummary(
                    runner->inputMeta[0])
            << "\n";
    }

    if (!runner->outputMeta.empty()) {

        oss << "output[0]="
            << tensorSummary(
                    runner->outputMeta[0])
            << "\n";
    }

    logi(
            "nativeDescribe RETURN");

    return env->NewStringUTF(
            oss.str().c_str());
}

// ============================================================================
// JNI: benchmark
// ============================================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_aiobs_ai_DepthQnnContextBinaryRunner_nativeBenchmark(
        JNIEnv* env,
        jclass,
        jlong handle,
        jint warmups,
        jint iterations) {

    logi(
            "nativeBenchmark ENTER handle=%p warmups=%d iterations=%d",
            reinterpret_cast<void*>(handle),
            static_cast<int>(warmups),
            static_cast<int>(iterations));

    if (env == nullptr) {

        loge(
                "nativeBenchmark FAIL: env == nullptr");

        return nullptr;
    }

    auto* runner =
            reinterpret_cast<Runner*>(
                    handle);

    logi(
            "nativeBenchmark runner=%p",
            static_cast<void*>(runner));

    if (runner == nullptr) {

        loge(
                "nativeBenchmark FAIL: runner == nullptr");

        return env->NewStringUTF(
                "FAIL: runner == null");
    }

    logi(
            "nativeBenchmark state initialized=%d graph=%p inputs=%zu outputs=%zu",
            runner->initialized ? 1 : 0,
            static_cast<void*>(runner->graph),
            runner->inputMeta.size(),
            runner->outputMeta.size());

    if (!runner->initialized) {

        loge(
                "nativeBenchmark FAIL: runner not initialized");

        return env->NewStringUTF(
                "FAIL: runner not initialized");
    }

    if (warmups < 0) {
        warmups = 0;
    }

    if (iterations <= 0) {
        iterations = 1;
    }

    logi(
            "nativeBenchmark normalized warmups=%d iterations=%d",
            static_cast<int>(warmups),
            static_cast<int>(iterations));

    std::vector<float> input(
            kExpectedInputElements);

    std::vector<float> output(
            kExpectedOutputElements,
            0.0f);

    logi(
            "nativeBenchmark allocated input=%zu floats output=%zu floats",
            input.size(),
            output.size());

    // -----------------------------------------------------------------------
    // Deterministic synthetic RGB NCHW input.
    // -----------------------------------------------------------------------

    const size_t plane =
            518u * 518u;

    for (size_t p = 0;
         p < plane;
         ++p) {

        const float x =
                static_cast<float>(
                        p % 518u) /
                517.0f;

        const float y =
                static_cast<float>(
                        p / 518u) /
                517.0f;

        input[p] =
                x;

        input[plane + p] =
                y;

        input[2 * plane + p] =
                0.5f * (x + y);
    }

    logi(
            "nativeBenchmark synthetic input prepared");

    // -----------------------------------------------------------------------
    // Warmup
    // -----------------------------------------------------------------------

    logi(
            "nativeBenchmark WARMUP BEGIN count=%d",
            static_cast<int>(warmups));

    for (int i = 0;
         i < warmups;
         ++i) {

        logi(
                "nativeBenchmark WARMUP[%d] executeOnce BEGIN",
                i);

        double elapsed = 0.0;

        if (!executeOnce(
                runner,
                input.data(),
                output.data(),
                &elapsed)) {

            loge(
                    "nativeBenchmark WARMUP[%d] executeOnce FAILED",
                    i);

            return env->NewStringUTF(
                    "FAIL: warmup graphExecute failed");
        }

        logi(
                "nativeBenchmark WARMUP[%d] executeOnce END elapsed=%.3f ms",
                i,
                elapsed);
    }

    logi(
            "nativeBenchmark WARMUP END");

    // -----------------------------------------------------------------------
    // Timed iterations
    // -----------------------------------------------------------------------

    std::vector<double> times;

    times.reserve(
            static_cast<size_t>(
                    iterations));

    logi(
            "nativeBenchmark TIMED BEGIN count=%d",
            static_cast<int>(iterations));

    for (int i = 0;
         i < iterations;
         ++i) {

        logi(
                "nativeBenchmark ITER[%d] executeOnce BEGIN",
                i);

        double elapsed = 0.0;

        if (!executeOnce(
                runner,
                input.data(),
                output.data(),
                &elapsed)) {

            loge(
                    "nativeBenchmark ITER[%d] executeOnce FAILED",
                    i);

            return env->NewStringUTF(
                    "FAIL: benchmark graphExecute failed");
        }

        times.push_back(
                elapsed);

        logi(
                "nativeBenchmark ITER[%d] executeOnce END elapsed=%.3f ms",
                i,
                elapsed);
    }

    logi(
            "nativeBenchmark TIMED END");

    std::sort(
            times.begin(),
            times.end());

    double sum = 0.0;

    for (double v : times) {
        sum += v;
    }

    const double mean =
            sum /
            static_cast<double>(
                    times.size());

    const double median =
            (times.size() % 2 == 0)
            ? (times[times.size() / 2 - 1] +
               times[times.size() / 2]) *
              0.5
            : times[times.size() / 2];

    // -----------------------------------------------------------------------
    // Output sanity
    // -----------------------------------------------------------------------

    float outMin =
            std::numeric_limits<float>::infinity();

    float outMax =
            -std::numeric_limits<float>::infinity();

    double outSum = 0.0;

    uint32_t finite = 0;

    for (float v : output) {

        if (std::isfinite(v)) {

            outMin =
                    std::min(
                            outMin,
                            v);

            outMax =
                    std::max(
                            outMax,
                            v);

            outSum +=
                    static_cast<double>(
                            v);

            ++finite;
        }
    }

    logi(
            "nativeBenchmark OUTPUT sanity finite=%u/%zu min=%f max=%f",
            finite,
            output.size(),
            outMin,
            outMax);

    std::ostringstream oss;

    oss.setf(
            std::ios::fixed);

    oss.precision(3);

    oss << "PASS: direct QNN Context Binary / HTP\n"
        << "graph="
        << runner->graphName
        << "\n"
        << "htpPerformanceVoteOwnership=NONE\n"
        << "warmup="
        << warmups
        << " iterations="
        << iterations
        << "\n"
        << "minMs="
        << times.front()
        << "\n"
        << "medianMs="
        << median
        << "\n"
        << "meanMs="
        << mean
        << "\n"
        << "maxMs="
        << times.back()
        << "\n"
        << "outputFinite="
        << finite
        << "/"
        << output.size();

    if (finite > 0) {

        oss << " outputMin="
            << outMin
            << " outputMax="
            << outMax
            << " outputMean="
            << (outSum /
                static_cast<double>(
                        finite));
    }

    logi(
            "nativeBenchmark SUCCESS returning report");

    return env->NewStringUTF(
            oss.str().c_str());
}

// ============================================================================
// JNI: run one inference
// ============================================================================

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_aiobs_ai_DepthQnnContextBinaryRunner_nativeRun(
        JNIEnv* env,
        jclass,
        jlong handle,
        jfloatArray inputArray) {

    logi(
            "nativeRun ENTER handle=%p",
            reinterpret_cast<void*>(handle));

    if (env == nullptr) {
        return nullptr;
    }

    auto* runner =
            reinterpret_cast<Runner*>(
                    handle);

    if (runner == nullptr ||
        !runner->initialized ||
        inputArray == nullptr) {

        loge(
                "nativeRun FAIL invalid state");

        return nullptr;
    }

    const jsize length =
            env->GetArrayLength(
                    inputArray);

    if (length !=
        static_cast<jsize>(
                kExpectedInputElements)) {

        loge(
                "nativeRun input length mismatch: got=%d expected=%u",
                static_cast<int>(length),
                kExpectedInputElements);

        return nullptr;
    }

    jfloat* input =
            env->GetFloatArrayElements(
                    inputArray,
                    nullptr);

    if (input == nullptr) {

        loge(
                "nativeRun GetFloatArrayElements failed");

        return nullptr;
    }

    jfloatArray result =
            env->NewFloatArray(
                    kExpectedOutputElements);

    if (result == nullptr) {

        loge(
                "nativeRun NewFloatArray failed");

        env->ReleaseFloatArrayElements(
                inputArray,
                input,
                JNI_ABORT);

        return nullptr;
    }

    std::vector<float> output(
            kExpectedOutputElements,
            0.0f);

    double elapsedMs = 0.0;

    logi(
            "nativeRun executeOnce BEGIN");

    const bool ok =
            executeOnce(
                    runner,
                    input,
                    output.data(),
                    &elapsedMs);

    env->ReleaseFloatArrayElements(
            inputArray,
            input,
            JNI_ABORT);

    if (!ok) {

        loge(
                "nativeRun executeOnce FAILED");

        env->DeleteLocalRef(
                result);

        return nullptr;
    }

    env->SetFloatArrayRegion(
            result,
            0,
            static_cast<jsize>(
                    output.size()),
            output.data());

    logi(
            "nativeRun SUCCESS elapsed=%.3f ms",
            elapsedMs);

    return result;
}

// ============================================================================
// JNI: close
// ============================================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_example_aiobs_ai_DepthQnnContextBinaryRunner_nativeClose(
        JNIEnv*,
        jclass,
        jlong handle) {

    logi(
            "nativeClose handle=%p",
            reinterpret_cast<void*>(handle));

    auto* runner =
            reinterpret_cast<Runner*>(
                    handle);

    delete runner;
}