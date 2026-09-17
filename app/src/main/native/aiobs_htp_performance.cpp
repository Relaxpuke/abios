#include <jni.h>
#include <android/log.h>

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <string>

#include "QnnInterface.h"
#include "QnnDevice.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

namespace {

    constexpr char kTag[] = "AIOBS_HTP_PERF";

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
// Dynamic loading
// ============================================================================

    std::string dlError() {
        const char* error = dlerror();

        return error
               ? std::string(error)
               : std::string("unknown dlerror");
    }

// ============================================================================
// QNN provider function
// ============================================================================

    using QnnInterfaceGetProvidersFn =
            Qnn_ErrorHandle_t (*)(
                    const QnnInterface_t*** providerList,
                    uint32_t* numProviders);

// ============================================================================
// Global HTP performance manager state
//
// IMPORTANT:
//
// This state intentionally lives independently from DepthQnnContextBinaryRunner.
//
// The performance vote therefore remains active while the rest of the
// application creates/destroys:
//   - OSNet
//   - XFeat
//   - YOLO
//   - Depth
//   - other QNN/HTP workloads
//
// This is the reason this manager is different from putting the vote inside
// one particular model runner.
// ============================================================================

    struct HtpPerformanceState {

        void* htpLib = nullptr;

        QNN_INTERFACE_VER_TYPE qnn{};

        Qnn_BackendHandle_t backend = nullptr;

        Qnn_DeviceHandle_t device = nullptr;

        // NOTE:
        //
        // QnnDevice_Infrastructure_t is a pointer typedef.
        // The actual HTP implementation returned by
        // deviceGetInfrastructure() is interpreted as:
        //
        // QnnHtpDevice_Infrastructure_t*
        //
        // This is the exact pattern already proven by the working
        // DepthQnnContextBinaryRunner.
        QnnHtpDevice_PerfInfrastructure_t perfInfra{};

        bool perfInfraAvailable = false;

        uint32_t powerConfigId = 0;

        bool powerConfigCreated = false;

        bool performanceVoteActive = false;

        bool initialized = false;

        std::string nativeLibDir;
    };

    HtpPerformanceState gState;
    std::mutex gMutex;

// ============================================================================
// Provider selection
// ============================================================================

    const QnnInterface_t* chooseQnnProvider(
            const QnnInterface_t** providers,
            uint32_t count) {

        const QnnInterface_t* best = nullptr;

        if (providers == nullptr) {
            return nullptr;
        }

        for (uint32_t i = 0;
             i < count;
             ++i) {

            const QnnInterface_t* candidate =
                    providers[i];

            if (candidate == nullptr) {
                continue;
            }

            if (best == nullptr ||
                candidate->apiVersion.coreApiVersion.major >
                best->apiVersion.coreApiVersion.major ||
                (
                        candidate->apiVersion.coreApiVersion.major ==
                        best->apiVersion.coreApiVersion.major &&
                        candidate->apiVersion.coreApiVersion.minor >
                        best->apiVersion.coreApiVersion.minor
                ) ||
                (
                        candidate->apiVersion.coreApiVersion.major ==
                        best->apiVersion.coreApiVersion.major &&
                        candidate->apiVersion.coreApiVersion.minor ==
                        best->apiVersion.coreApiVersion.minor &&
                        candidate->apiVersion.coreApiVersion.patch >
                        best->apiVersion.coreApiVersion.patch
                )) {

                best = candidate;
            }
        }

        return best;
    }

// ============================================================================
// Reset global state
// ============================================================================

    void resetStateFields() {

        gState.htpLib = nullptr;

        gState.qnn = {};

        gState.backend = nullptr;

        gState.device = nullptr;

        gState.perfInfra = {};

        gState.perfInfraAvailable = false;

        gState.powerConfigId = 0;

        gState.powerConfigCreated = false;

        gState.performanceVoteActive = false;

        gState.initialized = false;

        gState.nativeLibDir.clear();
    }

// ============================================================================
// Remove performance vote
// ============================================================================

    void disablePerformanceVoteLocked() {

        if (!gState.performanceVoteActive) {
            return;
        }

        if (!gState.perfInfraAvailable) {

            loge(
                    "Cannot reset performance vote: "
                    "perf infrastructure unavailable");

            gState.performanceVoteActive = false;

            return;
        }

        logi(
                "HTP PERFORMANCE VOTE RESET BEGIN id=%u",
                gState.powerConfigId);

        QnnHtpPerfInfrastructure_PowerConfig_t resetConfig{};

        resetConfig.option =
                QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;

        resetConfig.dcvsV3Config.contextId =
                gState.powerConfigId;

        resetConfig.dcvsV3Config.setDcvsEnable =
                1;

        resetConfig.dcvsV3Config.dcvsEnable =
                1;

        const QnnHtpPerfInfrastructure_PowerConfig_t*
                resetConfigs[] = {
                        &resetConfig,
                        nullptr
                };

        const Qnn_ErrorHandle_t rc =
                gState.perfInfra.setPowerConfig(
                        gState.powerConfigId,
                        resetConfigs);

        logi(
                "HTP PERFORMANCE VOTE RESET END rc=%u",
                static_cast<unsigned>(rc));

        gState.performanceVoteActive = false;
    }

// ============================================================================
// Destroy power config
// ============================================================================

    void destroyPowerConfigLocked() {

        if (!gState.powerConfigCreated) {
            return;
        }

        if (!gState.perfInfraAvailable) {

            gState.powerConfigCreated = false;

            return;
        }

        logi(
                "HTP power config destroy BEGIN id=%u",
                gState.powerConfigId);

        const Qnn_ErrorHandle_t rc =
                gState.perfInfra.destroyPowerConfigId(
                        gState.powerConfigId);

        logi(
                "HTP power config destroy END id=%u rc=%u",
                gState.powerConfigId,
                static_cast<unsigned>(rc));

        gState.powerConfigCreated = false;
        gState.powerConfigId = 0;
    }

// ============================================================================
// Configure HTP performance vote
// ============================================================================

    bool configurePerformanceVoteLocked() {

        if (!gState.perfInfraAvailable) {

            loge(
                    "configurePerformanceVoteLocked: "
                    "perf infrastructure unavailable");

            return false;
        }

        if (!gState.powerConfigCreated) {

            loge(
                    "configurePerformanceVoteLocked: "
                    "power config has not been created");

            return false;
        }

        if (gState.performanceVoteActive) {

            logi(
                    "HTP performance vote already ACTIVE id=%u",
                    gState.powerConfigId);

            return true;
        }

        // ------------------------------------------------------------------------
        // DCVS / core / bus configuration
        // ------------------------------------------------------------------------

        QnnHtpPerfInfrastructure_PowerConfig_t dcvsConfig{};

        dcvsConfig.option =
                QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;

        dcvsConfig.dcvsV3Config.contextId =
                gState.powerConfigId;

        // Disable DCVS.
        dcvsConfig.dcvsV3Config.setDcvsEnable =
                1;

        dcvsConfig.dcvsV3Config.dcvsEnable =
                0;

        // Performance mode.
        dcvsConfig.dcvsV3Config.powerMode =
                QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;

        // ------------------------------------------------------------------------
        // Sleep latency
        // ------------------------------------------------------------------------

        dcvsConfig.dcvsV3Config.setSleepLatency =
                1;

        dcvsConfig.dcvsV3Config.sleepLatency =
                40;

        // ------------------------------------------------------------------------
        // Disable HTP sleep
        // ------------------------------------------------------------------------

        dcvsConfig.dcvsV3Config.setSleepDisable =
                1;

        dcvsConfig.dcvsV3Config.sleepDisable =
                1;

        // ------------------------------------------------------------------------
        // Bus = MAX
        // ------------------------------------------------------------------------

        dcvsConfig.dcvsV3Config.setBusParams =
                1;

        dcvsConfig.dcvsV3Config.busVoltageCornerMin =
                DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        dcvsConfig.dcvsV3Config.busVoltageCornerTarget =
                DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        dcvsConfig.dcvsV3Config.busVoltageCornerMax =
                DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        // ------------------------------------------------------------------------
        // Core = MAX
        // ------------------------------------------------------------------------

        dcvsConfig.dcvsV3Config.setCoreParams =
                1;

        dcvsConfig.dcvsV3Config.coreVoltageCornerMin =
                DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        dcvsConfig.dcvsV3Config.coreVoltageCornerTarget =
                DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        dcvsConfig.dcvsV3Config.coreVoltageCornerMax =
                DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        // ------------------------------------------------------------------------
        // RPC control latency
        // ------------------------------------------------------------------------

        QnnHtpPerfInfrastructure_PowerConfig_t rpcLatencyConfig{};

        rpcLatencyConfig.option =
                QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;

        rpcLatencyConfig.rpcControlLatencyConfig =
                100;

        const QnnHtpPerfInfrastructure_PowerConfig_t*
                powerConfigs[] = {
                        &dcvsConfig,
                        &rpcLatencyConfig,
                        nullptr
                };

        logi(
                "HTP PERFORMANCE VOTE BEGIN "
                "mode=PERFORMANCE "
                "dcvs=0 "
                "core=MAX "
                "bus=MAX "
                "sleepDisable=1 "
                "sleepLatencyUs=40 "
                "rpcControlLatencyUs=100 "
                "id=%u",
                gState.powerConfigId);

        const Qnn_ErrorHandle_t rc =
                gState.perfInfra.setPowerConfig(
                        gState.powerConfigId,
                        powerConfigs);

        if (rc != QNN_SUCCESS) {

            loge(
                    "HTP PERFORMANCE VOTE FAILED rc=%u",
                    static_cast<unsigned>(rc));

            return false;
        }

        gState.performanceVoteActive =
                true;

        logi(
                "HTP PERFORMANCE VOTE ACTIVE id=%u",
                gState.powerConfigId);

        return true;
    }

// ============================================================================
// Get HTP infrastructure
// ============================================================================

    bool acquireHtpInfrastructureLocked() {

        if (!gState.initialized) {

            loge(
                    "acquireHtpInfrastructureLocked: "
                    "QNN manager is not initialized");

            return false;
        }

        if (gState.perfInfraAvailable) {

            logi(
                    "HTP infrastructure already acquired");

            return true;
        }

        if (gState.qnn.deviceGetInfrastructure == nullptr) {

            loge(
                    "QNN deviceGetInfrastructure API unavailable");

            return false;
        }

        QnnDevice_Infrastructure_t deviceInfra =
                nullptr;

        const Qnn_ErrorHandle_t rc =
                gState.qnn.deviceGetInfrastructure(
                        &deviceInfra);

        if (rc != QNN_SUCCESS ||
            deviceInfra == nullptr) {

            loge(
                    "deviceGetInfrastructure failed rc=%u infra=%p",
                    static_cast<unsigned>(rc),
                    static_cast<void*>(deviceInfra));

            return false;
        }

        // IMPORTANT:
        //
        // QnnDevice_Infrastructure_t is itself a pointer typedef:
        //
        //   typedef struct _QnnDevice_Infrastructure_t*
        //       QnnDevice_Infrastructure_t;
        //
        // Therefore deviceInfra is already a pointer.
        //
        // The HTP-specific infrastructure returned by the HTP device is:
        //
        //   QnnHtpDevice_Infrastructure_t*
        //
        // This is the form proven by the working Depth runner.

        auto* htpInfra =
                static_cast<QnnHtpDevice_Infrastructure_t*>(
                        deviceInfra);

        if (htpInfra == nullptr) {

            loge(
                    "HTP device infrastructure cast returned null");

            return false;
        }

        gState.perfInfra =
                htpInfra->perfInfra;

        gState.perfInfraAvailable =
                true;

        logi(
                "HTP performance infrastructure acquired");

        return true;
    }

// ============================================================================
// Create QNN backend/device
// ============================================================================

    bool createQnnRuntimeLocked(
            const std::string& nativeLibDir) {

        if (nativeLibDir.empty()) {

            loge(
                    "createQnnRuntimeLocked: nativeLibDir is empty");

            return false;
        }

        gState.nativeLibDir =
                nativeLibDir;

        // ------------------------------------------------------------------------
        // ADSP library path
        // ------------------------------------------------------------------------

        setenv(
                "ADSP_LIBRARY_PATH",
                nativeLibDir.c_str(),
                1);

        logi(
                "ADSP_LIBRARY_PATH=%s",
                nativeLibDir.c_str());

        // ------------------------------------------------------------------------
        // libQnnHtp.so
        // ------------------------------------------------------------------------

        const std::string htpPath =
                nativeLibDir +
                "/libQnnHtp.so";

        gState.htpLib =
                dlopen(
                        htpPath.c_str(),
                        RTLD_NOW | RTLD_LOCAL);

        if (gState.htpLib == nullptr) {

            loge(
                    "dlopen libQnnHtp.so failed path=%s error=%s",
                    htpPath.c_str(),
                    dlError().c_str());

            return false;
        }

        logi(
                "libQnnHtp.so loaded path=%s",
                htpPath.c_str());

        // ------------------------------------------------------------------------
        // QNN providers
        // ------------------------------------------------------------------------

        auto getProviders =
                reinterpret_cast<
                        QnnInterfaceGetProvidersFn>(
                        dlsym(
                                gState.htpLib,
                                "QnnInterface_getProviders"));

        if (getProviders == nullptr) {

            loge(
                    "QnnInterface_getProviders symbol not found");

            return false;
        }

        const QnnInterface_t** providers =
                nullptr;

        uint32_t providerCount =
                0;

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

            return false;
        }

        logi(
                "QNN HTP providers=%u",
                providerCount);

        const QnnInterface_t* provider =
                chooseQnnProvider(
                        providers,
                        providerCount);

        if (provider == nullptr) {

            loge(
                    "No QNN provider selected");

            return false;
        }

        gState.qnn =
                provider->QNN_INTERFACE_VER_NAME;

        logi(
                "QNN provider=%s "
                "core=%u.%u.%u "
                "backend=%u.%u.%u",
                provider->providerName
                ? provider->providerName
                : "<null>",
                provider->apiVersion.coreApiVersion.major,
                provider->apiVersion.coreApiVersion.minor,
                provider->apiVersion.coreApiVersion.patch,
                provider->apiVersion.backendApiVersion.major,
                provider->apiVersion.backendApiVersion.minor,
                provider->apiVersion.backendApiVersion.patch);

        // ------------------------------------------------------------------------
        // Required QNN APIs
        // ------------------------------------------------------------------------

        if (gState.qnn.backendCreate == nullptr ||
            gState.qnn.deviceCreate == nullptr ||
            gState.qnn.deviceGetInfrastructure == nullptr) {

            loge(
                    "Required QNN device APIs unavailable");

            return false;
        }

        // ------------------------------------------------------------------------
        // Backend
        // ------------------------------------------------------------------------

        rc =
                gState.qnn.backendCreate(
                        nullptr,
                        nullptr,
                        &gState.backend);

        if (rc != QNN_SUCCESS ||
            gState.backend == nullptr) {

            loge(
                    "backendCreate failed rc=%u",
                    static_cast<unsigned>(rc));

            return false;
        }

        logi(
                "QNN backend created");

        // ------------------------------------------------------------------------
        // Device
        // ------------------------------------------------------------------------

        rc =
                gState.qnn.deviceCreate(
                        nullptr,
                        nullptr,
                        &gState.device);

        if (rc != QNN_SUCCESS ||
            gState.device == nullptr) {

            loge(
                    "deviceCreate failed rc=%u",
                    static_cast<unsigned>(rc));

            return false;
        }

        logi(
                "QNN device created");

        gState.initialized =
                true;

        return true;
    }

// ============================================================================
// Create power configuration
// ============================================================================

    bool createPowerConfigLocked() {

        if (!gState.perfInfraAvailable) {

            loge(
                    "createPowerConfigLocked: "
                    "perf infrastructure unavailable");

            return false;
        }

        if (gState.powerConfigCreated) {

            logi(
                    "HTP power config already created id=%u",
                    gState.powerConfigId);

            return true;
        }

        const Qnn_ErrorHandle_t rc =
                gState.perfInfra.createPowerConfigId(
                        0,
                        0,
                        &gState.powerConfigId);

        if (rc != QNN_SUCCESS) {

            loge(
                    "createPowerConfigId failed rc=%u",
                    static_cast<unsigned>(rc));

            return false;
        }

        gState.powerConfigCreated =
                true;

        logi(
                "HTP power config created id=%u",
                gState.powerConfigId);

        return true;
    }

// ============================================================================
// Full acquisition sequence
// ============================================================================

    bool acquirePerformanceLocked(
            const std::string& nativeLibDir) {

        // ------------------------------------------------------------------------
        // Fast path
        // ------------------------------------------------------------------------

        if (gState.performanceVoteActive) {

            logi(
                    "HTP performance manager already ACTIVE");

            return true;
        }

        // ------------------------------------------------------------------------
        // Runtime creation
        // ------------------------------------------------------------------------

        if (!gState.initialized) {

            logi(
                    "HTP performance manager initializing");

            if (!createQnnRuntimeLocked(
                    nativeLibDir)) {

                loge(
                        "Failed to create QNN runtime");

                return false;
            }
        }

        // ------------------------------------------------------------------------
        // Infrastructure
        // ------------------------------------------------------------------------

        if (!acquireHtpInfrastructureLocked()) {

            loge(
                    "Failed to acquire HTP performance infrastructure");

            return false;
        }

        // ------------------------------------------------------------------------
        // Power config
        // ------------------------------------------------------------------------

        if (!createPowerConfigLocked()) {

            loge(
                    "Failed to create HTP power config");

            return false;
        }

        // ------------------------------------------------------------------------
        // Performance vote
        // ------------------------------------------------------------------------

        if (!configurePerformanceVoteLocked()) {

            loge(
                    "Failed to configure HTP performance vote");

            return false;
        }

        logi(
                "HTP PERFORMANCE MANAGER ACTIVE");

        return true;
    }

// ============================================================================
// Full release sequence
// ============================================================================

    void releasePerformanceLocked() {

        logi(
                "HTP performance manager RELEASE BEGIN");

        // ------------------------------------------------------------------------
        // Performance vote first.
        // ------------------------------------------------------------------------

        disablePerformanceVoteLocked();

        // ------------------------------------------------------------------------
        // Destroy performance configuration.
        // ------------------------------------------------------------------------

        destroyPowerConfigLocked();

        // ------------------------------------------------------------------------
        // Destroy device.
        // ------------------------------------------------------------------------

        if (gState.device != nullptr &&
            gState.qnn.deviceFree != nullptr) {

            const Qnn_ErrorHandle_t rc =
                    gState.qnn.deviceFree(
                            gState.device);

            logi(
                    "QNN deviceFree rc=%u",
                    static_cast<unsigned>(rc));
        }

        gState.device =
                nullptr;

        // ------------------------------------------------------------------------
        // Destroy backend.
        // ------------------------------------------------------------------------

        if (gState.backend != nullptr &&
            gState.qnn.backendFree != nullptr) {

            const Qnn_ErrorHandle_t rc =
                    gState.qnn.backendFree(
                            gState.backend);

            logi(
                    "QNN backendFree rc=%u",
                    static_cast<unsigned>(rc));
        }

        gState.backend =
                nullptr;

        // ------------------------------------------------------------------------
        // Close dynamic library.
        // ------------------------------------------------------------------------

        if (gState.htpLib != nullptr) {

            dlclose(
                    gState.htpLib);

            gState.htpLib =
                    nullptr;

            logi(
                    "libQnnHtp.so unloaded");
        }

        resetStateFields();

        logi(
                "HTP performance manager RELEASE END");
    }

// ============================================================================
// Describe current state
// ============================================================================

    std::string describeLocked() {

        std::ostringstream oss;

        oss << "AIOBS HTP Performance Manager\n";

        oss << "initialized="
            << (gState.initialized
                ? "true"
                : "false")
            << "\n";

        oss << "perfInfrastructure="
            << (gState.perfInfraAvailable
                ? "available"
                : "unavailable")
            << "\n";

        oss << "powerConfigCreated="
            << (gState.powerConfigCreated
                ? "true"
                : "false")
            << "\n";

        oss << "performanceVote="
            << (gState.performanceVoteActive
                ? "ACTIVE"
                : "INACTIVE")
            << "\n";

        if (gState.powerConfigCreated) {

            oss << "powerConfigId="
                << gState.powerConfigId
                << "\n";
        }

        if (!gState.nativeLibDir.empty()) {

            oss << "nativeLibDir="
                << gState.nativeLibDir
                << "\n";
        }

        return oss.str();
    }

}  // namespace

// ============================================================================
// JNI nativeAcquire
//
// Kotlin expected shape:
//
// external fun nativeAcquire(
//     nativeLibDir: String
// ): Boolean
// ============================================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_aiobs_ai_HtpPerformanceManager_nativeAcquire(
        JNIEnv* env,
        jclass,
        jstring nativeLibDir) {

    logi(
            "nativeAcquire ENTER");

    if (env == nullptr ||
        nativeLibDir == nullptr) {

        loge(
                "nativeAcquire FAIL: null argument");

        return JNI_FALSE;
    }

    const char* chars =
            env->GetStringUTFChars(
                    nativeLibDir,
                    nullptr);

    if (chars == nullptr) {

        loge(
                "nativeAcquire FAIL: GetStringUTFChars");

        return JNI_FALSE;
    }

    const std::string path(
            chars);

    env->ReleaseStringUTFChars(
            nativeLibDir,
            chars);

    logi(
            "nativeAcquire nativeLibDir=%s",
            path.c_str());

    std::lock_guard<std::mutex> lock(
            gMutex);

    const bool ok =
            acquirePerformanceLocked(
                    path);

    logi(
            "nativeAcquire RETURN ok=%d active=%d",
            ok ? 1 : 0,
            gState.performanceVoteActive
            ? 1
            : 0);

    return ok
           ? JNI_TRUE
           : JNI_FALSE;
}

// ============================================================================
// JNI nativeRelease
// ============================================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_example_aiobs_ai_HtpPerformanceManager_nativeRelease(
        JNIEnv*,
        jclass) {

    logi(
            "nativeRelease ENTER");

    std::lock_guard<std::mutex> lock(
            gMutex);

    releasePerformanceLocked();

    logi(
            "nativeRelease RETURN");
}

// ============================================================================
// JNI nativeDescribe
// ============================================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_aiobs_ai_HtpPerformanceManager_nativeDescribe(
        JNIEnv* env,
        jclass) {

    if (env == nullptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(
            gMutex);

    const std::string description =
            describeLocked();

    return env->NewStringUTF(
            description.c_str());
}

// ============================================================================
// Compatibility alias: nativeEnable
//
// Allows Kotlin code using:
//
// external fun nativeEnable(
//     nativeLibDir: String
// ): Boolean
//
// to use the same implementation.
// ============================================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_aiobs_ai_HtpPerformanceManager_nativeEnable(
        JNIEnv* env,
        jclass clazz,
        jstring nativeLibDir) {

    return Java_com_example_aiobs_ai_HtpPerformanceManager_nativeAcquire(
            env,
            clazz,
            nativeLibDir);
}

// ============================================================================
// Compatibility alias: nativeDisable
// ============================================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_example_aiobs_ai_HtpPerformanceManager_nativeDisable(
        JNIEnv* env,
        jclass clazz) {

    Java_com_example_aiobs_ai_HtpPerformanceManager_nativeRelease(
            env,
            clazz);
}

// ============================================================================
// Compatibility alias: nativeSetPerformanceMode
//
// This accepts a mode integer but intentionally treats all supported modes
// as the global high-performance configuration.
//
// Current project goal:
//
//     normal application runtime = HTP performance vote ACTIVE
//
// Therefore the manager does not create per-model performance ownership.
// ============================================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_aiobs_ai_HtpPerformanceManager_nativeSetPerformanceMode(
        JNIEnv* env,
        jclass clazz,
        jstring nativeLibDir,
        jint mode) {

    (void)mode;

    logi(
            "nativeSetPerformanceMode mode=%d",
            static_cast<int>(mode));

    return Java_com_example_aiobs_ai_HtpPerformanceManager_nativeAcquire(
            env,
            clazz,
            nativeLibDir);
}

// ============================================================================
// JNI OnLoad
//
// No registration is required because the public JNI symbols above follow
// the normal Java_com_example_aiobs_ai_HtpPerformanceManager_* naming rule.
// ============================================================================

JNIEXPORT jint JNICALL
JNI_OnLoad(
        JavaVM* vm,
        void*) {

    if (vm == nullptr) {
        return JNI_ERR;
    }

    JNIEnv* env =
            nullptr;

    if (vm->GetEnv(
            reinterpret_cast<void**>(&env),
            JNI_VERSION_1_6) != JNI_OK) {

        return JNI_ERR;
    }

    logi(
            "JNI_OnLoad aiobs_htp_performance");

    return JNI_VERSION_1_6;
}