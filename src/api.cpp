/*
 Copyright 2016-2026 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#define CCL_C_API_EXPORT

#include "internal/api/comm.h"
#include "internal/api/plugin.h"
#include "internal/debug.h"
#include "internal/dlopen_compat.h"
#include "internal/utils_compat.h"
#include "oneapi/ccl.h"
#include "oneapi/ccl/v2/types.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
thread_local struct onecclThreadContext {
    bool init_done = false;
    onecclPluginType_t plugin_type;
    onecclPlugin_t plugin;
} thread_context;

std::shared_mutex plugin_lock;
std::unordered_map<std::string, dl_handle> library_cache;

// This a map is storing possible names for plugins used by oneCCL,
// The name `libexternal_Soneccl_Slibccl_Ulegacy` was added to support
// XLA and it's build system based on bazel.
const std::map<onecclPluginType_t, std::vector<std::string>> kPluginPaths = {
    {onecclLegacy,
     {"libccl_legacy.so", "libexternal_Soneccl_Slibccl_Ulegacy.so"}},
    {onecclLegacyCPU, {"libccl_legacy_cpu.so"}},
    {onecclNull, {"libccl_null.so"}}};

// Safe call which will never throw exceptions
template <typename Func, typename... Args>
onecclResult_t __safe_call_impl(const char *func_name, Func &&func,
                                Args &&...args) noexcept {
    try {
        return func(std::forward<Args>(args)...);
    } catch (const std::exception &ex) {
        ONECCL_ERROR("[%s] caught std::exception: %s", func_name, ex.what());
        return onecclPluginException;
    } catch (...) {
        ONECCL_ERROR("[%s] caught unknown exception", func_name);
        return onecclPluginException;
    }
}

#define SAFE_CALL(X, ...) __safe_call_impl(__func__, X, ##__VA_ARGS__)

#define ONECCL_VARNAME(x) #x

#define ONECCL_CHECK_PTR(x)                                                    \
    if ((x) == nullptr) {                                                      \
        ONECCL_WARN(ONECCL_VARNAME(x) " cannot be nullptr");                   \
        return onecclInvalidArgument;                                          \
    }

#define ONECCL_CHECK_MALLOC(x)                                                 \
    if ((x) == nullptr) {                                                      \
        ONECCL_ERROR("failed to allocate memory for " ONECCL_VARNAME(x));      \
        return onecclAllocFailureCPU;                                          \
    }

#ifndef BUILD_GIT_COMMIT
#define BUILD_GIT_COMMIT "unknown"
#endif
#ifndef BUILD_TIME_UTC
#define BUILD_TIME_UTC "unknown"
#endif
#ifndef BUILD_COMPILER_ID
#define BUILD_COMPILER_ID "unknown"
#endif
#ifndef BUILD_COMPILER_VERSION
#define BUILD_COMPILER_VERSION "unknown"
#endif
#ifndef BUILD_COMPILER_PATH
#define BUILD_COMPILER_PATH "unknown"
#endif
#ifndef BUILD_TYPE
#define BUILD_TYPE "unknown"
#endif
#ifndef BUILD_SYSTEM_NAME
#define BUILD_SYSTEM_NAME "unknown"
#endif
#ifndef BUILD_SYSTEM_PROCESSOR
#define BUILD_SYSTEM_PROCESSOR "unknown"
#endif

// Read-write access to plugin structures
template <typename Func, typename... Args>
onecclResult_t plugin_write(Func func, Args &&...args) {
    std::scoped_lock const lock(plugin_lock);
    return func(std::forward<Args>(args)...);
}

// Read-only access to plugin structures
template <typename Func, typename... Args>
onecclResult_t plugin_read(Func func, Args &&...args) {
    std::shared_lock<std::shared_mutex> const lock(plugin_lock);
    return func(std::forward<Args>(args)...);
}

std::string to_string(onecclPluginType_t type) {
    switch (type) {
    case onecclPluginAny:
        return "ONECCL_PLUGIN_ANY";
    case onecclLegacy:
        return "ONECCL_LEGACY";
    case onecclLegacyCPU:
        return "ONECCL_LEGACY_CPU";
    case onecclNull:
        return "ONECCL_NULL";
    default:
        return "UNKNOWN";
    }
}

onecclResult_t load_plugin(onecclPlugin_t *plugin, const std::string &path) {
    if (plugin == nullptr) {
        return onecclInvalidArgument;
    }

    dl_handle handle = nullptr;
    {
        // Attempt to read from the library cache using the path as the key
        std::shared_lock<std::shared_mutex> const lock(plugin_lock);
        auto cached_library = library_cache.find(path);
        if (cached_library != library_cache.end()) {
            handle = cached_library->second;
        }
    }

    if (handle == nullptr) {
        {
            std::scoped_lock const lock(plugin_lock);
            handle = dlopen(path.c_str(), RTLD_LAZY);
            if (handle == nullptr) {
                ONECCL_INFO("Cannot open plugin library: %s", dlerror());
                return onecclError;
            }
            // Cache the loaded library
            library_cache[path] = handle;
        }
    }

    dlerror(); // Clear any existing error

    // Get pointer the plugin entrypoint
    const char *symbol = "onecclPluginCall";
    auto plugin_call =
        reinterpret_cast<onecclPluginEntrypoint_t>(dlsym(handle, symbol));
    const char *dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        ONECCL_ERROR("Cannot load plugin entrypoint: '%s': %s", symbol,
                     dlsym_error);
        return onecclError;
    }

    plugin->call = plugin_call;

    auto plugin_init_fn = reinterpret_cast<onecclPluginCallInit_t>(
        plugin->call(ONECCL_PLUGIN_INIT));
    if (plugin_init_fn != nullptr) {
        return plugin_init_fn();
    }

    return onecclSuccess;
}

onecclResult_t init_library_with_override(const char *ccl_plugin) {
    ONECCL_INFO("Using CCL_PLUGIN override (%s)", ccl_plugin);
    std::string plugin_str(ccl_plugin);

    // Check if it matches any known plugin names
    for (const auto &[type, paths] : kPluginPaths) {
        if (plugin_str == to_string(type)) {
            for (const auto &path : paths) {
                onecclResult_t const result =
                    load_plugin(&thread_context.plugin, path);
                if (result == onecclSuccess) {
                    thread_context.plugin_type = type;
                    thread_context.init_done = true;
                    return onecclSuccess;
                }
            }
            break; // If loading fails, proceed to the default procedure
        }
    }

    // Check if it's a direct file path
    if (plugin_str.front() == '/' &&
        plugin_str.find("..") == std::string::npos) {
        onecclResult_t const result =
            load_plugin(&thread_context.plugin, plugin_str);
        if (result == onecclSuccess) {
            thread_context.plugin_type = onecclUserBackend;
            thread_context.init_done = true;
            return onecclSuccess;
        }
    }

    ONECCL_ERROR("oneCCL could not initialize plugin privided in CCL_PLUGIN.");
    return onecclError;
}

onecclResult_t init_library_without_override() {
    ONECCL_INFO("Proceeding without CCL_PLUGIN override.");
    int best_platform_score = INT_MIN;

    std::set<onecclPluginType_t> loaded_plugins;
    std::unordered_map<onecclPluginType_t, std::vector<onecclPluginType_t>>
        conflicts_map = {{onecclLegacy, {onecclLegacyCPU}},
                         {onecclLegacyCPU, {onecclLegacy}}};

    for (const auto &[type, paths] : kPluginPaths) {

        auto conflicts_with = conflicts_map.find(type);

        if (conflicts_with != conflicts_map.end()) {
            bool conflict_found = false;
            for (auto conflicting_plugin : conflicts_with->second) {
                if (loaded_plugins.find(conflicting_plugin) !=
                    loaded_plugins.end()) {
                    conflict_found = true;
                    break;
                }
            }

            if (conflict_found) {
                // We already loaded a plugin that is conflicting with plugin of
                // `type`, so we cannot load this one now.
                ONECCL_INFO("Failed to load plugin type: %s due to conflicts "
                            "with previously loaded plugins.",
                            to_string(type).c_str());
                continue;
            }
        }

        onecclPlugin plugin;
        onecclResult_t result = onecclError;
        for (const auto &path : paths) {
            result = load_plugin(&plugin, path);
            if (result == onecclSuccess) {
                break; // Successfully loaded, no need to try other paths
            }
        }
        if (result != onecclSuccess) {
            ONECCL_INFO("Failed to load plugin, type: %s",
                        to_string(type).c_str());
            continue;
        }

        loaded_plugins.insert(type); // Record that plugin of the type is loaded

        auto platform_score_func =
            reinterpret_cast<onecclPluginCallPlatformScore_t>(
                plugin.call(ONECCL_PLUGIN_PLATFORM_SCORE));
        if (platform_score_func == nullptr) {
            ONECCL_ERROR(
                "Plugin does not provide function to evaluate platform!");
            continue;
        }

        int platform_score = INT_MIN;
        result =
            SAFE_CALL([&] { return platform_score_func(&platform_score); });
        if (result != onecclSuccess) {
            ONECCL_INFO("Failed to evaluate plugin score!");
            continue;
        }

        ONECCL_INFO("Platform score for %s: %d", to_string(type).c_str(),
                    platform_score);
        if (platform_score > 0 && platform_score > best_platform_score) {
            thread_context.plugin = plugin;
            thread_context.plugin_type = type;
            thread_context.init_done = true;

            best_platform_score = platform_score;
        }
    }

    if (thread_context.plugin_type == onecclPluginAny) {
        ONECCL_ERROR("oneCCL could not be initialized! Could not load any "
                     "plugin. Please rerun with `CCL_LOG_LEVEL=info`.");
        return onecclError;
    }

    return onecclSuccess;
}

onecclResult_t init_library() {
    onecclDebugInit();
    if (thread_context.init_done) {
        return onecclSuccess;
    }

    ONECCL_INFO("\n  oneCCL V2 build info:\n"
                "    ONECCL_VERSION      : %d.%d.%d\n"
                "    GIT_COMMIT          : %s\n"
                "    BUILD_TIME_UTC      : %s\n"
                "    COMPILER_ID         : %s\n"
                "    COMPILER_VERSION    : %s\n"
                "    COMPILER_PATH       : %s\n"
                "    BUILD_TYPE          : %s\n"
                "    SYSTEM_NAME         : %s\n"
                "    SYSTEM_PROCESSOR    : %s\n",
                ONECCL_MAJOR, ONECCL_MINOR, ONECCL_PATCH, BUILD_GIT_COMMIT,
                BUILD_TIME_UTC, BUILD_COMPILER_ID, BUILD_COMPILER_VERSION,
                BUILD_COMPILER_PATH, BUILD_TYPE, BUILD_SYSTEM_NAME,
                BUILD_SYSTEM_PROCESSOR);

    // Try initializing with CCL_PLUGIN environment variable if set
    const char *ccl2_plugin = utils_getenv("CCL_PLUGIN");
    if (ccl2_plugin == nullptr) {
        return init_library_without_override();
    }

    return init_library_with_override(ccl2_plugin);
}

#define INIT_CCL                                                               \
    if (init_library() != onecclSuccess)                                       \
    return onecclError

onecclResult_t init_comm(onecclComm_t *comm, size_t nranks,
                         onecclUniqueId commId, int rank,
                         const onecclConfig_t *config) {
    auto init_comm_func = reinterpret_cast<onecclPluginCallCommInit_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_INIT_COMM));
    if (init_comm_func == nullptr) {
        return onecclNotImplemented;
    }

    if (comm == nullptr || config == nullptr) {
        return onecclInvalidArgument;
    }

    auto *new_comm =
        static_cast<onecclComm_t>(calloc(sizeof(struct onecclComm), 1));
    ONECCL_CHECK_MALLOC(new_comm);

    new_comm->id = commId;
    new_comm->config = *config;

    auto result = SAFE_CALL([&] {
        return init_comm_func(&new_comm, nranks, commId, rank, config);
    });
    if (result != onecclSuccess) {
        std::cerr << "Plugin could not initialize communicator!" << '\n';
        free(new_comm);
        return result;
    }

    *comm = new_comm;
    return onecclSuccess;
}

} // namespace

onecclResult_t onecclGetVersion(int *version) {
    INIT_CCL;

    *version = ONECCL_VERSION(ONECCL_MAJOR, ONECCL_MINOR, ONECCL_PATCH);
    return onecclSuccess;
}

onecclResult_t CCL_C_API onecclExtractVersionComponents(int versionCode,
                                                        int *major, int *minor,
                                                        int *patch) {
    INIT_CCL;

    *major = versionCode / 10000;
    *minor = (versionCode / 100) % 100;
    *patch = versionCode % 100;
    return onecclSuccess;
}

const char CCL_C_API *onecclGetErrorString(onecclResult_t result) {
    switch (result) {
    // Don't be afraid to add new error types.
    // The more specific error is the better for the user.
    case onecclSuccess:
        return "Success";
    case onecclError:
        return "Generic error";
    case onecclSystemError:
        return "System error";
    case onecclInternalError:
        return "Internal error";
    case onecclInvalidArgument:
        return "Invalid argument";
    case onecclInvalidUsage:
        return "Invalid usage";
    case onecclInProgress:
        return "Operation in progress";
    case onecclFailureGPU:
        return "GPU failure";
    case onecclFailureCPU:
        return "CPU failure";
    case onecclNotImplemented:
        return "Not implemented";
    case onecclAllocFailureCPU:
        return "Could not allocate memory on the CPU";
    case onecclAllocFailureGPU:
        return "Could not allocate memory on the GPU";
    default:
        return "Unknown error code";
    }
}

const char *onecclGetLastError(onecclComm_t /*comm*/) {
    return onecclGetLastErrorString();
}

onecclResult_t onecclGetUniqueId(onecclUniqueId *uniqueId) {
    INIT_CCL;

    auto init_id_func = reinterpret_cast<onecclPluginCallInitId_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_INIT_ID));
    if (init_id_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclGetUniqueId (INIT_ID)");
        return onecclNotImplemented;
    }

    ONECCL_CHECK_PTR(uniqueId);

    return SAFE_CALL([=]() { return init_id_func(uniqueId); });
}

onecclResult_t onecclCommInitRank(onecclComm_t *comm, size_t nranks,
                                  onecclUniqueId commId, int rank) {
    INIT_CCL;

    onecclConfig_t const config{};
    return onecclCommInitRankConfig(comm, nranks, commId, rank, &config);
}

onecclResult_t onecclCommInitRankConfig(onecclComm_t *comm, size_t nranks,
                                        onecclUniqueId commId, int rank,
                                        const onecclConfig_t *config) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    return init_comm(comm, nranks, commId, rank, config);
}

onecclResult_t onecclCommDestroy(onecclComm_t comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->destroy == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommDestroy");
        return onecclNotImplemented;
    }

    onecclResult_t const result =
        SAFE_CALL([=] { return comm->destroy(comm); });
    if (result != onecclSuccess) {
        return result;
    }

    free(comm);
    return onecclSuccess;
}

onecclResult_t onecclCommSplit(onecclComm_t comm, int color, int key,
                               onecclComm_t *newcomm, onecclConfig_t *config) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->split == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommSplit");
        return onecclNotImplemented;
    }

    // If no config was specified use one from the base comm
    if (config == nullptr) {
        config = &comm->config;
    }

    *newcomm = static_cast<onecclComm_t>(calloc(sizeof(struct onecclComm), 1));
    ONECCL_CHECK_PTR(newcomm);

    (*newcomm)->config = *config;

    if (comm->destroy != nullptr) {
        auto result = SAFE_CALL(
            [=] { return comm->split(comm, color, key, newcomm, config); });
        if (result != onecclSuccess) {
            free(*newcomm);
            *newcomm = nullptr;
            return result;
        }
    }
    return onecclSuccess;
}

onecclResult_t onecclCommUserRank(onecclComm_t comm, int *rank) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    ONECCL_CHECK_PTR(rank);

    if (comm->get_rank == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommUserRank");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return comm->get_rank(comm, rank); });
}

onecclResult_t onecclCommCount(onecclComm_t comm, int *size) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    ONECCL_CHECK_PTR(size);

    if (comm->get_size == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommCount");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=] { return comm->get_size(comm, size); });
}

onecclResult_t onecclAllReduce(void *sendbuff, void *recvbuff, size_t count,
                               onecclDataType_t datatype,
                               onecclRedOp_t reduction_op, onecclComm_t comm,
                               void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->allreduce == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclAllReduce");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->allreduce(sendbuff, recvbuff, count, datatype,
                               reduction_op, comm, stream);
    });
}

onecclResult_t onecclAllGather(const void *sendbuff, void *recvbuff,
                               size_t sendcount, onecclDataType_t datatype,
                               onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->allgather == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclAllGather");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->allgather(sendbuff, recvbuff, sendcount, datatype, comm,
                               stream);
    });
}

onecclResult_t CCL_C_API onecclAllToAll(const void *sendbuff, void *recvbuff,
                                        size_t count, onecclDataType_t datatype,
                                        onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->allgather == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclAllToAll");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->alltoall(sendbuff, recvbuff, count, datatype, comm,
                              stream);
    });
}

onecclResult_t onecclBroadcast(const void *sendbuff, void *recvbuff,
                               size_t count, onecclDataType_t datatype,
                               int root, onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->broadcast == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclBroadcast");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->broadcast(sendbuff, recvbuff, count, datatype, root, comm,
                               stream);
    });
}

onecclResult_t onecclReduce(const void *sendbuff, void *recvbuff, size_t count,
                            onecclDataType_t datatype, onecclRedOp_t redop,
                            int root, onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->reduce == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclReduce");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() {
        return comm->reduce(sendbuff, recvbuff, count, datatype, redop, root,
                            comm, stream);
    });
}

onecclResult_t onecclReduceScatter(const void *sendbuff, void *recvbuff,
                                   size_t recvcount, onecclDataType_t datatype,
                                   onecclRedOp_t redop, onecclComm_t comm,
                                   void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->reduce_scatter == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclReduceScatter");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() {
        return comm->reduce_scatter(sendbuff, recvbuff, recvcount, datatype,
                                    redop, comm, stream);
    });
}

onecclResult_t onecclSend(const void *sendbuff, size_t count,
                          onecclDataType_t datatype, int peer,
                          onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->send == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclSend");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() {
        return comm->send(sendbuff, count, datatype, peer, comm, stream);
    });
}

onecclResult_t onecclRecv(void *recvbuff, size_t count,
                          onecclDataType_t datatype, int peer,
                          onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->recv == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclRecv");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() {
        return comm->recv(recvbuff, count, datatype, peer, comm, stream);
    });
}

onecclResult_t onecclGroupStart() {
    INIT_CCL;

    auto group_start_func = reinterpret_cast<onecclPluginCallGroupStart_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_GROUP_START));
    if (group_start_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclGroupStart");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return group_start_func(); });
}

onecclResult_t onecclGroupEnd() {
    INIT_CCL;

    auto group_end_func = reinterpret_cast<onecclPluginCallGroupEnd_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_GROUP_END));
    if (group_end_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclGroupEnd");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return group_end_func(); });
}

onecclResult_t onecclCommFinalize(onecclComm_t comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->finalize == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommFinalize");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=] { return comm->finalize(comm); });
}

onecclResult_t onecclCommAbort(onecclComm_t comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->abort == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommAbort");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=] { return comm->abort(comm); });
}

onecclResult_t onecclSetDevice(uint32_t index) {
    INIT_CCL;

    auto init_device_func = reinterpret_cast<onecclPluginCallDeviceInit_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_INIT_DEVICE));
    if (init_device_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclSetDevice (INIT_DEVICE)");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return init_device_func(index); });
}

onecclResult_t onecclMemAlloc(void **ptr, size_t size) {
    INIT_CCL;

    ONECCL_CHECK_PTR(ptr);

    auto mem_alloc_func = reinterpret_cast<onecclPluginCallMemAlloc_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_MEM_ALLOC));
    if (mem_alloc_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclMemAlloc");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() { return mem_alloc_func(ptr, size); });
}

onecclResult_t onecclMemFree(void *ptr) {
    INIT_CCL;

    ONECCL_CHECK_PTR(ptr);

    auto mem_free_func = reinterpret_cast<onecclPluginCallMemFree_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_MEM_FREE));
    if (mem_free_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclMemFree");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() { return mem_free_func(ptr); });
}

onecclResult_t onecclCommRegister(onecclComm_t comm, void *buff, size_t size,
                                  void **handle) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    ONECCL_CHECK_PTR(buff);
    ONECCL_CHECK_PTR(handle);

    if (comm->comm_register == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommRegister");
        return onecclNotImplemented;
    }

    return SAFE_CALL(
        [=]() { return comm->comm_register(comm, buff, size, handle); });
}

onecclResult_t onecclCommDeregister(onecclComm_t comm, void *handle) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->comm_deregister == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommDeregister");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() { return comm->comm_deregister(comm, handle); });
}

onecclResult_t onecclCommWindowRegister(onecclComm_t comm, void *buff,
                                        size_t size, onecclWindow_t *window,
                                        onecclWindowFlags_t flags) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    ONECCL_CHECK_PTR(buff);
    ONECCL_CHECK_PTR(window);

    if (comm->comm_window_register == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommWindowRegister");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->comm_window_register(comm, buff, size, window, flags);
    });
}

onecclResult_t onecclCommWindowDeregister(onecclComm_t comm,
                                          onecclWindow_t window) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    // Allow nullptr window as a no-op, consistent with onecclCommDeregister

    if (comm->comm_window_deregister == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommWindowDeregister");
        return onecclNotImplemented;
    }

    return SAFE_CALL(
        [=]() { return comm->comm_window_deregister(comm, window); });
}

onecclResult_t onecclCommDevice(onecclComm_t comm, int *device) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->create_pre_mul_sum == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommDevice");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return comm->get_device(comm, device); });
}

onecclResult_t onecclRedOpCreatePreMulSum(onecclRedOp_t *redop, void *scalar,
                                          onecclDataType_t datatype,
                                          onecclScalarResidence_t residence,
                                          onecclComm_t comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->create_pre_mul_sum == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclRedOpCreatePreMulSum");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() {
        return comm->create_pre_mul_sum(redop, scalar, datatype, residence,
                                        comm);
    });
}

onecclResult_t onecclRedOpDestroy(onecclRedOp_t redop, onecclComm_t comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->reduction_destroy == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclRedOpDestroy");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return comm->reduction_destroy(redop, comm); });
}
