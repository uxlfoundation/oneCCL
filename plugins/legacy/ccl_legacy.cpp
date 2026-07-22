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

// plugins/legacy/ccl_legacy.cpp
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <sys/mman.h>
#include <utility>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "internal/api/comm.h"
#include "internal/api/plugin.h"
#include "internal/debug.h"
#include "internal/itt_wrapper.hpp"
#include "oneapi/ccl.h"
#include "oneapi/ccl.hpp"
#include "oneapi/ccl/aliases.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/v2/types.h"

namespace {
std::map<ccl::kvs::address_type, ccl::shared_ptr_class<ccl::kvs>> kvs_for;
std::mutex kvs_for_mutex;
thread_local std::optional<int> selected_device_index;
thread_local std::optional<sycl::device> selected_device;
thread_local std::optional<sycl::context> selected_context;
thread_local std::optional<ccl::stream> default_stream;

using onecclUniqueIdLegacy = struct onecclUniqueIdLegacy {
    onecclPluginType_t magic;
    ccl::kvs::address_type address;
};

class CommunicatorLegacy {
  public:
    int rank;
    int size;
    std::optional<int> device_index;
    std::optional<ccl::device> device;
    std::optional<ccl::context> context;
    ccl::communicator comm;

    // Constructor using initializer list
    CommunicatorLegacy(int rank, int size, ccl::shared_ptr_class<ccl::kvs> kvs,
                       int device_index, ccl::device in_device,
                       ccl::context in_context, bool is_ext)
        : rank(rank), size(size), device_index(device_index),
          device(std::move(in_device)), context(std::move(in_context)),
          comm(is_ext ? std::move(ccl::create_communicatorExt(
                            size, rank, *device, *context, kvs))
                      : std::move(ccl::create_communicator(size, rank, *device,
                                                           *context, kvs))) {}

    // Constructor using initializer list
    CommunicatorLegacy(int rank, int size, ccl::shared_ptr_class<ccl::kvs> kvs)
        : rank(rank), size(size),
          comm(std::move(ccl::create_communicator(size, rank, kvs))) {}

    // Construction using external ccl::communicator, for example as a result of
    // a split
    explicit CommunicatorLegacy(ccl::communicator &&comm)
        : rank(comm.rank()), size(comm.size()), comm(std::move(comm)) {}
};

onecclResult_t oneccl_finalize_communicator_impl(onecclComm_t comm) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    comm_legacy->comm.finalize();
    return onecclSuccess;
}

onecclResult_t oneccl_destroy_communicator_impl(onecclComm_t comm) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    delete comm_legacy;
    return onecclSuccess;
}

onecclResult_t oneccl_get_rank_impl(onecclComm_t comm, int *out_rank) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    *out_rank = comm_legacy->rank;
    return onecclSuccess;
}

onecclResult_t oneccl_get_local_rank_impl(onecclComm_t comm, int *local_rank) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    *local_rank = comm_legacy->rank;
    return onecclSuccess;
}

onecclResult_t oneccl_get_size_impl(onecclComm_t comm, int *out_size) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    *out_size = comm_legacy->size;
    return onecclSuccess;
}

onecclResult_t oneccl_get_local_size_impl(onecclComm_t comm, int *local_size) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    *local_size = comm_legacy->size;
    return onecclSuccess;
}

ccl::reduction convert(onecclRedOp_t red_op) {
    switch (red_op) {
    case onecclSum:
        return ccl::reduction::sum;
    case onecclProd:
        return ccl::reduction::prod;
    case onecclMin:
        return ccl::reduction::min;
    case onecclMax:
        return ccl::reduction::max;
    case onecclAvg:
        return ccl::reduction::avg;
    default:
        // support for user-defined reduction
        return static_cast<ccl::reduction>(red_op);
    }
}

ccl::scalar_residence_type convert(onecclScalarResidence_t residence) {
    switch (residence) {
    case onecclScalarHostImmediate:
        return ccl::scalar_residence_type::scalar_host_immediate;
    case onecclScalarDevice:
        return ccl::scalar_residence_type::scalar_device;
    default:
        throw std::invalid_argument(
            "Unsupported onecclScalarResidence_t value");
    }
}

ccl::datatype convert(onecclDataType_t datatype) {
    switch (datatype) {
    case onecclInt8:
        return ccl::datatype::int8;
    case onecclUint8:
        return ccl::datatype::uint8;
    case onecclInt32:
        return ccl::datatype::int32;
    case onecclUint32:
        return ccl::datatype::uint32;
    case onecclInt64:
        return ccl::datatype::int64;
    case onecclUint64:
        return ccl::datatype::uint64;
    case onecclFloat16:
        return ccl::datatype::float16;
    case onecclFloat32:
        return ccl::datatype::float32;
    case onecclFloat64:
        return ccl::datatype::float64;
    case onecclBfloat16:
        return ccl::datatype::bfloat16;
    default:
        throw std::invalid_argument("Unsupported onecclDataType_t value");
    }
}

bool is_host_pointer(const void *ptr) {
    long const page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        throw std::runtime_error("Failed to get page size from sysconf");
    }

    auto page_size = static_cast<size_t>(page_size_long);
    auto address = reinterpret_cast<uintptr_t>(ptr);
    size_t const page_count = address / static_cast<uintptr_t>(page_size);

    void *page_aligned_ptr = reinterpret_cast<void *>(
        page_count * static_cast<uintptr_t>(page_size));

    return msync(page_aligned_ptr, page_size, MS_ASYNC) == 0;
}

onecclResult_t execute_collective(
    const void *buf, onecclComm_t comm, void *stream,
    const std::function<void(CommunicatorLegacy *comm_legacy,
                             sycl::queue *sycl_queue)> &gpu_collective,
    const std::function<ccl::event(CommunicatorLegacy *comm_legacy)>
        &cpu_collective) {

    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    if (comm_legacy == nullptr) {
        return onecclInvalidArgument;
    }

    auto cpu_buffer = is_host_pointer(buf);

    if (stream == nullptr) {

        auto event = cpu_collective(comm_legacy);
        event.wait();

        return onecclSuccess;
    }

    auto *queue = static_cast<sycl::queue *>(stream);

    if (!queue->is_in_order()) {
        ONECCL_ERROR(
            "oneCCL collective was called using out of order queue which is "
            "not supported! Please use in order queues for correct results "
            "(For example: `sycl::queue "
            "queue(sycl::property::queue::in_order{});`).");
    }

    if (!cpu_buffer) {
        gpu_collective(comm_legacy, queue);
    } else {
        // TODO: This `wait` is a workaround for
        // problems with calling oneCCL API inside sycl host task
        queue->wait();
        auto event = std::make_shared<ccl::event>(cpu_collective(comm_legacy));
        queue->submit([=](sycl::handler &cgh) {
            cgh.host_task([=]() { event->wait(); });
        });
    }

    if (comm->config.blocking == 1) {
        queue->wait();
    }

    return onecclSuccess;
}

std::vector<ccl::event> get_deps(sycl::queue * /*sycl_queue*/) {
    std::vector<ccl::event> deps;
    // deps.reserve(sycl_queue->deps.size());
    // for (auto &event : sycl_ext->deps) {
    //     deps.push_back(ccl::create_event(event));
    // }

    return deps;
}

ccl::stream &get_stream(sycl::queue *sycl_queue) {
    thread_local std::unordered_map<sycl::queue, ccl::stream> stream_map;

    auto streams_iterator = stream_map.find(*sycl_queue);
    if (streams_iterator == stream_map.end()) {
        ccl::stream new_stream = ccl::create_stream(*sycl_queue);
        auto result = stream_map.emplace(*sycl_queue, std::move(new_stream));
        streams_iterator = result.first;
    }

    return streams_iterator->second;
}

onecclResult_t oneccl_allreduce_impl(void *sendbuff, void *recvbuff,
                                     size_t count, onecclDataType_t datatype,
                                     onecclRedOp_t red_op, onecclComm_t comm,
                                     void *stream) {
    return execute_collective(
        sendbuff, comm, stream,
        [=](CommunicatorLegacy *comm_legacy, sycl::queue *sycl_ext) {
            auto attrs = ccl::create_operation_attr<ccl::allreduce_attr>();

            static itt::Task allreduce_s("CPU::AllReduce::Submit");

            allreduce_s.start();
            ccl::allreduce(sendbuff, recvbuff, count, convert(datatype),
                           convert(red_op), comm_legacy->comm,
                           get_stream(sycl_ext), attrs, get_deps(sycl_ext));
            allreduce_s.end();
        },
        [=](CommunicatorLegacy *comm_legacy) {
            return ccl::allreduce(sendbuff, recvbuff, count, convert(datatype),
                                  convert(red_op), comm_legacy->comm);
        });
}

onecclResult_t oneccl_allgather_impl(const void *sendbuff, void *recvbuff,
                                     size_t sendcount,
                                     onecclDataType_t datatype,
                                     onecclComm_t comm, void *stream) {
    return execute_collective(
        sendbuff, comm, stream,
        [=](CommunicatorLegacy *comm_legacy, sycl::queue *sycl_ext) {
            auto attrs = ccl::create_operation_attr<ccl::allgatherv_attr>();
            size_t const comm_size = comm_legacy->comm.size();
            std::vector<size_t> const recvcounts(comm_size, sendcount);

            ccl::allgatherv(sendbuff, sendcount, recvbuff, recvcounts,
                            convert(datatype), comm_legacy->comm,
                            get_stream(sycl_ext), attrs, get_deps(sycl_ext));
        },
        [=](CommunicatorLegacy *comm_legacy) {
            size_t const comm_size = comm_legacy->comm.size();
            std::vector<size_t> const recvcounts(comm_size, sendcount);

            return ccl::allgatherv(sendbuff, sendcount, recvbuff, recvcounts,
                                   convert(datatype), comm_legacy->comm);
        });
}

onecclResult_t oneccl_alltoall_impl(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, onecclDataType_t datatype,
                                    onecclComm_t comm, void *stream) {
    return execute_collective(
        sendbuff, comm, stream,
        [=](CommunicatorLegacy *comm_legacy, sycl::queue *sycl_ext) {
            auto attrs = ccl::create_operation_attr<ccl::alltoall_attr>();

            ccl::alltoall(sendbuff, recvbuff, sendcount, convert(datatype),
                          comm_legacy->comm, get_stream(sycl_ext), attrs,
                          get_deps(sycl_ext));
        },
        [=](CommunicatorLegacy *comm_legacy) {
            return ccl::alltoall(sendbuff, recvbuff, sendcount,
                                 convert(datatype), comm_legacy->comm);
        });
}

onecclResult_t oneccl_broadcast_impl(const void *sendbuff, void *recvbuff,
                                     size_t count, onecclDataType_t datatype,
                                     int root, onecclComm_t comm,
                                     void *stream) {
    return execute_collective(
        sendbuff, comm, stream,
        [=](CommunicatorLegacy *comm_legacy, sycl::queue *sycl_ext) {
            auto attrs = ccl::create_operation_attr<ccl::broadcast_attr>();

            ccl::broadcast(const_cast<void *>(sendbuff), recvbuff, count,
                           convert(datatype), root, comm_legacy->comm,
                           get_stream(sycl_ext), attrs, get_deps(sycl_ext));
        },
        [=](CommunicatorLegacy *comm_legacy) {
            return ccl::broadcast(const_cast<void *>(sendbuff), recvbuff, count,
                                  convert(datatype), root, comm_legacy->comm);
        });
}

onecclResult_t oneccl_reduce_impl(const void *sendbuff, void *recvbuff,
                                  size_t count, onecclDataType_t datatype,
                                  onecclRedOp_t reduction_op, int root,
                                  onecclComm_t comm, void *stream) {
    return execute_collective(
        sendbuff, comm, stream,
        [=](CommunicatorLegacy *comm_legacy, sycl::queue *sycl_ext) {
            auto attrs = ccl::create_operation_attr<ccl::reduce_attr>();

            ccl::reduce(sendbuff, recvbuff, count, convert(datatype),
                        convert(reduction_op), root, comm_legacy->comm,
                        get_stream(sycl_ext), attrs, get_deps(sycl_ext));
        },
        [=](CommunicatorLegacy *comm_legacy) {
            return ccl::reduce(sendbuff, recvbuff, count, convert(datatype),
                               convert(reduction_op), root, comm_legacy->comm);
        });
}

onecclResult_t oneccl_reduce_scatter_impl(const void *sendbuff, void *recvbuff,
                                          size_t recvcount,
                                          onecclDataType_t datatype,
                                          onecclRedOp_t redop,
                                          onecclComm_t comm, void *stream) {
    return execute_collective(
        sendbuff, comm, stream,
        [=](CommunicatorLegacy *comm_legacy, sycl::queue *sycl_ext) {
            auto attrs = ccl::create_operation_attr<ccl::reduce_scatter_attr>();

            ccl::reduce_scatter(sendbuff, recvbuff, recvcount,
                                convert(datatype), convert(redop),
                                comm_legacy->comm, get_stream(sycl_ext), attrs,
                                get_deps(sycl_ext));
        },
        [=](CommunicatorLegacy *comm_legacy) {
            return ccl::reduce_scatter(sendbuff, recvbuff, recvcount,
                                       convert(datatype), convert(redop),
                                       comm_legacy->comm);
        });
}

onecclResult_t oneccl_send_impl(const void *sendbuff, size_t count,
                                onecclDataType_t datatype, int peer,
                                onecclComm_t comm, void *stream) {
    return execute_collective(
        sendbuff, comm, stream,
        [=](CommunicatorLegacy *comm_legacy, sycl::queue *sycl_ext) {
            auto attrs = ccl::create_operation_attr<ccl::pt2pt_attr>();

            ccl::send(const_cast<void *>(sendbuff), count, convert(datatype),
                      peer, comm_legacy->comm, get_stream(sycl_ext), attrs,
                      get_deps(sycl_ext));
        },
        [=](CommunicatorLegacy *comm_legacy) {
            return ccl::send(const_cast<void *>(sendbuff), count,
                             convert(datatype), peer, comm_legacy->comm);
        });
}

onecclResult_t oneccl_recv_impl(void *recvbuff, size_t count,
                                onecclDataType_t datatype, int peer,
                                onecclComm_t comm, void *stream) {
    return execute_collective(
        recvbuff, comm, stream,
        [=](CommunicatorLegacy *comm_legacy, sycl::queue *sycl_ext) {
            auto attrs = ccl::create_operation_attr<ccl::pt2pt_attr>();

            ccl::recv(recvbuff, count, convert(datatype), peer,
                      comm_legacy->comm, get_stream(sycl_ext), attrs,
                      get_deps(sycl_ext));
        },
        [=](CommunicatorLegacy *comm_legacy) {
            return ccl::recv(recvbuff, count, convert(datatype), peer,
                             comm_legacy->comm);
        });
}

onecclResult_t oneccl_barrier_impl(onecclComm_t comm) {
    return execute_collective(
        nullptr, comm, nullptr,
        [=](CommunicatorLegacy *comm_legacy, sycl::queue * /*sycl_ext*/) {
            ccl::barrier(comm_legacy->comm);
        },
        [=](CommunicatorLegacy *comm_legacy) {
            return ccl::barrier(comm_legacy->comm);
        });
}

onecclResult_t oneccl_comm_split_impl(onecclComm_t comm, int color, int key,
                                      onecclComm_t *newcomm,
                                      onecclConfig_t *config) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);

    if (color == INT_MAX) {
        ONECCL_ERROR(
            "oneCCL legacy plugin does not support color=%d, it is reserved "
            "for internal purposes. Please use a different color",
            INT_MAX);
        return onecclInvalidArgument;
    }

    bool no_color = false;
    if (color == ONECCL_SPLIT_NOCOLOR) {
        // This is a workaround because legacy ccl does not support such feature
        color = INT_MAX;
        no_color = true;
    }

    auto *new_comm_legacy = new CommunicatorLegacy(
        ccl::split_communicator(comm_legacy->comm, color, key));

    if (no_color) {
        // If user supplied ONECCL_SPLIT_NOCOLOR, we should delete all the
        // structures and return nullptr
        delete new_comm_legacy;
        free(*newcomm);
        (*newcomm) = nullptr;

        return onecclSuccess;
    }

    (**newcomm) = (*comm); // Copy all fields from the base comm
    (*newcomm)->pExt = new_comm_legacy;
    (*newcomm)->config = *config;
    return onecclSuccess;
}

onecclResult_t oneccl_plugin_init() { return onecclSuccess; }

onecclResult_t oneccl_get_unique_id_impl(onecclUniqueId *id_ptr) {
    auto *legacy_id = reinterpret_cast<onecclUniqueIdLegacy *>(&id_ptr->legacy);

    legacy_id->magic = onecclLegacy;

    auto kvs = ccl::create_main_kvs();
    auto address = kvs->get_address();

    legacy_id->address = address;
    {
        std::lock_guard<std::mutex> lock(kvs_for_mutex);
        kvs_for[address] = kvs;
    }

    return onecclSuccess;
}

onecclResult_t oneccl_set_device_impl(uint32_t index) {
    auto platforms = sycl::platform::get_platforms();
    sycl::platform l0_platform;
    bool l0_found = false;

    // Find platform with Level-Zero
    for (const auto &platform : platforms) {
        if (platform.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            l0_platform = platform;
            l0_found = true;
            break;
        }
    }

    if (!l0_found) {
        ONECCL_ERROR("Could not find level-zero platform in "
                     "sycl::platform::get_platforms()! Check if your system "
                     "has level-zero devices avilable (Use `SYCL_UR_TRACE=1 "
                     "sycl-ls` for debug).");
        return onecclError;
    }

    // Create a queue from the device on the Level-Zero platform
    selected_device_index = index;
    selected_device = l0_platform.get_devices()[index];
#if defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER < 20250200)
    // Older Intel compiler uses ext_oneapi prefix
    selected_context = l0_platform.ext_oneapi_get_default_context();
#else
    // Newer Intel or open-source compilers use khr prefix
    selected_context = l0_platform.khr_get_default_context();
#endif

    auto default_queue = sycl::queue(*selected_context, *selected_device,
                                     {sycl::property::queue::in_order{}});
    default_stream = ccl::create_stream(default_queue);

    return onecclSuccess;
}

onecclResult_t oneccl_get_device_impl(onecclComm_t comm, int *index) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);

    if (!comm_legacy->device_index) {
        ONECCL_WARN("This communicator was not assigned any device. Use "
                    "onecclSetDevice before creating communicator.");
        return onecclInvalidUsage;
    }

    *index = *comm_legacy->device_index;
    return onecclSuccess;
}

onecclResult_t oneccl_platform_score_impl(int *score) {
    int const new_score = 0;

    auto platforms = sycl::platform::get_platforms();
    sycl::platform l0_platform;
    bool l0_found = false;

    // Find platform with Level-Zero
    for (const auto &platform : platforms) {
        if (platform.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            l0_platform = platform;
            l0_found = true;
            break;
        }
    }

    if (!l0_found) {
        *score = -1000;
    } else {
        *score = static_cast<int>(l0_platform.get_devices().size()) * 10000;
    }

    return onecclSuccess;
}

onecclResult_t oneccl_mem_alloc_impl(void **ptr, size_t size) {
    if (ptr == nullptr) {
        return onecclInvalidArgument;
    }

    if (!(selected_device && selected_context)) {
        ONECCL_WARN("Device is not selected. Call onecclSetDevice before "
                    "onecclMemAlloc.");
        return onecclInvalidUsage;
    }

    ccl::mem_alloc(*default_stream, size, ptr);

    return onecclSuccess;
}

onecclResult_t oneccl_mem_free_impl(void *ptr) {
    if (ptr == nullptr) {
        return onecclInvalidArgument;
    }

    if (!(selected_device && selected_context)) {
        ONECCL_WARN("Device is not selected. Call onecclSetDevice before "
                    "onecclMemFree.");
        return onecclInvalidUsage;
    }

    ccl::mem_free(*default_stream, ptr);

    return onecclSuccess;
}

onecclResult_t oneccl_comm_register_impl(onecclComm_t comm, void *buff,
                                         size_t size, void **handle) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    if (comm_legacy == nullptr || buff == nullptr) {
        return onecclInvalidArgument;
    }

    ccl::comm_register(comm_legacy->comm, buff, size, handle);

    return onecclSuccess;
}

onecclResult_t oneccl_comm_deregister_impl(onecclComm_t comm, void *handle) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    if (comm_legacy == nullptr) {
        return onecclInvalidArgument;
    }

    ccl::comm_deregister(comm_legacy->comm, handle);
    return onecclSuccess;
}

onecclResult_t oneccl_comm_window_register_impl(onecclComm_t comm, void *buff,
                                                size_t size,
                                                onecclWindow_t *window,
                                                onecclWindowFlags_t flags) {
    if (comm == nullptr || buff == nullptr || window == nullptr) {
        return onecclInvalidArgument;
    }

    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    if (comm_legacy == nullptr) {
        return onecclInvalidArgument;
    }

    ccl::window window_obj = ccl::comm_window_register(
        comm_legacy->comm, buff, size, static_cast<int>(flags));
    *window = new ccl::window(std::move(window_obj));

    return onecclSuccess;
}

onecclResult_t oneccl_comm_window_deregister_impl(onecclComm_t comm,
                                                  onecclWindow_t window) {
    if (comm == nullptr || window == nullptr) {
        return onecclInvalidArgument;
    }

    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    if (comm_legacy == nullptr) {
        return onecclInvalidArgument;
    }

    auto *window_obj = static_cast<ccl::window *>(window);
    ccl::comm_window_deregister(comm_legacy->comm, *window_obj);
    delete window_obj;

    return onecclSuccess;
}

onecclResult_t oneccl_create_pre_mul_sum_impl(onecclRedOp_t *redop,
                                              void *scalar,
                                              onecclDataType_t datatype,
                                              onecclScalarResidence_t residence,
                                              onecclComm_t comm) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    ccl::reduction custom_redop{};
    ccl::reduction_create_pre_mul_sum(&custom_redop, scalar, convert(datatype),
                                      convert(residence), comm_legacy->comm);
    *redop = static_cast<onecclRedOp_t>(custom_redop);
    return onecclSuccess;
}

onecclResult_t oneccl_reduction_destroy_impl(onecclRedOp_t redop,
                                             onecclComm_t comm) {
    auto *comm_legacy = static_cast<CommunicatorLegacy *>(comm->pExt);
    ccl::reduction_destroy(convert(redop), comm_legacy->comm);
    return onecclSuccess;
}

onecclResult_t oneccl_init_communicator_impl(onecclComm_t *comm, size_t nranks,
                                             onecclUniqueId commId, int rank,
                                             const onecclConfig_t *config) {
    auto *legacy_id = reinterpret_cast<onecclUniqueIdLegacy *>(&commId.legacy);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    {
        std::lock_guard<std::mutex> lock(kvs_for_mutex);
        if (kvs_for.find(legacy_id->address) == kvs_for.end()) {
            kvs_for[legacy_id->address] =
                ccl::create_kvs(legacy_id->address); // Create missing kvs
        }
        kvs = kvs_for[legacy_id->address];
    }

    CommunicatorLegacy *comm_legacy = nullptr; // NOLINT(misc-const-correctness)
    if ((selected_device && selected_context && selected_device_index)) {
        // SYCL path
        int const device_index = *selected_device_index;

        auto multi_threaded = config->multiThreaded == 1;
        comm_legacy = new CommunicatorLegacy(
            rank, static_cast<int>(nranks), kvs, device_index,
            std::move(ccl::create_device(*selected_device)),
            std::move(ccl::create_context(*selected_context)), multi_threaded);
    } else {
        // CPU only path
        comm_legacy =
            new CommunicatorLegacy(rank, static_cast<int>(nranks), kvs);
    }

    (*comm)->pExt = comm_legacy;
    (*comm)->finalize = oneccl_finalize_communicator_impl;
    (*comm)->destroy = oneccl_destroy_communicator_impl;
    (*comm)->allreduce = oneccl_allreduce_impl;
    (*comm)->allgather = oneccl_allgather_impl;
    (*comm)->alltoall = oneccl_alltoall_impl;
    (*comm)->broadcast = oneccl_broadcast_impl;
    (*comm)->reduce = oneccl_reduce_impl;
    (*comm)->reduce_scatter = oneccl_reduce_scatter_impl;
    (*comm)->send = oneccl_send_impl;
    (*comm)->recv = oneccl_recv_impl;
    (*comm)->barrier = oneccl_barrier_impl;
    (*comm)->split = oneccl_comm_split_impl;
    (*comm)->get_rank = oneccl_get_rank_impl;
    (*comm)->get_local_rank = oneccl_get_local_rank_impl;
    (*comm)->get_size = oneccl_get_size_impl;
    (*comm)->get_local_size = oneccl_get_local_size_impl;
    (*comm)->get_device = oneccl_get_device_impl;
    (*comm)->comm_register = oneccl_comm_register_impl;
    (*comm)->comm_deregister = oneccl_comm_deregister_impl;
    (*comm)->comm_window_register = oneccl_comm_window_register_impl;
    (*comm)->comm_window_deregister = oneccl_comm_window_deregister_impl;
    (*comm)->create_pre_mul_sum = oneccl_create_pre_mul_sum_impl;
    (*comm)->reduction_destroy = oneccl_reduction_destroy_impl;
    return onecclSuccess;
}

onecclResult_t oneccl_group_start_impl() {
    ccl::group_start();
    return onecclSuccess;
}

onecclResult_t oneccl_group_end_impl() {
    ccl::group_end();
    return onecclSuccess;
}
} // namespace

extern "C" void *onecclPluginCall(onecclPluginCall_t call_type) {
    switch (call_type) {
    case ONECCL_PLUGIN_INIT:
        return reinterpret_cast<void *>(&oneccl_plugin_init);

    case ONECCL_PLUGIN_INIT_ID:
        return reinterpret_cast<void *>(&oneccl_get_unique_id_impl);

    case ONECCL_PLUGIN_INIT_DEVICE:
        return reinterpret_cast<void *>(&oneccl_set_device_impl);

    case ONECCL_PLUGIN_MEM_ALLOC:
        return reinterpret_cast<void *>(&oneccl_mem_alloc_impl);

    case ONECCL_PLUGIN_MEM_FREE:
        return reinterpret_cast<void *>(&oneccl_mem_free_impl);

    case ONECCL_PLUGIN_INIT_COMM:
        return reinterpret_cast<void *>(&oneccl_init_communicator_impl);

    case ONECCL_PLUGIN_GROUP_START:
        return reinterpret_cast<void *>(&oneccl_group_start_impl);

    case ONECCL_PLUGIN_GROUP_END:
        return reinterpret_cast<void *>(&oneccl_group_end_impl);

    case ONECCL_PLUGIN_PLATFORM_SCORE:
        return reinterpret_cast<void *>(&oneccl_platform_score_impl);

    default:
        return nullptr;
    }
}
