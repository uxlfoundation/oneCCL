/*
 Copyright 2016-2025 Intel Corporation
 
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

// plugins/legacy/ccl_legacy_cpu.cpp
#include "internal/api/comm.h"
#include "internal/api/plugin.h"
#include "internal/debug.h"
#include "internal/itt_wrapper.hpp"
#include "oneapi/ccl.h"
#include "oneapi/ccl/v2/types.h"

#undef SYCL_LANGUAGE_VERSION
#undef __INTEL_LLVM_COMPILER

#include "oneapi/ccl.hpp"

#include <functional>

namespace {
std::map<ccl::kvs::address_type, ccl::shared_ptr_class<ccl::kvs>> kvs_for;

using onecclUniqueIdLegacy = struct onecclUniqueIdLegacy {
    ccl::kvs::address_type address;
};

class CommunicatorLegacyCpu {
  public:
    int rank;
    int size;
    ccl::communicator comm;

    CommunicatorLegacyCpu(int rank, int size, ccl::kvs::address_type address)
        : rank(rank), size(size), comm(std::move(ccl::create_communicator(
                                      size, rank, kvs_for[address]))) {}

    // Construction using external ccl::communicator, for example as a result of
    // a split
    explicit CommunicatorLegacyCpu(ccl::communicator &&comm)
        : comm(std::move(comm)) {}
};

// Implementation of onecclCommunicatorDestroy function
onecclResult_t oneccl_destroy_communicator_impl(onecclComm_t comm) {
    auto *comm_legacy = static_cast<CommunicatorLegacyCpu *>(comm->pExt);
    auto *legacy_id =
        reinterpret_cast<onecclUniqueIdLegacy *>(&comm->id.legacy);
    delete comm_legacy;

    kvs_for.erase(legacy_id->address);

    return onecclSuccess;
}

onecclResult_t oneccl_get_rank_impl(onecclComm_t comm, int *out_rank) {
    auto *comm_legacy = static_cast<CommunicatorLegacyCpu *>(comm->pExt);
    *out_rank = comm_legacy->rank;
    return onecclSuccess;
}

onecclResult_t oneccl_get_local_rank_impl(onecclComm_t comm, int *local_rank) {
    auto *comm_legacy = static_cast<CommunicatorLegacyCpu *>(comm->pExt);
    *local_rank = comm_legacy->rank;
    return onecclSuccess;
}

onecclResult_t oneccl_get_size_impl(onecclComm_t comm, int *out_size) {
    auto *comm_legacy = static_cast<CommunicatorLegacyCpu *>(comm->pExt);
    *out_size = comm_legacy->size;
    return onecclSuccess;
}

onecclResult_t oneccl_get_local_size_impl(onecclComm_t comm, int *local_size) {
    auto *comm_legacy = static_cast<CommunicatorLegacyCpu *>(comm->pExt);
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
    default:
        throw std::invalid_argument("Unsupported onecclRedOp_t value");
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

onecclResult_t execute_collective(
    onecclComm_t comm, void *stream,
    const std::function<void(CommunicatorLegacyCpu *comm_legacy,
                             void *async_cpu_queue)> &cpu_collective) {

    auto *comm_legacy = static_cast<CommunicatorLegacyCpu *>(comm->pExt);
    if (comm_legacy == nullptr) {
        return onecclInvalidArgument;
    }

    cpu_collective(comm_legacy, stream);

    return onecclSuccess;
}

onecclResult_t oneccl_allreduce_impl(void *sendbuff, void *recvbuff,
                                     size_t count, onecclDataType_t datatype,
                                     onecclRedOp_t red_op, onecclComm_t comm,
                                     void *stream) {
    return execute_collective(
        comm, stream,
        [=](CommunicatorLegacyCpu *comm_legacy, void * /*async_queue*/) {
            static itt::Task allreduce("CPU::AllReduce");

            allreduce.start();
            ccl::allreduce(sendbuff, recvbuff, count, convert(datatype),
                           convert(red_op), comm_legacy->comm)
                .wait();
            allreduce.end();
        });
}

onecclResult_t oneccl_allgather_impl(const void *sendbuff, void *recvbuff,
                                     size_t sendcount,
                                     onecclDataType_t datatype,
                                     onecclComm_t comm, void *stream) {
    return execute_collective(
        comm, stream,
        [=](CommunicatorLegacyCpu *comm_legacy, void * /*async_queue*/) {
            static itt::Task allgather("CPU::AllGather");

            allgather.start();
            size_t const comm_size = comm_legacy->comm.size();
            std::vector<size_t> const recvcounts(comm_size, sendcount);
            ccl::allgatherv(sendbuff, sendcount, recvbuff, recvcounts,
                            convert(datatype), comm_legacy->comm)
                .wait();
            allgather.end();
        });
}

onecclResult_t oneccl_alltoall_impl(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, onecclDataType_t datatype,
                                    onecclComm_t comm, void *stream) {
    return execute_collective(
        comm, stream,
        [=](CommunicatorLegacyCpu *comm_legacy, void * /*async_queue*/) {
            static itt::Task alltoall("CPU::AllToAll");

            alltoall.start();
            size_t const comm_size = comm_legacy->comm.size();
            std::vector<size_t> const recvcounts(comm_size, sendcount);
            ccl::alltoall(sendbuff, recvbuff, sendcount, convert(datatype),
                          comm_legacy->comm)
                .wait();
            alltoall.end();
        });
}

onecclResult_t oneccl_broadcast_impl(const void *sendbuff, void *recvbuff,
                                     size_t count, onecclDataType_t datatype,
                                     int root, onecclComm_t comm,
                                     void *stream) {
    return execute_collective(
        comm, stream,
        [=](CommunicatorLegacyCpu *comm_legacy, void * /*async_queue*/) {
            static itt::Task broadcast("CPU::Broadcast");

            broadcast.start();
            ccl::broadcast(const_cast<void *>(sendbuff), recvbuff, count,
                           convert(datatype), root, comm_legacy->comm)
                .wait();
            broadcast.end();
        });
}

onecclResult_t oneccl_reduce_impl(const void *sendbuff, void *recvbuff,
                                  size_t count, onecclDataType_t datatype,
                                  onecclRedOp_t reduction_op, int root,
                                  onecclComm_t comm, void *stream) {
    return execute_collective(
        comm, stream,
        [=](CommunicatorLegacyCpu *comm_legacy, void * /*async_queue*/) {
            static itt::Task reduce("CPU::Reduce");

            reduce.start();
            ccl::reduce(sendbuff, recvbuff, count, convert(datatype),
                        convert(reduction_op), root, comm_legacy->comm)
                .wait();
            reduce.end();
        });
}

onecclResult_t oneccl_reduce_scatter_impl(const void *sendbuff, void *recvbuff,
                                          size_t recvcount,
                                          onecclDataType_t datatype,
                                          onecclRedOp_t redop,
                                          onecclComm_t comm, void *stream) {
    return execute_collective(
        comm, stream,
        [=](CommunicatorLegacyCpu *comm_legacy, void * /*async_queue*/) {
            static itt::Task reduce_scatter("CPU::ReduceScatter");
            reduce_scatter.start();
            ccl::reduce_scatter(sendbuff, recvbuff, recvcount,
                                convert(datatype), convert(redop),
                                comm_legacy->comm)
                .wait();
            reduce_scatter.end();
        });
}

onecclResult_t oneccl_send_impl(const void *sendbuff, size_t count,
                                onecclDataType_t datatype, int peer,
                                onecclComm_t comm, void *stream) {
    return execute_collective(
        comm, stream,
        [=](CommunicatorLegacyCpu *comm_legacy, void * /*async_queue*/) {
            static itt::Task send("CPU::Send");

            send.start();
            ccl::send(const_cast<void *>(sendbuff), count, convert(datatype),
                      peer, comm_legacy->comm)
                .wait();
            send.end();
        });
}

onecclResult_t oneccl_recv_impl(void *recvbuff, size_t count,
                                onecclDataType_t datatype, int peer,
                                onecclComm_t comm, void *stream) {
    return execute_collective(
        comm, stream,
        [=](CommunicatorLegacyCpu *comm_legacy, void * /*async_queue*/) {
            static itt::Task recv("CPU::Recv");

            recv.start();
            ccl::recv(recvbuff, count, convert(datatype), peer,
                      comm_legacy->comm)
                .wait();
            recv.end();
        });
}

onecclResult_t oneccl_barrier_impl(onecclComm_t comm) {
    return execute_collective(
        comm, nullptr,
        [=](CommunicatorLegacyCpu *comm_legacy, void * /*async_queue*/) {
            static itt::Task barrier("CPU::Barrier");
            barrier.start();
            ccl::barrier(comm_legacy->comm).wait();
            barrier.end();
        });
}

onecclResult_t oneccl_comm_split_impl(onecclComm_t comm, int color, int key,
                                      onecclComm_t *newcomm,
                                      onecclConfig_t * /*config*/) {
    auto *comm_legacy = static_cast<CommunicatorLegacyCpu *>(comm->pExt);

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

    auto *new_comm_legacy = new CommunicatorLegacyCpu(
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
    return onecclSuccess;
}

onecclResult_t oneccl_plugin_init() { return onecclSuccess; }

onecclResult_t oneccl_get_unique_id_impl(onecclUniqueId *unique_id) {
    auto *legacy_id =
        reinterpret_cast<onecclUniqueIdLegacy *>(&unique_id->legacy);

    auto kvs = ccl::create_main_kvs();
    auto address = kvs->get_address();

    legacy_id->address = address;
    kvs_for[address] = kvs;

    return onecclSuccess;
}

onecclResult_t oneccl_platform_score_impl(int *score) {
    *score = 1000;
    return onecclSuccess;
}

onecclResult_t
oneccl_init_communicator_impl(onecclComm_t *comm, size_t nranks,
                              onecclUniqueId commId, int rank,
                              const onecclConfig_t * /*config*/) {
    auto *legacy_id = reinterpret_cast<onecclUniqueIdLegacy *>(&commId.legacy);

    if (kvs_for.find(legacy_id->address) == kvs_for.end()) {
        kvs_for[legacy_id->address] =
            ccl::create_kvs(legacy_id->address); // Create missing kvs
    }

    auto *comm_legacy = new CommunicatorLegacyCpu(
        rank, static_cast<int>(nranks), legacy_id->address);

    (*comm)->pExt = comm_legacy;
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
