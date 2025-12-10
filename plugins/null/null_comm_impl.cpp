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

#include <cstddef>
#include <cstdint>

#include "internal/api/comm.h"
#include "internal/api/plugin.h"
#include "internal/itt_wrapper.hpp"
#include "null_hook.hpp" // Include the hooks header
#include "oneapi/ccl/v2/types.h"

namespace {

thread_local itt::Task allreduce_task("NULL AllReduce");

onecclResult_t oneccl_destroy_communicator_impl(onecclComm_t /*comm*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_DESTROY_COMM);

    PROCESS_HOOKS(ONECCL_AFTER_DESTROY_COMM);

    return onecclSuccess;
}

onecclResult_t oneccl_get_rank_impl(onecclComm_t /*comm*/, int *out_rank) {
    PROCESS_HOOKS(ONECCL_BEFORE_GET_RANK);

    *out_rank = 0;

    PROCESS_HOOKS(ONECCL_AFTER_GET_RANK);

    return onecclSuccess;
}

onecclResult_t oneccl_get_size_impl(onecclComm_t /*comm*/, int *out_size) {
    PROCESS_HOOKS(ONECCL_BEFORE_GET_SIZE);

    *out_size = 1;

    PROCESS_HOOKS(ONECCL_AFTER_GET_SIZE);

    return onecclSuccess;
}

onecclResult_t oneccl_comm_device_impl(onecclComm_t /*comm*/, int *device) {
    PROCESS_HOOKS(ONECCL_BEFORE_GET_DEVICE);

    *device = 0;

    PROCESS_HOOKS(ONECCL_AFTER_GET_DEVICE);

    return onecclSuccess;
}

onecclResult_t oneccl_allreduce_impl(void * /*sendbuff*/, void * /*recvbuff*/,
                                     size_t /*count*/,
                                     onecclDataType_t /*datatype*/,
                                     onecclRedOp_t /*op*/,
                                     onecclComm_t /*comm*/, void * /*stream*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_ALLREDUCE);

    PROCESS_HOOKS(ONECCL_AFTER_ALLREDUCE);

    return onecclSuccess;
}

onecclResult_t oneccl_allgather_impl(const void * /*sendbuff*/,
                                     void * /*recvbuff*/, size_t /*sendcount*/,
                                     onecclDataType_t /*datatype*/,
                                     onecclComm_t /*comm*/, void * /*stream*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_ALLGATHER);

    PROCESS_HOOKS(ONECCL_AFTER_ALLGATHER);
    return onecclSuccess;
}

onecclResult_t oneccl_broadcast_impl(const void * /*sendbuff*/,
                                     void * /*recvbuff*/, size_t /*count*/,
                                     onecclDataType_t /*datatype*/,
                                     int /*root*/, onecclComm_t /*comm*/,
                                     void * /*stream*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_BROADCAST);

    PROCESS_HOOKS(ONECCL_AFTER_BROADCAST);
    return onecclSuccess;
}

onecclResult_t oneccl_reduce_impl(const void * /*sendbuff*/,
                                  void * /*recvbuff*/, size_t /*count*/,
                                  onecclDataType_t /*datatype*/,
                                  onecclRedOp_t /*reduction_op*/, int /*root*/,
                                  onecclComm_t /*comm*/, void * /*stream*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_REDUCE);

    PROCESS_HOOKS(ONECCL_AFTER_REDUCE);
    return onecclSuccess;
}

onecclResult_t
oneccl_reduce_scatter_impl(const void * /*sendbuff*/, void * /*recvbuff*/,
                           size_t /*recvcount*/, onecclDataType_t /*datatype*/,
                           onecclRedOp_t /*redop*/, onecclComm_t /*comm*/,
                           void * /*stream*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_REDUCE_SCATTER);

    PROCESS_HOOKS(ONECCL_AFTER_REDUCE_SCATTER);
    return onecclSuccess;
}

onecclResult_t oneccl_send_impl(const void * /*sendbuff*/, size_t /*count*/,
                                onecclDataType_t /*datatype*/, int /*peer*/,
                                onecclComm_t /*comm*/, void * /*stream*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_SEND);

    PROCESS_HOOKS(ONECCL_AFTER_SEND);
    return onecclSuccess;
}

onecclResult_t oneccl_recv_impl(void * /*recvbuff*/, size_t /*count*/,
                                onecclDataType_t /*datatype*/, int /*peer*/,
                                onecclComm_t /*comm*/, void * /*stream*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_RECV);

    PROCESS_HOOKS(ONECCL_AFTER_RECV);
    return onecclSuccess;
}

onecclResult_t oneccl_finalize_impl(onecclComm_t /*comm*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_FINALIZE);
    // Finalize logic
    PROCESS_HOOKS(ONECCL_AFTER_FINALIZE);
    return onecclSuccess;
}

onecclResult_t oneccl_abort_impl(onecclComm_t /*comm*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_ABORT);
    // Abort logic
    PROCESS_HOOKS(ONECCL_AFTER_ABORT);
    return onecclSuccess;
}

onecclResult_t oneccl_plugin_init() {
    PROCESS_HOOKS(ONECCL_BEFORE_PLUGIN_INIT);
    PROCESS_HOOKS(ONECCL_AFTER_PLUGIN_INIT);

    return onecclSuccess;
}

onecclResult_t oneccl_get_unique_id_impl(onecclUniqueId *id_ptr) {
    PROCESS_HOOKS(ONECCL_BEFORE_UNIQUE_ID);

    *(reinterpret_cast<int *>(&id_ptr->any[0])) = onecclNull;

    PROCESS_HOOKS(ONECCL_AFTER_UNIQUE_ID);

    return onecclSuccess;
}

onecclResult_t oneccl_set_device_impl(uint32_t /*index*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_SET_DEVICE);

    // Device setting logic would go here

    PROCESS_HOOKS(ONECCL_AFTER_SET_DEVICE);

    return onecclSuccess;
}

onecclResult_t oneccl_platform_score_impl(int *score) {
    *score = 10;
    return onecclSuccess;
}

onecclResult_t oneccl_create_pre_mul_sum_impl(
    onecclRedOp_t * /*redop*/, void * /*scalar*/, onecclDataType_t /*datatype*/,
    onecclScalarResidence_t /*residence*/, onecclComm_t /*comm*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_CREATE_PRE_MUL_SUM);
    // Device setting logic would go here
    PROCESS_HOOKS(ONECCL_AFTER_CREATE_PRE_MUL_SUM);
    return onecclSuccess;
}

onecclResult_t oneccl_reduction_destroy_impl(onecclRedOp_t /*redop*/,
                                             onecclComm_t /*comm*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_REDUCTION_DESTROY);
    // Device setting logic would go here
    PROCESS_HOOKS(ONECCL_AFTER_REDUCTION_DESTROY);
    return onecclSuccess;
}

onecclResult_t
oneccl_init_communicator_impl(onecclComm_t *comm, size_t /*nranks*/,
                              onecclUniqueId /*commId*/, int /*rank*/,
                              const onecclConfig_t * /*config*/) {
    PROCESS_HOOKS(ONECCL_BEFORE_INIT_COMM);

    (*comm)->pExt = nullptr;
    (*comm)->destroy = oneccl_destroy_communicator_impl;
    (*comm)->allreduce = oneccl_allreduce_impl;
    (*comm)->allgather = oneccl_allgather_impl;
    (*comm)->broadcast = oneccl_broadcast_impl;
    (*comm)->reduce = oneccl_reduce_impl;
    (*comm)->reduce_scatter = oneccl_reduce_scatter_impl;
    (*comm)->send = oneccl_send_impl;
    (*comm)->recv = oneccl_recv_impl;
    (*comm)->get_device = oneccl_comm_device_impl;
    (*comm)->get_rank = oneccl_get_rank_impl;
    (*comm)->get_size = oneccl_get_size_impl;
    (*comm)->create_pre_mul_sum = oneccl_create_pre_mul_sum_impl;
    (*comm)->reduction_destroy = oneccl_reduction_destroy_impl;
    (*comm)->finalize = oneccl_finalize_impl;
    (*comm)->abort = oneccl_abort_impl;

    PROCESS_HOOKS(ONECCL_AFTER_INIT_COMM);

    return onecclSuccess;
}
} // namespace

extern "C" void *onecclPluginCall(onecclPluginCall_t call_type) {
    switch (call_type) {
    case ONECCL_PLUGIN_INIT:
        return reinterpret_cast<void *>(&oneccl_plugin_init);

    case ONECCL_PLUGIN_INIT_COMM:
        return reinterpret_cast<void *>(&oneccl_init_communicator_impl);

    case ONECCL_PLUGIN_INIT_ID:
        return reinterpret_cast<void *>(&oneccl_get_unique_id_impl);

    case ONECCL_PLUGIN_INIT_DEVICE:
        return reinterpret_cast<void *>(&oneccl_set_device_impl);

    case ONECCL_PLUGIN_PLATFORM_SCORE:
        return reinterpret_cast<void *>(&oneccl_platform_score_impl);

    case ONECCL_PLUGIN_GROUP_START:

    case ONECCL_PLUGIN_GROUP_END:

    default:
        return nullptr;
    }
}
