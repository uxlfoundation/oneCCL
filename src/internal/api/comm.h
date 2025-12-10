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

#include "oneapi/ccl/v2/types.h"
#include <cstddef>

#ifndef ONECCL_C_INTERNAL_COMMUNICATOR_H
#define ONECCL_C_INTERNAL_COMMUNICATOR_H

#ifdef __cplusplus
extern "C" {
#endif

// Structure representing a communicator implementation
struct onecclComm {
    size_t size;
    int magic;
    int version;
    void *pExt;
    onecclUniqueId id;
    onecclConfig_t config;
    onecclResult_t error;
    onecclResult_t (*allreduce)(void *sendbuff, void *recvbuff, size_t count,
                                onecclDataType_t datatype,
                                onecclRedOp_t reduction_op, onecclComm_t comm,
                                void *stream);
    onecclResult_t (*allgather)(const void *sendbuff, void *recvbuff,
                                size_t sendcount, onecclDataType_t datatype,
                                onecclComm_t comm, void *stream);
    onecclResult_t (*alltoall)(const void *sendbuff, void *recvbuff,
                               size_t sendcount, onecclDataType_t datatype,
                               onecclComm_t comm, void *stream);
    onecclResult_t (*broadcast)(const void *sendbuff, void *recvbuff,
                                size_t count, onecclDataType_t datatype,
                                int root, onecclComm_t comm, void *stream);
    onecclResult_t (*reduce)(const void *sendbuff, void *recvbuff, size_t count,
                             onecclDataType_t datatype,
                             onecclRedOp_t reduction_op, int root,
                             onecclComm_t comm, void *stream);
    onecclResult_t (*reduce_scatter)(const void *sendbuff, void *recvbuff,
                                     size_t recvcount,
                                     onecclDataType_t datatype,
                                     onecclRedOp_t redop, onecclComm_t comm,
                                     void *stream);
    onecclResult_t (*send)(const void *sendbuff, size_t count,
                           onecclDataType_t datatype, int peer,
                           onecclComm_t comm, void *stream);
    onecclResult_t (*recv)(void *recvbuff, size_t count,
                           onecclDataType_t datatype, int peer,
                           onecclComm_t comm, void *stream);
    onecclResult_t (*barrier)(onecclComm_t comm);
    onecclResult_t (*get_rank)(onecclComm_t comm, int *rank);
    onecclResult_t (*get_local_rank)(onecclComm_t comm, int *local_rank);
    onecclResult_t (*get_size)(onecclComm_t comm, int *size);
    onecclResult_t (*get_local_size)(onecclComm_t comm, int *local_size);
    onecclResult_t (*get_device)(onecclComm_t comm, int *rank);
    onecclResult_t (*create_pre_mul_sum)(onecclRedOp_t* redop, void* scalar,
                                         onecclDataType_t datatype,
                                         onecclScalarResidence_t residence,
                                         onecclComm_t comm);
    onecclResult_t (*reduction_destroy)(onecclRedOp_t redop, onecclComm_t comm);
    onecclResult_t (*split)(onecclComm_t comm, int color, int key,
                            onecclComm_t *newcomm, onecclConfig_t *config);
    onecclResult_t (*destroy)(onecclComm_t comm);
    onecclResult_t (*finalize)(onecclComm_t comm);
    onecclResult_t (*abort)(onecclComm_t comm);
};

#ifdef __cplusplus
}
#endif

#endif // ONECCL_C_INTERNAL_COMMUNICATOR_H