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

/**
 * @file ccl.h
 * @brief OneAPI Collective Communications Library (oneCCL) API definitions
 */

#include <cstddef>
#include <cstdint>

#include "oneapi/ccl/v2/types.h"

#ifndef ONECCL_C_API_H
#define ONECCL_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @defgroup CommunicatorCreation Communicator creation APIs
 */

/**
 *  @defgroup CollectiveFunctions Collective communication functions APIs
 */

/**
 *  @defgroup Types Datatypes used in the API
 */

// Definition for Windows and Unix-based systems
#ifdef _WIN32
#ifdef CCL_C_API_EXPORT
#define CCL_C_API __declspec(dllexport)
#else
#define CCL_C_API __declspec(dllimport)
#endif
#define CCL_C_EXPORT __declspec(dllexport)
#else
#if __GNUC__ >= 4
#define CCL_C_API __attribute__((visibility("default"))) __attribute__((weak))
#define CCL_C_EXPORT __attribute__((visibility("default")))
#else
#define CCL_C_API
#endif
#endif

#define CCL_C_NOT_IMPLEMENTED                                                  \
    __attribute__((                                                            \
        error("This API is not yet implemented in C API for oneCCL. Please "   \
              "see https://uxlfoundation.github.io/oneCCL/v2/api.html for "    \
              "more information on oneCCL APIs")))

/**
 *  @defgroup Macros Macros provided by oneCCL
 */

/**
 * @def ONECCL_MAJOR
 * @brief Major version of oneCCL
 * @ingroup Macros
 */
#define ONECCL_MAJOR 2021

/**
 * @def ONECCL_MINOR
 * @brief Minor version of oneCCL
 * @ingroup Macros
 */
#define ONECCL_MINOR 17

/**
 * @def ONECCL_PATCH
 * @brief Patch version of oneCCL
 * @ingroup Macros
 */
#define ONECCL_PATCH 0

/**
 * @def ONECCL_SUFFIX
 * @brief Suffix version of oneCCL
 * @ingroup Macros
 */
#define ONECCL_SUFFIX 0

/**
 * @def ONECCL_VERSION_CODE
 * @brief Encoded version number
 * @ingroup Macros
 */
#define ONECCL_VERSION_CODE                                                    \
    ((ONECCL_MAJOR)*10000 + (ONECCL_MINOR)*100 + (ONECCL_SUFFIX))

/**
 * @def ONECCL_VERSION
 * @brief Helper macro to encode a version number
 * @ingroup Macros
 */
#define ONECCL_VERSION(X, Y, Z) (((X)*10000) + ((Y)*100) + (Z))


/**
 * @brief Function to obtain the oneCCL version encoded as an integer
 *
 * This function returns the version number encoded as an integer, which includes
 * the oneCCL major version, oneCCL minor version, and oneCCL patch. The user can
 * use the onecclExtractVersionComponents function to extract the version components.
 *
 * @param[out] version Pointer to store the version encoded integer
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API onecclGetVersion(int *version);

/**
 * @brief Function to get a unique ID
 *
 * This function generates a unique ID to be used in onecclCommInitRank or onecclCommInitRankConfig.
 * ncclGetUniqueId is called once before creating a communicator. The ID should be sent to all the ranks
 * that are going to participate in the communicator before they call onecclCommInitRank or
 * onecclCommInitRankConfig.
 *
 * @param[out] uniqueId Pointer to store the unique ID
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API onecclGetUniqueId(onecclUniqueId *uniqueId);

/**
 * @brief Function to extract version components from the version encoded integer
 *
 * This function takes the version encoded integer obtained with oneCCLGetVErsion and  returns
 * the oneCCL major version, minor version, and patch.
 *
 * @param[in] versionCode Encoded version integer
 * @param[out] major Pointer to store major version
 * @param[out] minor Pointer to store minor version
 * @param[out] patch Pointer to store patch version
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API onecclExtractVersionComponents(int versionCode,
                                                       int *major, int *minor,
                                                       int *patch);

/**
 * @brief Function to create a new communicator
 *
 * This function creates a  new communicator with nranks, where rank must be an integer between 0 and
 * nranks-1 and unique in the communicator. commId is the unique ID obtained with onecclGetUniqueId.
 * This is a collective call and needs to be called by all the processes participating in the communicator.
 *
 * Before this call, each rank needs to specify the device it is associated with (this can be done with
 * onecclSetDevice(devide-idx), where device-idx is the device index.
 *
 * @param[out] comm Pointer to store the initialized communicator
 * @param[in] nranks Number of ranks
 * @param[in] commId Unique ID for the communicator
 * @param[in] rank Rank within the communicator
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API onecclCommInitRank(onecclComm_t *comm, size_t nranks,
                                           onecclUniqueId commId, int rank);

/**
 * @brief Function to create a new communicator using a config argument
 *
 * This function is similar to onecclCommInitRank but it also takes a config argument with additional attributes
 * for the communicator.
 *
 * @param[out] comm Pointer to store the new communicator
 * @param[in] nranks Number of ranks
 * @param[in] commId Unique ID for the communicator
 * @param[in] rank Rank within the communicator
 * @param[in] config Configuration attributes for the communicator
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API onecclCommInitRankConfig(onecclComm_t *comm, size_t nranks,
                                                 onecclUniqueId commId, int rank,
                                                 const onecclConfig_t *config);

/**
 * @brief Function to create a single-process communicator
 *
 * This API is not implemented yet.
 *
 * @param[out] comm Pointer to store the initialized communicator
 * @param[in] ndev Number of devices
 * @param[in] devlist List of devices
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API CCL_C_NOT_IMPLEMENTED onecclCommInitAll(onecclComm_t *comm, int ndev,
                                          const int *devlist);

/**
 * @brief Function to flush all communication inside the communicator
 *
 * This API is not implemented yet.
 *
 * @param[in] comm Communicator to finalize
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API CCL_C_NOT_IMPLEMENTED onecclCommFinalize(onecclComm_t comm);

/**
 * @brief Function to destroy a communicator
 *
 * This function frees the local resources allocated to the communicator object comm
 *
 * @param[in] comm Communicator to destroy
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API onecclCommDestroy(onecclComm_t comm);

/**
 * @brief Funtion to abort uncompleted operations and destroy the communicator
 *
 * This API is not implemented yet
 *
 * @param[in] comm Communicator to abort
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API CCL_C_NOT_IMPLEMENTED onecclCommAbort(onecclComm_t comm);

/**
 * @brief Function to create a set of new communicators from an existing one
 *
 * This functions creates a set of new communicators. Ranks with the same color will be in the same communicator.
 * Ranks with  `ONECCL_SPLIT_NOCOLOR` will not be part of any new communicator and will return NULL in
 * the new communicator.
 * The key determines the order in the new communicator, with a smaller key value indicating a smaller rank in
 * the new communicator. Ranks with the same key will be ordered based on order in the orginal communicator.
 * If the communicator needs to have a new configuration, this has to be passed in the config argument. Otherwise,
 * setting config to NULL will make the new communicators inherit the configuration of the original communicator.
 * When calling this function, there should not be any pending operations in the communicator. Otherwise, there
 * could be a deadlock.
 *
 * @param[in] comm Original communicator
 * @param[in] color Color identifier for splitting
 * @param[in] key Key for ranking within color
 * @param[out] newcomm Pointer to store newly created communicator
 * @param[out] config Configuration for the new communicator
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API onecclCommSplit(onecclComm_t comm, int color, int key,
                                        onecclComm_t *newcomm,
                                        onecclConfig_t *config);


/**
 * @brief Returns a  string for the error code in result
 *
 * Returns a human-readable string corresponding to the error code in result.
 *
 * @param[in] result Return code to describe
 * @return Description of the return code
 * @ingroup CommunicatorCreation
 */
const char CCL_C_API *onecclGetErrorString(onecclResult_t result);

/**
 * @brief Funtion that returns an error message for the last error that occurred in the communicator
 *
 * This function returns a human-readable string corresponding to the last error that occurred in the communicator.
 * Notice that the error message may not be related to the current call, but rather to a previous non-blocking call.
 *
 * @param[in] comm Communicator to query for last error
 * @return Description of the last error
 * @ingroup CommunicatorCreation
 */
const char CCL_C_API *onecclGetLastError(onecclComm_t comm);

void CCL_C_API onecclDebugInit();

void CCL_C_API onecclResetDebugInit();

/**
 * @brief Function to obtain communicator size
 *
 * This function returns the number of ranks in the communicator.
 *
 * @param[in] comm Communicator to query
 * @param[out] size Pointer to store the size
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API onecclCommCount(const onecclComm_t comm, int *size);

/**
 * @brief Function to get the device used by the communicator
 *
 * This function returns the device associated with the communicator
 *
 * @param[in] comm Communicator to query
 * @param[out] device Pointer to store the device index
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API onecclCommDevice(const onecclComm_t comm, int *device);

/**
 * @brief Function to get the rank within the communicator
 *
 * This function returns the rank of the caller in the communicator.
 *
 * @param[in] comm Communicator to query
 * @param[out] rank Pointer to store the rank
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */

onecclResult_t CCL_C_API onecclCommUserRank(const onecclComm_t comm, int *rank);

/**
 * @brief Function to set the device index for the calling rank
 *
 * This function records the device index associated with the calling rank/thread.
 *
 * @param[in] index Index of device to select
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API onecclSetDevice(uint32_t index);

/**
 * @brief Function to check for errors of asynchronous oneCCL operations in the communicator
 *
 * This API is not implemented yet
 *
 * @param[in] comm Communicator to check
 * @return Result of the operation
 * @ingroup CommunicatorCreation
 */
onecclResult_t CCL_C_API CCL_C_NOT_IMPLEMENTED onecclCommGetAsyncError(onecclComm_t comm);

/**
 * @brief Functon to perform a Reduce operation
 *
 * Reduce is a collective communication operation that performs a reduction operation redop on count elements in
 * sendbuf and places the result into the recvbuff of the root rank. The recvbuf is only used on the root rank.
 *
 * This operation is in-place if sendbuff == recvbuff.
 *
 * The stream is usually a pointer to a SYCL queue, but it can be NULL for host buffers. For details, see <a href="./plugins.html">plugin</a>.
 *
 * @param[in] sendbuff Buffer with the data to send ABCDCACSAC
 * @param[out] recvbuff Buffer to receive reduced data
 * @param[in] count Number of elements
 * @param[in] datatype Data type of elements
 * @param[in] redop Reduction operation
 * @param[in] root Root rank of the operation
 * @param[in] comm Communicator for the operation
 * @param[in] stream Stream for the reduction
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclReduce(const void *sendbuff, void *recvbuff,
                                     size_t count, onecclDataType_t datatype,
                                     onecclRedOp_t redop, int root,
                                     onecclComm_t comm, void *stream);

/**
 * @brief Function to perform an AllReduce operation
 *
 * Allreduce is a collective communication operation that performs a reduction operation redop on count elements in
 * sendbuf and places the result into the recvbuff of each rank. recvbuff is equal in all the ranks.
 *
 * This operation is in-place if sendbuff == recvbuff.
 *
 * The stream is usually a pointer to a SYCL queue, but it can be NULL for host buffers. For details, see <a href="./plugins.html">plugin</a>.
 *
 * @param[in] sendbuff Buffer with the data to send
 * @param[out] recvbuff Buffer to receive reduced data
 * @param[in] count Number of elements
 * @param[in] datatype Data type of elements
 * @param[in] reduction_op Reduction operation
 * @param[in] comm Communicator for the operation
 * @param[in] stream Stream for the reduction
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclAllReduce(void *sendbuff, void *recvbuff,
                                        size_t count, onecclDataType_t datatype,
                                        onecclRedOp_t reduction_op,
                                        onecclComm_t comm, void *stream);

/**
 * @brief Function to performs a Broadcast operation
 *
 * Broadcast is a collective communication operation that copies count elements from the sendbuf in the root rank to the
 * recvbuff of all the ranks. sendbuf is only used in the root rank.
 *
 * The operation is in-place if sendbuff == recvbuff.
 *
 * The stream is usually a pointer to a SYCL queue, but it can be NULL for host buffers. For details, see <a href="./plugins.html">plugin</a>.
 *
 * @param[in] sendbuff Buffer with the data to send from root
 * @param[out] recvbuff Buffer to receive broadcasted data
 * @param[in] count Number of elements
 * @param[in] datatype Data type of elements
 * @param[in] root Root rank of the operation
 * @param[in] comm Communicator for the operation
 * @param[in] stream Stream for the broadcast
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclBroadcast(const void *sendbuff, void *recvbuff,
                                        size_t count, onecclDataType_t datatype,
                                        int root, onecclComm_t comm,
                                        void *stream);

/**
 * @brief Function to perform a ReduceScatter operation
 *
 * ReduceScatter is a collective communication operation that performs a reduction operation redop on count elements in
 * sendbuf and places the result scattered over the recvbuff of the participating ranks, so that the recvbuff in rank i contains
 * the i-th chunk of the result.
 *
 * This operation assumes that send count is equal to nranks*recvcount, that is, the sendbuf has a count of at least
 * nranks*recvcount elements.
 *
 * This operation is in-place if recvbuff == sendbuff + rank * recvcount.
 *
 * The stream is usually a pointer to a SYCL queue, but it can be NULL for host buffers. For details, see <a href="./plugins.html">plugin</a>.
 *
 * @param[in] sendbuff Buffer with the data to send
 * @param[out] recvbuff Buffer to receive scattered data
 * @param[in] recvcount Number of elements to receive
 * @param[in] datatype Data type of elements
 * @param[in] redop Reduction operation
 * @param[in] comm Communicator for the operation
 * @param[in] stream Stream for the scatter
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclReduceScatter(const void *sendbuff,
                                            void *recvbuff, size_t recvcount,
                                            onecclDataType_t datatype,
                                            onecclRedOp_t redop,
                                            onecclComm_t comm, void *stream);

/**
 * @brief Function to perform an AllGgather operation
 *
 * Allgather is a collective communication operation that gathers sendcount elements from the sendbuf in each rank and
 * places them in the recvbuff of all the participating ranks. The data in the sendbuf in rank i can be found in recvbuf
 * at offset i*sendcount.
 *
 * This operation assumes that the receive count is equal to nranks*sendcount, that is, the recvbuff has a count of at least
 * nranks*sendcount elements.
 *
 * This operation is in-place if sendbuff == recvbuff + rank * sendcount.
 *
 * The stream is usually a pointer to a SYCL queue, but it can be NULL for host buffers. For details, see <a href="./plugins.html">plugin</a>.
 *
 * @param[in] sendbuff Buffer with the data to send
 * @param[out] recvbuff Buffer to receive gathered data
 * @param[in] sendcount Number of elements to send
 * @param[in] datatype Data type of elements
 * @param[in] comm Communicator for the operation
 * @param[in] stream Stream for the gather
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclAllGather(const void *sendbuff, void *recvbuff,
                                        size_t sendcount,
                                        onecclDataType_t datatype,
                                        onecclComm_t comm, void *stream);

/**
 * @brief Function to perform an AlltoAll operation
 *
 * Alltoall is a collective communication operation where each rank sends count elements to all other ranks and receives count
 * elements from all other ranks. The data to send to destination rank j is located at sendbuff+j*count and data received from
 * source rank i is placed at recvbuff+i*count.
 *
 * This collective assumes that the count in sendbuff and recvbuff is the same and it is equal to nranks*count.
 *
 * The stream is usually a pointer to a SYCL queue, but it can be NULL for host buffers. For details, see <a href="./plugins.html">plugin</a>
 *
 * @param[in] sendbuff Buffer with the data to send to each process
 * @param[out] recvbuff Buffer to receive data
 * @param[in] count Number of elements to send and receive from each process
 * @param[in] datatype Data type of elements
 * @param[in] comm Communicator for the operation
 * @param[in] stream Stream for the AllToAll operation
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclAllToAll(const void *sendbuff, void *recvbuff,
                                        size_t count,
                                        onecclDataType_t datatype,
                                       onecclComm_t comm, void *stream);

/**
 * @brief Fun ction to perform a send operation
 *
 * This operation sends count data from sendbuff to peer rank. The peer rank needs to call onecclRecv
 * with the same count and dataype as the calling rank.
 *
 * This operation may block the GPU. If multiple  onecclSend() and onecclRecv() operations need to
 * progress concurrently in a non-blocking fashion, they need to be placed within the onecclGroupStart() and
 * oneccllGroupEnd() calls.
 *
 * The stream is usually a pointer to a SYCL queue, but it can be NULL for host buffers. For details, see <a href="./plugins.html">plugin</a>
 *
 * @param[in] sendbuff Buffer with the data to send
 * @param[in] count Number of elements
 * @param[in] datatype Data type of elements
 * @param[in] peer Rank of the peer
 * @param[in] comm Communicator for the operation
 * @param[in] stream Stream for the send
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclSend(const void *sendbuff, size_t count,
                                   onecclDataType_t datatype, int peer,
                                   onecclComm_t comm, void *stream);

/**
 * @brief Function to perform a receive operation
 *
 * This operation receives count data from recvbuff from peer rank. The peer rank needs to call onecclSend
 * with the same count and dataype as the calling rank.
 *
 * This operation may block the GPU. If multiple  onecclSend() and onecclRecv() operations need to
 * progress concurrently in a non-blocking fashion, they need to be placed within the onecclGroupStart() and
 * oneccllGroupEnd() calls.
 *
 * The stream is usually a pointer to a SYCL queue, but it can be NULL for host buffers. For details, see <a href="./plugins.html">plugin</a>
 *
 * @param[out] recvbuff Buffer to receive data
 * @param[in] count Number of elements
 * @param[in] datatype Data type of elements
 * @param[in] peer Rank of the peer
 * @param[in] comm Communicator for the operation
 * @param[in] stream Stream for the receive
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclRecv(void *recvbuff, size_t count,
                                   onecclDataType_t datatype, int peer,
                                   onecclComm_t comm, void *stream);

/**
 * @brief Function to start a group call
 *
 * This function indicates that all the subsequent oneCCL calls until onecclGrouEnd will not block due to CPU synchronization.
 *
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclGroupStart();

/**
 * @brief Function to end a group call
 *
 * This operation stars all the oneCCL operations submitted after the most recent onecclGroupStart.
 *
 * At the moment, the only operations supported between onecclGroupStart and onecclGroupEnd are collectives and
 * point to point send and receive.
 *
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclGroupEnd();

/**
 * @brief Function to create a custom reduction operation that performs a pre-multiplied sum
 *
 * This function creates a new reduction operator that pre-mulitplies the input values by the scalar before reducing them
 * with peer values wih a sum.
 *
 * The input data and the scalar are of type datatype.
 *
 * The residence argument indicates whether the memory pointed by the scalar is in the host or device memory.
 * See  <a href="./api.html#c.onecclScalarResidence_t">onecclScalarResidence_t</a>.
 *
 * The handle to the new created reduction operation is stored in redop.
 *
 * @param[out] redop Pointer to store the created reduction operation
 * @param[in] scalar Pointer to the scalar value to pre-multiply
 * @param[in] datatype Data type of the scalar value
 * @param[in] residence Memory residence of the scalar value
 * @param[in] comm Communicator for the operation
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclRedOpCreatePreMulSum(
    onecclRedOp_t *redop, void *scalar, onecclDataType_t datatype,
    onecclScalarResidence_t residence, onecclComm_t comm);

/**
 * @brief Function to destroy a previously created custom reduction operation
 *
 * Destroys the reduction operation redop.
 *
 * This API assumes the reduction operation has been created with onecclRedOpCreatePreMul with the communicator comm.
 * An operation can be destroyed when the last oneccl function using that reduction operation returns.
 *
 * @param[in] redop Previously created reduction operation
 * @param[in] comm Communicator for the operation
 * @return Result of the operation
 * @ingroup CollectiveFunctions
 */
onecclResult_t CCL_C_API onecclRedOpDestroy(onecclRedOp_t redop,
                                           onecclComm_t comm);

#ifdef __cplusplus
}
#endif

#endif // ONECCL_C_API_H
