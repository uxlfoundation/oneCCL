/*
 Copyright 2016-2020 Intel Corporation
 
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
#pragma once

#include "oneapi/ccl/config.h"

/*
 * This file uses local .clang-format file in order to have
 * unlimited line length which is required for proper
 * handling in doxygen.
*/

/**
 * @defgroup OneCCLvars OneCCL Environment Variables
 * @{
 * @}
 **/

constexpr const char* CCL_ENV_STR_NOT_SPECIFIED = "<not specified>";
constexpr const ssize_t CCL_ENV_SIZET_NOT_SPECIFIED = -1;
constexpr const int CCL_ENV_INT_NOT_SPECIFIED = -1;

/**
 * @addtogroup OneCCLvars
 * @{
 */

/**
 * @brief Set this environment variable to control logging level
 *
 * @details The `CCL_LOG_LEVEL` environment variable can be set
 * to control the level of detail in the logging output generated
 * by the CCL library.
 *
 * "<value>": "error", "warn", "info", "debug", "trace"
 *
 * By-default: "warn"
 *
 */
constexpr const char* CCL_LOG_LEVEL = "CCL_LOG_LEVEL";
/** @} */

constexpr const char* CCL_ABORT_ON_THROW = "CCL_ABORT_ON_THROW";
constexpr const char* CCL_QUEUE_DUMP = "CCL_QUEUE_DUMP";
constexpr const char* CCL_SCHED_DUMP = "CCL_SCHED_DUMP";
constexpr const char* CCL_SCHED_PROFILE = "CCL_SCHED_PROFILE";
// maximum amount of time in seconds an entry can spend in update. for debug purpose
constexpr const char* CCL_ENTRY_MAX_UPDATE_TIME_SEC = "CCL_ENTRY_MAX_UPDATE_TIME_SEC";

constexpr const char* CCL_FRAMEWORK = "CCL_FRAMEWORK";

/**
 * @addtogroup OneCCLvars
 * @{
 */

/**
 * @brief Set specify the number of oneCCL worker threads.
  *
 * @details "<value>" - The number of worker threads for oneCCL rank
 *
 * By-default: "1"
 */
constexpr const char* CCL_WORKER_COUNT = "CCL_WORKER_COUNT";
/** @} */

constexpr const char* CCL_WORKER_OFFLOAD = "CCL_WORKER_OFFLOAD";
constexpr const char* CCL_WORKER_WAIT = "CCL_WORKER_WAIT";

/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set to specify cpu affinity for oneCCL worker threads.
 *
 * @details "<value>": "auto", "<cpulist>": \n
 * "auto" - Workers are automatically pinned to last cores of pin domain.
 * Pin domain depends from process launcher. If mpirun from oneCCL package
 * is used then pin domain is MPI process pin domain. Otherwise, pin domain
 * is all cores on the node. \n "<cpulist>" - A comma-separated list of core
 * numbers and/or ranges of core numbers for all local workers, one number
 * per worker. The i-th local worker is pinned to the i-th core in the list.
 * For example 'a','b'-'c'defines list of cores contaning core with number
 *'a' and range of cores with numbers from 'b' to 'c'. The number should
 * not exceed the number of cores available on the system.
 *
 * By-default: "not-specified"
 *
 */
constexpr const char* CCL_WORKER_AFFINITY = "CCL_WORKER_AFFINITY";
/**
 * @brief Set to specify memory affinity for oneCCL worker threads.
 * \n
 * @details "<nodelist>" : \n "auto" - Workers are automatically pinned to
 * NUMA nodes that correspond to CPU affinity of workers. \n A comma-separated
 * list of NUMA node numbers for all local workers, one number per worker.
 * The i-th local worker is pinned to the i-th NUMA node in the list.
 * The number should not exceed the number of NUMA nodes available on the system.
 *
 * By-default: "not-specified"
 *
 */
constexpr const char* CCL_WORKER_MEM_AFFINITY = "CCL_WORKER_MEM_AFFINITY";
/** @} */

constexpr const char* I_MPI_AVAILABLE_CORES_ENV = "I_MPI_PIN_INFO";
constexpr const char* I_MPI_AVAILABLE_CORES_DELIMS = ",x";

/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Select the mechanism to collect ranks while creating a communicator.
 * 
 * @details 
 * "<value>": \n
 * "0" - use default implementation using sockets \n
 * "1" - use mpi \n
 * KVS implemention with sockets is used to collect the rank information
 * while creating communicator by default. \n
 * 
 * By-default: "0"
 */
constexpr const char* CCL_KVS_MODE = "CCL_KVS_MODE";
/**
 * @brief Set the timeout for setting up connections during kvs initialization
 *
 * @details "<timeout>" - Timeout in seconds to use for setting up sockets during kvs initialization
 *
 * By-default: "120"
 */
constexpr const char* CCL_KVS_CONNECTION_TIMEOUT = "CCL_KVS_CONNECTION_TIMEOUT";
/**
 * @brief Set whether to use MPI_Allgather or custom implementation while creating a communicator.
 *
 * @details
 * "<value>": \n
 * "0" - use only custom implementation of allgather \n
 * "1" - use MPI_Allgather whenever possible \n
 *
 * By-default: "1"
 */
constexpr const char* CCL_KVS_MPI_ALLGATHER = "CCL_KVS_MPI_ALLGATHER";
/**
 * @brief Set whether to use mpi ranks directly while creating a communicator.
 * In general, MPI_COMM_WORLD rank does not have to be the same as CCL communicator rank.
 * Therefore, we need to gather MPI rank of all participants while creating a communicator.
 * If user decide to use MPI rank and CCL rank as same, then they can set this CVAR to 1,
 * and CCL can skip gathering MPI rank of all participants and use the information that
 * CCL rank is same as MPI rank while creating communicator.
 *
 * @details
 * "<value>": \n
 * "0" - collect mpi ranks from everyone \n
 * "1" - use mpi ranks directly without collecting \n
 *
 * By-default: "0"
 */
constexpr const char* CCL_KVS_USE_MPI_RANKS = "CCL_KVS_USE_MPI_RANKS";
/** @} */

constexpr const char* CCL_ATL_TRANSPORT = "CCL_ATL_TRANSPORT";
/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set this environment variable to enable the OFI shared memory provider for communication between ranks in the same node of host (CPU) buffers.
 * \n
 * @details
 * Syntax \n
 * CCL_ATL_SHM="<value>"\n
 * \n
 * Arguments\n
 * "<value>"	Description\n
 * 	- 0	Disables OFI shared memory provider (default).\n
 * 	- 1	Enables OFI shared memory provider.\n
 * \n
 * Description\n
 *
 * Set this environment variable to enable the OFI shared memory provider for communication between ranks in the same node of host (CPU) buffers.
 *
 * By-default: "0"
 */
constexpr const char* CCL_ATL_SHM = "CCL_ATL_SHM";
/**  @} */
constexpr const char* CCL_ATL_RMA = "CCL_ATL_RMA";
constexpr const char* CCL_ATL_HMEM = "CCL_ATL_HMEM";
constexpr const char* CCL_ATL_SEND_PROXY = "CCL_ATL_SEND_PROXY";
constexpr const char* CCL_ATL_SYNC_COLL = "CCL_ATL_SYNC_COLL";
constexpr const char* CCL_ATL_EXTRA_EP = "CCL_ATL_EXTRA_EP";
constexpr const char* CCL_ATL_CACHE = "CCL_ATL_CACHE";
/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set this environment variable to enable cache model automatically for synchronous collectives with direct algorithms.
 * @details
 * Syntax \n
 * CCL_ENABLE_AUTO_CACHE="<value>"\n
 * \n
 * Arguments\n
 * "<value>"	Description\n
 * 	- 0	Does not allow enabling cache model automatically. (default)\n
 * 	- 1	Allows enabling cache model automatically.\n
 * \n
 *
 *
 * By-default: "0"
 */
constexpr const char* CCL_ENABLE_AUTO_CACHE = "CCL_ENABLE_AUTO_CACHE";
/**  @} */

#if defined(CCL_ENABLE_MPI) && defined(CCL_ENABLE_OMP)
/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set this environment variable to use openmp threads for synchronous collectives with direct algorithms on host buffers.
 * @details
 * Syntax \n
 * CCL_ENABLE_OMP_COLL="<value>"\n
 * \n
 * Arguments\n
 * "<value>"	Description\n
 * 	- 0	Does not use openmp threads for collectives even the flag for individual collective (for example, CCL_ENABLE_OMP_ALLREDUCE) is set to true.\n
 * 	- 1	Use openmp threads for collectives (default). The flag for individual collective is also needed to be true to use the openMP threads.\n
 * \n
 *
 *
 * By-default: "1"
 */
constexpr const char* CCL_ENABLE_OMP_COLL = "CCL_ENABLE_OMP_COLL";
/**  @} */
constexpr const char* CCL_ENABLE_OMP_ALLREDUCE = "CCL_ENABLE_OMP_ALLREDUCE";
constexpr const char* CCL_ENABLE_OMP_ALLGATHERV = "CCL_ENABLE_OMP_ALLGATHERV";

constexpr const char* CCL_OMP_ALLREDUCE_NUM_THREADS = "CCL_OMP_ALLREDUCE_NUM_THREADS";
#endif // CCL_ENABLE_MPI && CCL_ENABLE_OMP

constexpr const char* CCL_MNIC = "CCL_MNIC";
constexpr const char* CCL_MNIC_NAME = "CCL_MNIC_NAME";
constexpr const char* CCL_MNIC_COUNT = "CCL_MNIC_COUNT";
constexpr const char* CCL_MNIC_OFFSET = "CCL_MNIC_OFFSET";

constexpr const char* CCL_ALGO_FALLBACK = "CCL_ALGO_FALLBACK";
/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set allgather algorithm
 *
 * @details
 * ALLGATHER algorithms
 *  - direct    Based on MPI_Iallgather
 *  - naive     Send to all, receive from all
 *  - ring      Alltoall-based algorithm
 *  - flat      Alltoall-based algorithm
 *  - multi_bcast   Series of broadcast operations with different root ranks
 *  - topo	     Topo scaleup algorithm
 *
 *
 * By-default: "topo", if sycl and l0 are enabled,
 *      otherwise "naive" for ofi or "direct" for mpi; "ring" used as fallback
 */
constexpr const char* CCL_ALLGATHER = "CCL_ALLGATHER";
/**
 * @brief Set allgatherv algorithm
 *
 * @details
 * ALLGATHERV algorithms
 *  - direct    Based on MPI_Iallgatherv
 *  - naive     Send to all, receive from all
 *  - ring      Alltoall-based algorithm
 *  - flat      Alltoall-based algorithm
 *  - multi_bcast   Series of broadcast operations with different root ranks
 *  - topo	     Topo scaleup algorithm
 *
 *
 * By-default: "topo", if sycl and l0 are enabled,
 *      otherwise "naive" for ofi or "direct" for mpi; "ring" used as fallback
 */
constexpr const char* CCL_ALLGATHERV = "CCL_ALLGATHERV";
/**
 * @brief Set allreduce algorithm
 *
 * @details
 * ALLREDUCE algorithms
 *  - direct        Based on MPI_Iallreduce
 *  - rabenseifner  Rabenseifner’s algorithm
 *  - nreduce       May be beneficial for imbalanced workloads
 *  - ring          Reduce_scatter + allgather ring. Use CCL_RS_CHUNK_COUNT
 *      and CCL_RS_MIN_CHUNK_SIZE to control pipelining on reduce_scatter phase.
 *  - double_tree   Double-tree algorithm
 *  - recursive_doubling    Recursive doubling algorithm
 *  - 2d            Two-dimensional algorithm (reduce_scatter + allreduce + allgather).
 *                  Only available for Host (CPU) buffers.
 *  - topo          Topo scaleup algorithm (available if sycl and l0 are enabled)
 *
 *
 * By-default: "topo", if sycl and l0 are enable, otherwise "ring"
 */
constexpr const char* CCL_ALLREDUCE = "CCL_ALLREDUCE";
/**
 * @brief Set alltoall algorithm
 *
 * @details
 * ALLTOALLV algorithms
 *  - direct    Based on MPI_Ialltoallv
 *  - naive     Send to all, receive from all
 *  - scatter   Scatter-based algorithm
 *  - topo	    Topo scaleup algorithm (available if sycl and l0 are enabled)
 *
 * By-default: "topo", if sycl and l0 are enable, otherwise "scatter"
 */
constexpr const char* CCL_ALLTOALL = "CCL_ALLTOALL";
/**
 * @brief Set alltoallv algorithm
 *
 * @details
 * ALLTOALLV algorithms
 *  - direct    Based on MPI_Ialltoallv
 *  - naive     Send to all, receive from all
 *  - topo      Topo scaleup algorithm (available if sycl and l0 are enabled)
 *
 * By-default: "topo", if sycl and l0 are enable, otherwise "scatter"
 */
constexpr const char* CCL_ALLTOALLV = "CCL_ALLTOALLV";
/**
 * @brief Set barrier algorithm
 *
 * @details
 * BARRIER algorithms
 *  - direct    Based on MPI_Ibarrier
 *  - ring      Ring-based algorithm
 *
 * Note: BARRIER does not support the CCL_BARRIER_SCALEOUT environment
 * variable. To change the algorithm for scaleout, use CCL_BARRIER.
 *
 * By-default: "direct"
 */
constexpr const char* CCL_BARRIER = "CCL_BARRIER";
/**
 * @brief Set broadcast algorithm
 *
 * @details
 * BCAST algorithms
 *  - direct        Based on MPI_Ibcast
 *  - ring          Ring
 *  - double_tree   Double-tree algorithm
 *  - naive         Send to all from root rank
 *
 *  Note: BCAST algorithm does not support yet the  CCL_BCAST_SCALEOUT
 * environment variable. To change the algorithm for BCAST, use CCL_BCAST.
 *
 * By-default: "direct"
 */
constexpr const char* CCL_BCAST = "CCL_BCAST";
/**
 * @brief Set broadcast algorithm (send_buf, recv_buf)
 *
 * @details
 * BCAST algorithms
 *  - direct        Based on MPI_Ibcast
 *  - ring          Ring
 *  - double_tree   Double-tree algorithm
 *  - naive         Send to all from root rank
 *
 *  Note: BCAST algorithm does not support yet the  CCL_BCAST_SCALEOUT
 * environment variable. To change the algorithm for BCAST, use CCL_BCAST.
 *
 * By-default: "direct"
 */
constexpr const char* CCL_BROADCAST = "CCL_BROADCAST";
/**
 * @brief Set reduce algorithm
 *
 * @details
 * REDUCE algorithms
 *  - direct        Based on MPI_Ireduce
 *  - rabenseifner  Rabenseifner’s algorithm
 *  - ring          Ring algorithm
 *  - tree          Tree algorithm
 *  - double_tree   Double-tree algorithm
 *  - topo          Topo scaleup algorithm (available if sycl and l0 are enabled)
 *
 * By-default: "topo" if sycl and l0 are enabled,
 *      otherwise tree for ofi transport or direct for mpi
 */
constexpr const char* CCL_REDUCE = "CCL_REDUCE";
/**
 * @brief Set reduce-scatter algorithm
 *
 * @details
 * REDUCE_SCATTER algorithms
 *  - direct    Based on MPI_Ireduce_scatter_block
 *  - naive     Send to all, receive and reduce from all
 *  - ring      Ring-based algorithm. Use CCL_RS_CHUNK_COUNT and CCL_RS_MIN_CHUNK_SIZE to control pipelining.
 *  - topo      Topo algorithm (available if sycl and l0 are enabled, scaleup only)
 *
 * By-default: "topo" if sycl and l0 are enabled,
 *      otherwise naive for ofi transport or direct for mpi
 */
constexpr const char* CCL_REDUCE_SCATTER = "CCL_REDUCE_SCATTER";

/**
 * @brief Set recv algorithm
 *
 * @details
 * RECV algorithms
 *  - direct        Using prepost(d2h-h2d) copies to get host buffers to invoke mpi/ofi->recv()
 *  - topo          Topo scale-up algorithm (available if sycl and l0 are enabled)
 *  - offload       Using device buffers directly into mpi/ofi layer
 *                  skipping prepost copies d2h h2d. By-default used for scale-out.
 *                  Setting extra MPI env vars for getting better performance
 *                  (available if sycl and l0 are enabled)
 *
 * By-default: "topo" if sycl and l0 are enabled,
 *      otherwise offload for ofi/mpi transport
 */
constexpr const char* CCL_RECV = "CCL_RECV";
/**
 * @brief Set send algorithm
 *
 * @details
 * SEND algorithms
 *  - direct        Using prepost(d2h-h2d) copies to get host buffers to invoke mpi/ofi->send()
 *  - topo          Topo scale-up algorithm (available if sycl and l0 are enabled)
 *  - offload       Using device buffers directly into mpi/ofi layer
 *                  skipping prepost copies d2h h2d. By-default used for scale-out.
 *                  Setting extra MPI env vars for getting better performance
 *                  (available if sycl and l0 are enabled)
 *
 * By-default: "topo" if sycl and l0 are enabled,
 *      otherwise offload for ofi/mpi transport
 */
constexpr const char* CCL_SEND = "CCL_SEND";
/** @} */

constexpr const char* CCL_UNORDERED_COLL = "CCL_UNORDERED_COLL";
/*
 * SCALEOUT
 *
 * The following environment variables can be used to select the scaleout algorithm used:
 *
 * Syntax
 *
 * To set a specific algorithm for scaleout for device (GPU)  buffers for the whole message size range:
 *
 * CCL_<coll_name>_SCALEOUT=<algo_name>
 * To set a specific algorithm for scaleout for device (GPU) buffers for a specific message size range:
 *
 * CCL_<coll_name>_SCALEOUT="<algo_name_1>[:<size_range_1>][;<algo_name_2>:<size_range_2>][;...]"
 * Where:
 *
 * <coll_name> is selected from a list of available collective operations.
 *
 * <algo_name> is selected from a list of available algorithms for a specific collective operation.
 *
 * <size_range> is described by the left and the right size borders in a format <left>-<right>.
 * Size is specified in bytes. Use reserved word "max" to specify the maximum message size.
 *
 * oneCCL internally fills algorithm selection table with sensible defaults.
 * User input complements the selection table. To see the actual table values set CCL_LOG_LEVEL=info.
 *
 * Example
 *
 * CCL_ALLREDUCE_SCALEOUT="recursive_doubling:0-8192;rabenseifner:8193-1048576;ring:1048577-max"
 */
/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set scaleout allgather algorithm
 *
 * @details
 * ALLGATHER algorithms
 * - direct        Based on MPI_Iallgather
 * - naive         Send to all, receive from all
 * - ring          Alltoall-based algorithm
 * - flat          Alltoall-based algorithm
 * - multi_bcast   Series of broadcast operations with different root ranks
 * 
 *
 * By-default: "naive" for ofi or "direct" for mpi; "ring" used as fallback
 */
constexpr const char* CCL_ALLGATHER_SCALEOUT = "CCL_ALLGATHER_SCALEOUT";
/**
 * @brief Set scaleout allgatherv algorithm
 *
 * @details
 * ALLGATHERV algorithms
 * - direct        Based on MPI_Iallgatherv
 * - naive         Send to all, receive from all
 * - ring          Alltoall-based algorithm
 * - flat          Alltoall-based algorithm
 * - multi_bcast   Series of broadcast operations with different root ranks
 * 
 *
 * By-default: "naive" for ofi or "direct" for mpi; "ring" used as fallback
 */
constexpr const char* CCL_ALLGATHERV_SCALEOUT = "CCL_ALLGATHERV_SCALEOUT";
/**
 * @brief Set allreduce scaleout algorithm
 *
 * @details
 * ALLREDUCE algorithms
 * - direct         Based on MPI_Iallreduce
 * - rabenseifner   Rabenseifner’s algorithm
 * - nreduce        May be beneficial for imbalanced workloads
 * - ring           Reduce_scatter + allgather ring. Use CCL_RS_CHUNK_COUNT
 *      and CCL_RS_MIN_CHUNK_SIZE to control pipelining on reduce_scatter phase.
 * - double_tree    Double-tree algorithm
 * - recursive_doubling Recursive doubling algorithm
 * - 2d             Two-dimensional algorithm
 *      (reduce_scatter + allreduce + allgather). Only available
 *      for Host (CPU) buffers.
 * 
 *
 * By-default: "ring"
 */
constexpr const char* CCL_ALLREDUCE_SCALEOUT = "CCL_ALLREDUCE_SCALEOUT";
/**
 * @brief Set alltoall scaleout algorithm
 *
 * @details
 * ALLTOALL algorithms
 * - direct     Based on MPI_Ialltoall
 * - naive      Send to all, receive from all
 * - scatter    Scatter-based algorithm
 *
 * By-default: "scatter"
 */
constexpr const char* CCL_ALLTOALL_SCALEOUT = "CCL_ALLTOALL_SCALEOUT";
/**
 * @brief Set alltoallv scaleout algorithm
 *
 * @details
 * ALLTOALLV algorithms
 * - direct     Based on MPI_Ialltoallv
 * - naive      Send to all, receive from all
 * - scatter    Scatter-based algorithm
 *
 * By-default: "scatter"
 */
constexpr const char* CCL_ALLTOALLV_SCALEOUT = "CCL_ALLTOALLV_SCALEOUT";
/**
 * @brief Set reduce scaleout algorithm
 *
 * @details
 * REDUCE algorithms
 * - direct         Based on MPI_Ireduce
 * - rabenseifner   Rabenseifner’s algorithm
 * - ring           Ring algorithm
 * - tree           Tree algorithm
 * - double_tree    Double-tree algorithm
 *
 * By-default: "double_tree"
 */
constexpr const char* CCL_REDUCE_SCALEOUT = "CCL_REDUCE_SCALEOUT";
/**
 * @brief Set reduce-scatter scaleout algorithm
 *
 * @details
 * REDUCE_SCATTER algorithms
 * - direct    Based on MPI_Ireduce_scatter_block
 * - naive     Send to all, receive and reduce from all
 * - ring      Ring-based algorithm. Use CCL_RS_CHUNK_COUNT and CCL_RS_MIN_CHUNK_SIZE to control pipelining.
 *
 * By-default: "naive"
 */
constexpr const char* CCL_REDUCE_SCATTER_SCALEOUT = "CCL_REDUCE_SCATTER_SCALEOUT";
/** @} */

constexpr const char* CCL_FUSION = "CCL_FUSION";
constexpr const char* CCL_FUSION_BYTES_THRESHOLD = "CCL_FUSION_BYTES_THRESHOLD";
constexpr const char* CCL_FUSION_COUNT_THRESHOLD = "CCL_FUSION_COUNT_THRESHOLD";
constexpr const char* CCL_FUSION_CHECK_URGENT = "CCL_FUSION_CHECK_URGENT";
constexpr const char* CCL_FUSION_CYCLE_MS = "CCL_FUSION_CYCLE_MS";

constexpr const char* CCL_PRIORITY = "CCL_PRIORITY";
constexpr const char* CCL_SPIN_COUNT = "CCL_SPIN_COUNT";
constexpr const char* CCL_YIELD = "CCL_YIELD";
constexpr const char* CCL_MAX_SHORT_SIZE = "CCL_MAX_SHORT_SIZE";
constexpr const char* CCL_BCAST_PART_COUNT = "CCL_BCAST_PART_COUNT";
constexpr const char* CCL_CACHE_KEY = "CCL_CACHE_KEY";
constexpr const char* CCL_CACHE_FLUSH = "CCL_CACHE_FLUSH";
constexpr const char* CCL_BUFFER_CACHE = "CCL_BUFFER_CACHE";
constexpr const char* CCL_STRICT_ORDER = "CCL_STRICT_ORDER";
constexpr const char* CCL_STAGING_BUFFER = "CCL_STAGING_BUFFER";
constexpr const char* CCL_OP_SYNC = "CCL_OP_SYNC";
constexpr const char* CCL_OFI_ENABLE_HOSTNAME_SHARING = "CCL_OFI_ENABLE_HOSTNAME_SHARING";
constexpr const char* CCL_OFI_INIT_ENABLE_HOSTNAME_SHARING = "CCL_OFI_INIT_ENABLE_HOSTNAME_SHARING";

constexpr const char* CCL_CHUNK_COUNT = "CCL_CHUNK_COUNT";
constexpr const char* CCL_MIN_CHUNK_SIZE = "CCL_MIN_CHUNK_SIZE";

/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Specifies the size of the intermediate buffer used by oneCCL for collective operations.
 *
 * @details The CCL_ZE_TMP_BUF_SIZE environment variable controls the size of the buffer that is used 
 * for temporary buffers of collective operations in 'topo' algorithms. It has no effect on other 
 * algorithms. Smaller values can reduce memory usage at the 
 * expense of performance for 'topo' algorithms.
 *
 * Syntax
 *
 * CCL_ZE_TMP_BUF_SIZE="<value>"
 *
 * Arguments
 *
 * "<value>"    Description
 *  	- SIZE	The size of the buffer in bytes.
 * 
 * By-default: "536870912"
 */
constexpr const char* CCL_ZE_TMP_BUF_SIZE = "CCL_ZE_TMP_BUF_SIZE";

/**
 * @brief Set to specify maximum number
 * of chunks for reduce_scatter phase in ring allreduce
 *
 *
 * @details "<count>" - Maximum number of chunks for reduce_scatter
 * phase in ring allreduce
 *
 *
 * By-default: "1"
 */
constexpr const char* CCL_RS_CHUNK_COUNT = "CCL_RS_CHUNK_COUNT";
/**
 * @brief Set to specify minimum number of bytes in chunk for
 * reduce_scatter phase in ring allreduce
 *
 *
 * @details "<size>" - Minimum number of bytes in chunk for reduce_scatter
 * phase in ring allreduce. Affects actual value of CCL_RS_CHUNK_COUNT.
 *
 *
 * By-default: "65536"
 */
constexpr const char* CCL_RS_MIN_CHUNK_SIZE = "CCL_RS_MIN_CHUNK_SIZE";
/** @} */

#ifdef CCL_ENABLE_SYCL
// use alternative allgatherv topo algorithm
constexpr const char* CCL_ALLGATHERV_TOPO_LARGE_SCALE = "CCL_ALLGATHERV_TOPO_LARGE_SCALE";
constexpr const char* CCL_ALLGATHERV_TOPO_READ = "CCL_ALLGATHERV_TOPO_READ";
constexpr const char* CCL_ALLTOALLV_TOPO_READ = "CCL_ALLTOALLV_TOPO_READ";
/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set this environment variable to select read or write based device-to-device data copy during the reduce_scatter stage of Allreduce, Reduce, and Reduce-Scatter collectives using device (GPU) buffers.
 *
 * @details
 *
 * Syntax
 * CCL_REDUCE_SCATTER_TOPO_READ="<value>"
 *
 * Arguments
 *
 * "<value>"	Description
 * 	- 1	Uses read based copy to transfer data across GPUs for the reduce_scatter stage of Allreduce, Reduce, and Reduce-Scatter collectives (default).
 * 	- 0	Uses write based copy to transfer data across GPUs for the reduce_scatter stage of Allreduce, Reduce, and Reduce-Scatter collectives.
 *
 * Description
 *
 * Set this environment variable to select read or write based device-to-device data copy during the reduce_scatter stage of Allreduce, Reduce, and Reduce-Scatter collectives using device (GPU) buffers.
 *
 * By-default: "1"
 */
constexpr const char* CCL_REDUCE_SCATTER_TOPO_READ = "CCL_REDUCE_SCATTER_TOPO_READ";

/**
 * @brief Set this environment variable to 1 to enable synchronous dependencies processing for oneCCL operations.
 *
 * @details
 *
 * Syntax
 * CCL_ZE_DEPS_SYNC="<value>"
 *
 * Arguments
 *
 * "<value>"	Description
 * 	- 1	Dependencies of oneCCL operations are processed synchronously.
 * 	- 0	Dependencies of oneCCL operations are processed asynchronously (default), meaning that further L0 submissions are being done while dependencies are in progress. Dependencies are signaling when processed.
 *
 * Description
 *
 * Set this environment variable to 1 to make oneCCL block the thread while previous sycl/L0 submissions are not finished.
 *
 * By-default: "0"
 */
constexpr const char* CCL_ZE_DEPS_SYNC = "CCL_ZE_DEPS_SYNC";

/**
 * @brief Set this environment variable to enable compute kernels for Allreduce, Reduce, and Reduce-Scatter collectives using device (GPU) buffers
 *
 * @details
 *
 * Syntax
 * CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL="<value>"
 *
 * Arguments
 *
 * "<value>"	Description
 * 	- 1	Uses compute kernels to transfer data across GPUs for Allreduce, Reduce, and Reduce-Scatter collectives
 * 	- 0	Uses copy engines to transfer data across GPUs for Allreduce, Reduce, and Reduce-Scatter collectives (default).
 *
 * Description
 *
 * Set this environment variable to enable compute kernels for Allreduce, Reduce, and Reduce-Scatter collectives using device (GPU) buffers
 *
 * By-default: "0"
 */
constexpr const char* CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL = "CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL";
/** @} */
constexpr const char* CCL_ALLGATHERV_MONOLITHIC_KERNEL = "CCL_ALLGATHERV_MONOLITHIC_KERNEL";
/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set this environment variable to enable compute kernels for Allgather collectives using device (GPU) buffers
 *
 * @details
 *
 * Syntax
 *
 * CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL="<value>"
 * Arguments
 *
 * "<value>"	Description
 * 	- 1	Uses compute kernels to transfer data across GPUs for Allgatherv collectives
 * 	- 0	Uses copy engines to transfer data across GPUs for Allgatherv collectives (default)
 *
 * Description
 *
 * Set this environment variable to enable compute kernels for Allgatherv collectives using device (GPU) buffers
 *
 * By-default: "0"
 */
constexpr const char* CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL = "CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL";
/**
 * @brief Set this environment variable to enable compute kernels for Alltoall and Alltoallv collectives using device (GPU) buffers
 *
 * @details
 *
 * Syntax
 *
 * CCL_ALLTOALLV_MONOLITHIC_KERNEL="<value>"
 *
 * Arguments
 *
 * "<value>"	Description
 * 	- 1	Uses compute kernels to transfer data across GPUs for AlltoAll and Alltoallv collectives (default)
 * 	- 0	Uses copy engines to transfer data across GPUs for AlltoAll and Alltoallv collectives
 *
 * Description
 *
 * Set this environment variable to enable compute kernels for Alltoall and Alltoallv collectives using device (GPU) buffers
 *
 * By-default: "1"
 */
constexpr const char* CCL_ALLTOALLV_MONOLITHIC_KERNEL = "CCL_ALLTOALLV_MONOLITHIC_KERNEL";
/** @} */
constexpr const char* CCL_ALLTOALLV_MONOLITHIC_READ_KERNEL = "CCL_ALLTOALLV_MONOLITHIC_READ_KERNEL";
constexpr const char* CCL_REDUCE_MONOLITHIC_KERNEL = "CCL_REDUCE_MONOLITHIC_KERNEL";

/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set this environment variable to enable pipelining implementation for Allgatherv collectives using device (GPU) buffers
 *
 * @details
 *
 * Syntax
 *
 * CCL_ALLGATHERV_PIPE_CHUNK_COUNT="<value>"
 * Arguments
 *
 * "<value>"	Description
 * 	- 0:    (default) Bypasses the chunking/pipelining code and directly calls
 *          the topology-aware code
 * 	- 1:    Calls the pipelining code with a single chunk. Effectively, it has
 *          identical behavior and performance as with "0", but exercises the
 *          chunking code path with a single chunk.
 * 	- 2 or higher:  Divides the message into as many logical parts, or chunks,
 *          as specified. Then, it executes the collective with each logical
 *          chunk. This should allow for several phases of the algorithm to
 *          run in parallel, as long as they don't use the same physical
 *          resource. Effectively, this should increase performance.
 *
 * Description
 *
 * Set this environment variable to enable control how many chunks are used for
 * Allgatherv, pipeline-based collectives using device (GPU) buffers.
 *
 * By-default: "0"
 */
constexpr const char* CCL_ALLGATHERV_PIPE_CHUNK_COUNT = "CCL_ALLGATHERV_PIPE_CHUNK_COUNT";

/**
 * @brief Set this environment variable to enable pipelining implementation for Allreduce collectives using device (GPU) buffers
 *
 * @details
 *
 * Syntax
 *
 * CCL_ALLREDUCE_PIPE_CHUNK_COUNT="<value>"
 * Arguments
 *
 * "<value>"	Description
 * 	- 0:    (default) Bypasses the chunking/pipelining code and directly calls
 *          the topology-aware code
 * 	- 1:    Calls the pipelining code with a single chunk. Effectively, it has
 *          identical behavior and performance as with "0", but exercises the
 *          chunking code path with a single chunk.
 * 	- 2 or higher:  Divides the message into as many logical parts, or chunks,
 *          as specified. Then, it executes the collective with each logical
 *          chunk. This should allow for several phases of the algorithm to
 *          run in parallel, as long as they don't use the same physical
 *          resource. Effectively, this should increase performance.
 *
 * Description
 *
 * Set this environment variable to enable control how many chunks are used for
 * Allreduce pipeline-based collectives using device (GPU) buffers.
 *
 * By-default: "0"
 */
constexpr const char* CCL_ALLREDUCE_PIPE_CHUNK_COUNT = "CCL_ALLREDUCE_PIPE_CHUNK_COUNT";

/**
 * @brief Set this environment variable to enable pipelining implementation for Reduce_Scatter collectives using device (GPU) buffers
 *
 * @details
 *
 * Syntax
 *
 * CCL_REDUCE_SCATTER_PIPE_CHUNK_COUNT="<value>"
 * Arguments
 *
 * "<value>"	Description
 * 	- 0:    (default) Bypasses the chunking/pipelining code and directly calls
 *          the topology-aware code
 * 	- 1:    Calls the pipelining code with a single chunk. Effectively, it has
 *          identical behavior and performance as with "0", but exercises the
 *          chunking code path with a single chunk.
 * 	- 2 or higher:  Divides the message into as many logical parts, or chunks,
 *          as specified. Then, it executes the collective with each logical
 *          chunk. This should allow for several phases of the algorithm to
 *          run in parallel, as long as they don't use the same physical
 *          resource. Effectively, this should increase performance.
 *
 * Description
 *
 * Set this environment variable to enable control how many chunks are used for
 * Reduce_Scatter pipeline-based collectives using device (GPU) buffers.
 *
 * By-default: "0"
 */
constexpr const char* CCL_REDUCE_SCATTER_PIPE_CHUNK_COUNT = "CCL_REDUCE_SCATTER_PIPE_CHUNK_COUNT";

/**
 * @brief Set this environment variable to enable pipelining implementation for Reduce collectives using device (GPU) buffers
 *
 * @details
 *
 * Syntax
 *
 * CCL_REDUCE_PIPE_CHUNK_COUNT="<value>"
 * Arguments
 *
 * "<value>"	Description
 * 	- 0:    (default) Bypasses the chunking/pipelining code and directly calls
 *          the topology-aware code
 * 	- 1:    Calls the pipelining code with a single chunk. Effectively, it has
 *          identical behavior and performance as with "0", but exercises the
 *          chunking code path with a single chunk.
 * 	- 2 or higher:  Divides the message into as many logical parts, or chunks,
 *          as specified. Then, it executes the collective with each logical
 *          chunk. This should allow for several phases of the algorithm to
 *          run in parallel, as long as they don't use the same physical
 *          resource. Effectively, this should increase performance.
 *
 * Description
 *
 * Set this environment variable to enable control how many chunks are used for
 * Reduce pipeline-based collectives using device (GPU) buffers.
 *
 * By-default: "0"
 */
constexpr const char* CCL_REDUCE_PIPE_CHUNK_COUNT = "CCL_REDUCE_PIPE_CHUNK_COUNT";

/** @} */

#endif // CCL_ENABLE_SYCL

constexpr const char* CCL_ALLREDUCE_NREDUCE_BUFFERING = "CCL_ALLREDUCE_NREDUCE_BUFFERING";
constexpr const char* CCL_ALLREDUCE_NREDUCE_SEGMENT_SIZE = "CCL_ALLREDUCE_NREDUCE_SEGMENT_SIZE";

constexpr const char* CCL_ALLREDUCE_2D_CHUNK_COUNT = "CCL_ALLREDUCE_2D_CHUNK_COUNT";
constexpr const char* CCL_ALLREDUCE_2D_MIN_CHUNK_SIZE = "CCL_ALLREDUCE_2D_MIN_CHUNK_SIZE";
constexpr const char* CCL_ALLREDUCE_2D_SWITCH_DIMS = "CCL_ALLREDUCE_2D_SWITCH_DIMS";

constexpr const char* CCL_DTREE_PARTITION_COUNT = "CCL_DTREE_PARTITION_COUNT";

constexpr const char* CCL_CHECK_INPLACE_ALIASING = "CCL_CHECK_INPLACE_ALIASING";

constexpr const char* CCL_ALLTOALL_SCATTER_MAX_OPS = "CCL_ALLTOALL_SCATTER_MAX_OPS";

constexpr const char* CCL_BACKEND = "CCL_BACKEND";

constexpr const char* CCL_KERNEL_PATH = "CCL_KERNEL_PATH";
constexpr const char* CCL_KERNEL_MODULE_CACHE = "CCL_KERNEL_MODULE_CACHE";
constexpr const char* CCL_KERNEL_DEBUG = "CCL_KERNEL_DEBUG";
constexpr const char* CCL_KERNEL_GROUP_SIZE = "CCL_KERNEL_GROUP_SIZE";
constexpr const char* CCL_KERNEL_GROUP_COUNT = "CCL_KERNEL_GROUP_COUNT";
constexpr const char* CCL_KERNEL_MEM_ALIGN = "CCL_KERNEL_MEM_ALIGN";
constexpr const char* CCL_KERNEL_SYNC = "CCL_KERNEL_SYNC";
constexpr const char* CCL_KERNEL_1S_LEAD = "CCL_KERNEL_1S_LEAD";
constexpr const char* CCL_KERNEL_1S_USE_COPY_OPS = "CCL_KERNEL_1S_USE_COPY_OPS";
constexpr const char* CCL_KERNEL_1S_IPC_WA = "CCL_KERNEL_1S_IPC_WA";
constexpr const char* CCL_KERNEL_CLOSE_FD_WA = "CCL_KERNEL_CLOSE_FD_WA";

/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set this environment variable to specify the rank number of the current process in the local host
 *
 * @details
 *
 *  Syntax
 *
 *  CCL_LOCAL_RANK="<value>"
 *
 *  Arguments
 *
 *  "<value>"	Description
 *  	- RANK	Rank number of the current process in the local host
 *
 *  Description
 *
 *  Set this environment variable to specify the rank number of the current process in the local host
 *
 * By-default: N/A; job/process launcher (CCL_PROCESS_LAUNCHER) needs to be used if variable not specified
 */
constexpr const char* CCL_LOCAL_RANK = "CCL_LOCAL_RANK";
/**
 * @brief Set this environment variable to specify the total number of ranks on the local host
 *
 * @details
 *
 *  Syntax
 *
 *  CCL_LOCAL_SIZE="<value>"
 *
 *  Arguments
 *
 *  "<value>"	Description
 *  	- SIZE	Total number of ranks on the local host.
 *
 *  Description
 *
 *  Set this environment variable to specify the total number of ranks on the local host
 *
 * By-default: N/A; job/process launcher (CCL_PROCESS_LAUNCHER) needs to be used if variable not specified
 */
constexpr const char* CCL_LOCAL_SIZE = "CCL_LOCAL_SIZE";

/**
 * @brief Set this environment variable to specify the job launcher to use.
 *
 * @details
 *
 *  Syntax
 *
 *  CCL_PROCESS_LAUNCHER="<value>"
 *
 *  Arguments
 *
 *  "<value>"	Description
 *  	- hydra	Uses the MPI hydra job launcher (default)
 *  	- torchrun	Uses torchrun job launcher
 *  	- pmix	It is used with the PALS job launcher which uses the pmix API,
 *  		so your mpiexec command should look something like this:
 *  		CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi mpiexec -np 2 -ppn 2 --pmi=pmix ...
 *  	- none	No Job launcher is used. In this case, the user needs to specify the values for CCL_LOCAL_SIZE and CCL_LOCAL_RANK
 *
 *  Description
 *
 *  Set this environment variable to specify the job launcher to use.
 *
 * By-default: "hydra"
 */
constexpr const char* CCL_PROCESS_LAUNCHER = "CCL_PROCESS_LAUNCHER";
/** @} */

constexpr const char* CCL_TOPO_ALGO = "CCL_TOPO_ALGO";
constexpr const char* CCL_TOPO_COLOR = "CCL_TOPO_COLOR";
constexpr const char* CCL_TOPO_P2P_ACCESS = "CCL_TOPO_P2P_ACCESS";
constexpr const char* CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK = "CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK";
constexpr const char* CCL_TOPO_WA_FABRIC_VERTEX_CONNECTION_CHECK = "CCL_TOPO_WA_FABRIC_VERTEX_CONNECTION_CHECK";
constexpr const char* CCL_USE_MPI_BCAST_WA = "CCL_USE_MPI_BCAST_WA";
constexpr const char* CCL_USE_ROOT_PRINT_WA = "CCL_USE_ROOT_PRINT_WA";

#ifdef CCL_ENABLE_MPI
constexpr const char* CCL_MPI_LIBRARY_PATH = "CCL_MPI_LIBRARY_PATH";
constexpr const char* CCL_ATL_MPI_BF16 = "CCL_ATL_MPI_BF16";
constexpr const char* CCL_ATL_MPI_FP16 = "CCL_ATL_MPI_FP16";
#endif // CCL_ENABLE_MPI
constexpr const char* CCL_OFI_LIBRARY_PATH = "CCL_OFI_LIBRARY_PATH";

#ifdef CCL_ENABLE_SYCL
constexpr const char* CCL_SYCL_OUTPUT_EVENT = "CCL_SYCL_OUTPUT_EVENT";
constexpr const char* CCL_USE_HMEM = "CCL_USE_HMEM";

constexpr const char* CCL_ZE_BARRIER = "CCL_ZE_BARRIER";
constexpr const char* CCL_ZE_BIDIR_ALGO = "CCL_ZE_BIDIR_ALGO";
constexpr const char* CCL_ZE_CACHE = "CCL_ZE_CACHE";
constexpr const char* CCL_ZE_DEVICE_CACHE_EVICT_SMALLEST = "CCL_ZE_DEVICE_CACHE_EVICT_SMALLEST";
constexpr const char* CCL_ZE_DEVICE_CACHE_UPPER_LIMIT = "CCL_ZE_DEVICE_CACHE_UPPER_LIMIT";
constexpr const char* CCL_ZE_DEVICE_CACHE_NUM_BLOCKS_IN_CHUNK = "CCL_ZE_DEVICE_CACHE_NUM_BLOCKS_IN_CHUNK";
constexpr const char* CCL_ZE_DEVICE_CACHE_POLICY = "CCL_ZE_DEVICE_CACHE_POLICY";
constexpr const char* CCL_ZE_PTR_REGISTER_THRESHOLD = "CCL_ZE_PTR_REGISTER_THRESHOLD";
/**
 * @addtogroup OneCCLvars
 * @{
 */

/**
 * @brief Set this environment variable to enable or disable the caching of IPC handles opened with
 * zeMemOpenIpcHandle().
 *
 * @details This controls whether it caches IPC handles opened with zeMemOpenIpcHandle() on receiver's side.
 * When enabled, it caches opened IPC handles, which can improve performance in certain scenarios.
 * See https://spec.oneapi.io/level-zero/latest/core/PROG.html#memory-1
 * 
 *  CCL_ZE_CACHE_OPEN_IPC_HANDLES="<value>"
 *
 *  "<value>"
 *  	- 0	Disables the caching of opened IPC handles.
 *  	- 1	Enables the caching of opened IPC handles (default).
 *
 * By-default: "1"
 */
constexpr const char* CCL_ZE_CACHE_OPEN_IPC_HANDLES = "CCL_ZE_CACHE_OPEN_IPC_HANDLES";

/**
 * @brief Set this environment variable to specify the per process threshold for caching IPC handles opened with zeMemOpenIpcHandle().
 *
 * @details  This specifies the threshold for caching open IPC handles on receiver's side. When the number
 * of open IPC handles exceeds this threshold, the cache will start evicting handles via LRU from the cache.
 *
 *  CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD="<value>"
 *
 *  "<value>"
 *  	- SIZE	The threshold value for caching open IPC handles.
 *
 * By-default: "1000"
 */
constexpr const char* CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD = "CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD";

/**
 * @brief Set this environment variable to enable or disable the caching of IPC handles obtained with zeMemGetIpcHandle(). 
 *
 * @details This environment variable specifies the threshold for caching get IPC handles on sender's side.
 * When the number of IPC handles obtained with zeMemGetIpcHandle() exceeds this threshold, the cache will start evicting 
 * handles via LRU from the cache.
 * 
 *  CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD="<value>"
 *
 *  "<value>"
 *  	- SIZE	The threshold value for caching get IPC handles.
 *
 * By-default: "1000"
 */
constexpr const char* CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD = "CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD";

/**
 * @brief Set this environment variable to specify the per process threshold for caching IPC handles obtained with zeMemGetIpcHandle().
 *
 * @details This controls whether it caches IPC handles obtained with zeMemGetIpcHandle() on sender's side. When enabled, it caches IPC handles,
 * which can improve performance in certain scenarios. By default, the caching of get IPC handles is enabled.
 * See https://spec.oneapi.io/level-zero/latest/core/PROG.html#memory-1
 *
 *  CCL_ZE_CACHE_GET_IPC_HANDLES="<value>"
 *
 *  "<value>"
 *  	- 0	Disables the caching of get IPC handles.
 *  	- 1	Enables the caching of get IPC handles (default).
 * 
 * By-default: "1"
 */
constexpr const char* CCL_ZE_CACHE_GET_IPC_HANDLES = "CCL_ZE_CACHE_GET_IPC_HANDLES";

/**
 * @brief Set to enable oversubscription in topo fallback stage for
 * all collectives.
 *
 * @details This enviroment variable enables or disables the oversubscription fallback
 * from topo algorithm to copy in/out
 *
 * "<value>" :  "0", "1"
 *
 * By-default: "1"
 */
constexpr const char* CCL_ZE_ENABLE_OVERSUBSCRIPTION_FALLBACK = "CCL_ZE_ENABLE_OVERSUBSCRIPTION_FALLBACK";
/**
 * @brief Set to enable oversubscription throw for all collectives.
 *
 * @details This enviroment variable enables or disables the oversubscription throw check
 *
 * "<value>" :  "0", "1"
 *
 * By-default: "1"
 */
constexpr const char* CCL_ZE_ENABLE_OVERSUBSCRIPTION_THROW = "CCL_ZE_ENABLE_OVERSUBSCRIPTION_THROW";
/** @} */

constexpr const char* CCL_ZE_SERIALIZE = "CCL_ZE_SERIALIZE";

constexpr const char* CCL_ZE_COPY_ENGINE = "CCL_ZE_COPY_ENGINE";
constexpr const char* CCL_ZE_H2D_COPY_ENGINE = "CCL_ZE_H2D_COPY_ENGINE";
constexpr const char* CCL_ZE_D2D_COPY_ENGINE = "CCL_ZE_D2D_COPY_ENGINE";
constexpr const char* CCL_ZE_MAX_COMPUTE_QUEUES = "CCL_ZE_MAX_COMPUTE_QUEUES";
constexpr const char* CCL_ZE_MAX_COPY_QUEUES = "CCL_ZE_MAX_COPY_QUEUES";
// use CCS for intra-card copy if main CE is not available
constexpr const char* CCL_ZE_ENABLE_CCS_FALLBACK_FOR_COPY = "CCL_ZE_ENABLE_CCS_FALLBACK_FOR_COPY";

constexpr const char* CCL_ZE_LIST_DUMP = "CCL_ZE_LIST_DUMP";
constexpr const char* CCL_ZE_QUEUE_INDEX_OFFSET = "CCL_ZE_QUEUE_INDEX_OFFSET";
constexpr const char* CCL_ZE_CLOSE_IPC_WA = "CCL_ZE_CLOSE_IPC_WA";
constexpr const char* CCL_ZE_SINGLE_LIST = "CCL_ZE_SINGLE_LIST";
constexpr const char* CCL_ZE_DISABLE_FAMILY_CHECK = "CCL_ZE_DISABLE_FAMILY_CHECK";
constexpr const char* CCL_ZE_DISABLE_PORT_CHECK = "CCL_ZE_DISABLE_PORT_CHECK";
constexpr const char* CCL_ZE_LIBRARY_PATH = "CCL_ZE_LIBRARY_PATH";
constexpr const char* CCL_ZE_ENABLE = "CCL_ZE_ENABLE";
constexpr const char* CCL_ZE_FINI_WA = "CCL_ZE_FINI_WA";
constexpr const char* CCL_ZE_MULTI_WORKERS = "CCL_ZE_MULTI_WORKERS";

/**
 * @addtogroup OneCCLvars
 * @{
 */
/**
 * @brief Set this environment variable to select read or write based device-to-device data copy for the alltoall collective using device (GPU) buffers.
 *
 * @details
 *
 * Syntax
 * CCL_SYCL_ALLTOALL_PROTOCOL="<value>"
 *
 * Arguments
 *
 * "<value>"	Description
 * 	- read	Uses read based copy to transfer data across GPUs for the alltoall collective.
 * 	- write	Uses write based copy to transfer data across GPUs for the alltoall collective (default).
 *
 * Description
 *
 * Set this environment variable to select read or write based device-to-device data copy for the alltoall collective using device (GPU) buffers.
 *
 * By-default: "write"
 */
constexpr const char* CCL_SYCL_ALLTOALL_PROTOCOL = "CCL_SYCL_ALLTOALL_PROTOCOL";
/**
 * @brief Set the directory path for DRM render devices.
 *
 * @details This environment variable specifies the directory path where DRM render devices are located.
 *
 * Example value: "/custom/path/to/devices/"
 *
 * By-default: "/dev/dri/by-path/"
 */
constexpr const char* CCL_DRMFD_DEV_RENDER_DIR_PATH = "CCL_DRMFD_DEV_RENDER_DIR_PATH";
/**
 * @brief Set the suffix for DRM render device names.
 *
 * @details This environment variable specifies the suffix to be used when searching for DRM render device names.
 *
 * Example value: "-customsuffix"
 *
 * By-default: "-render"
 */
constexpr const char* CCL_DRMFD_DEV_RENDER_SUFFIX = "CCL_DRMFD_DEV_RENDER_SUFFIX";
/** @} */
#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_PMIX
constexpr const char* CCL_PMIX_LIBRARY_PATH = "CCL_PMIX_LIBRARY_PATH";
#endif // CCL_ENABLE_PMIX
constexpr const char* CCL_OFI_ENABLE_PMIX_SUPPORT = "CCL_OFI_ENABLE_PMIX_SUPPORT";

#ifdef CCL_ENABLE_ITT
constexpr const char* CCL_ITT_LEVEL = "CCL_ITT_LEVEL";
#endif // CCL_ENABLE_ITT
constexpr const char* CCL_DEBUG_TIMESTAMPS_LEVEL = "CCL_DEBUG_TIMESTAMPS_LEVEL";

/**
 * @brief Set to enable profiling of the communication operations
 *
 *
 * By-default: "0"
 */
constexpr const char* CCL_PROFILING_ENABLE = "CCL_PROFILING_ENABLE";

constexpr const char* CCL_BF16 = "CCL_BF16";
constexpr const char* CCL_FP16 = "CCL_FP16";
