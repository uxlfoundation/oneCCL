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

#pragma once

#include "oneapi/ccl/config.h"

/*
 * This file uses local .clang-format file in order to have
 * unlimited line length which is required for proper
 * handling in doxygen.
*/

#ifdef CCL_ENABLE_SYCL

/**
 * @brief Experimental OneCCL Environment Variables
 * Functionality of these variables has not been (fully)
 * tested and, therefore, cannot be supported nor guaranteed.
 *
 * @defgroup ExpOneCCLvars Experimental OneCCL Environment Variables
 * @{
 * @}
 **/

/**
 * @addtogroup ExpOneCCLvars
 * @{
 */

/**
 * @brief Set to specify monolithic pipeline approach for
 * reduce_scatter phase in allreduceand reduce collectives.
 * 
 * @details This enviroment variable has the advantage of forming a seamless
 * pipeline that conceals the data transfer time across MDFI. This way,
 * a process reads the data from its peer tile on the same GPU, performs
 * the reduction, and writes to a intermediate buffer located on a different
 * GPU. This modification will cover the time for transferring
 * the data through XeLinks during the reduce-scatter phase in allreduce
 * and reduce collectives.
 * 
 * "<value>" :  "0", "1"
 * 
 * By-default: "1"
 */
constexpr const char* CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL = "CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL";

/**
 * @brief Set to specify the mechanism to use for Level Zero IPC exchange
 * 
 * @details \n "drmfd" - Uses a the DRM mechanism for Level Zero IPC exchange.
 * This is an experimental mechanism that is used with OS kernels previous
 * to SP4. To use the DRM mechanism, the libdrm and drm headers must be available
 * on a system. \n "pidfd" - Uses pidfd mechanism for Level Zero IPC exchange.
 * It requires OS kernel SP4 or above as it requires Linux 5.6 kernel or above \n 
 * "sockets" - Uses socket mechanism for Level Zero IPC exchange. It is usually
 * slower than the other two mechanisms, but can be used for debugging as
 * it is usually available on most systems
 * 
 * "<value>": "drmfd", "pidfd", "sockets"
 * 
 * By-default: "pidfd"
 */
constexpr const char* CCL_ZE_IPC_EXCHANGE = "CCL_ZE_IPC_EXCHANGE";

/**
 * @brief Set to specify the type to use for Level Zero IPC memory handle
 *
 * @details \n "regular" - This will use the default IPC handle provided by
 * Level Zero IPC APIs.\n
 * "fabric" - Uses an explicit FABRIC_ACCESSIBLE flag to create IPC handle from
 * Level Zero. This is an experimental feature and may not work for unsupported
 * network devices
 *
 * "<value>": "regular", "fabric"
 *
  By-default: "regular"
 */
constexpr const char* CCL_ZE_IPC_MEM_HANDLE_TYPE = "CCL_ZE_IPC_MEM_HANDLE_TYPE";

/**
 * @brief Use bdf support for mapping logical to physical devices
 *
 * @details To obtain the physical device id based on the bdf,
 * we need get and then parse the bdf values. Then using those
 * values we can identify the particular device by referencing
 * the appropriate fields in a pci configuration space for
 * pci devices.to utilize bdf for the purpose of mapping logical
 * devices to their corresponding physical devices.
 *
 * "<value>" :  "0", "1"
 *
 * By-default: "1"
 */
constexpr const char* CCL_ZE_DRM_BDF_SUPPORT = "CCL_ZE_DRM_BDF_SUPPORT";

/**
 * @brief Enable sysman API for device discovery using UUID matching
 *
 * @details Controls whether to use zesInit and zesDriverGetDeviceByUuidExp
 * for sysman device discovery. When enabled (1), uses sysman API to match
 * ze_device handles to zes_device handles by UUID. When disabled (0),
 * falls back to legacy index-based mapping or casting of ze_device to zes_device.
 * This is useful for platforms where sysman API is not fully supported.
 *
 * "<value>" :  "0", "1"
 *
 * By-default: "1"
 */
constexpr const char* CCL_USE_ZESINIT = "CCL_USE_ZESINIT";

/**
 * @brief Use the fallback algorithm for reduce_scatter
 *
 * @details The fallback algorithm performs a full allreduce and
 * then copies a subset of its output to the recv buffer.
 * Currently, the fallback algorithm is used for scaleout whereas
 * scaleup uses optimized algorithm.
 *
 * "<value>" :  "0", "1"
 *
 * By-default: "0"
 */
constexpr const char* CCL_REDUCE_SCATTER_FALLBACK_ALGO = "CCL_REDUCE_SCATTER_FALLBACK_ALGO";

/**
 * @brief Automatically tune algorithm protocols based on port count
 *
 * @details Use number of ports to detect the 12 ports system and
 * use write protocols on such systems for collectives. Users can
 * disable this automatic detection and select the protocols manually.
 *
 * "<value>" :  "0", "1"
 *
 * By-default: "1"
 */
constexpr const char* CCL_ZE_AUTO_TUNE_PORTS = "CCL_ZE_AUTO_TUNE_PORTS";

/**
 * @brief Enable switching of read and write protocols for pt2pt topo algorithm
 *
 * @details Control pt2pt read/write protocols.\n Read Protocol:\n
 * It means SEND side is exchanging the handle with RECV side.
 * Then execute the copy operation on the RECV operation side, where the dst buf
 * is the local buffer and the source buffer is the remote buffer.\n
 *
 * Write Protocol:\n
 * it means RECV side is exchanging the handle with SEND side.
 * Execute the copy operation on the SEND operation side, where the dst buf is the
 * remote buffer and the source buffer is the local buffer.
 *\n
 * "<value>" :  "0", "1"
 *\n
 * By-default: "1"
 */
constexpr const char* CCL_ZE_PT2PT_READ = "CCL_ZE_PT2PT_READ";

/**
 * @brief Tunable value for collectives to adjust copy engine indexes
 *
 * @details use 2,4,6 copy engine indexes for host with 6 ports
 * for allreduce, reduce and allgatherv
 * "<value>":
 * "on" - always use write mode with calculated indexes
 * "off" - always disabled
 * "detected" - determined by the logic in detection
 * "undetected" - the default value, used before the logic in
 * detection
 *
 * By-default: "undetected"
 */
constexpr const char* CCL_ZE_TYPE2_TUNE_PORTS = "CCL_ZE_TYPE2_TUNE_PORTS";

/**
 * @brief Switch ccl::barrier() host-sync / host-async options
 *
 * @details Historically ccl::barrier() was always synchronous.
 * That does not match with oneCCL asynchronous concept. Same as other
 * collectives, ccl::barrier() should be host-asynchronous if possible.
 * As it would be too much to change in one moment, we start through
 * experimental variable which introduces the option to make barrier
 * host-asynchronous. Use CCL_BARRIER_SYNC=0 to achieve that.
 *
 * By-default: "1 (SYNC)"
 */
constexpr const char* CCL_BARRIER_SYNC = "CCL_BARRIER_SYNC";
/**
 * @brief Enable SYCL kernels
 *
 * @details Setting this environment variable to 1 enables SYCL kernel-based
 * implementation for allgatherv, allreduce, and reduce_scatter. Support includes
 * all message sizes and some data types (int32, fp32, fp16, and bf16), sum operation,
 * and single node. oneCCL falls back to other implementations when the support is not
 * available with SYCL kernels, so the user can safely setup this environment variable.
 *
 * "<value>" :  "0", "1"
 *
 * By-default: "0 (disabled)"
 */
constexpr const char* CCL_ENABLE_SYCL_KERNELS = "CCL_ENABLE_SYCL_KERNELS";

/**
 * @brief Enable the use of persistent intermediate buffer in allgatherv
 *
 * @details Setting this environment variable to 1 enables the use of a persistent intermediate
 * buffer to perform the allgatherv operation. This implementation makes the collective fully
 * asynchronous but adds some additional overhead due to the extra copy of the user buffer
 * to a persistent intermediate buffer.
 *
 * "<value>" : "0", "1"
 *
 * By-default: "0 (disabled)"
 */
constexpr const char* CCL_SYCL_ALLGATHERV_TMP_BUF = "CCL_SYCL_ALLGATHERV_TMP_BUF";

/**
 * @brief Specify the threshold for the small size algorithm in allgatherv
 *
 * @details Set the threshold in bytes to specify the small size algorithm in the allgatherv
 * collective. Default value is 131072. "<value>" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD = "CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD";

/**
 * @brief Specify the threshold for the medium size algorithm in allgatherv
 *
 * @details Set the threshold in bytes to specify the medium size algorithm in the allgatherv
 * collective. Default value is 2097152. "<value>" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_ALLGATHERV_MEDIUM_THRESHOLD = "CCL_SYCL_ALLGATHERV_MEDIUM_THRESHOLD";

/**
 * @brief Specify the threshold for the scaleout algorithm in allgatherv
 *
 * @details Set the threshold in bytes to specify the scaleout algorithm in the allgatherv
 * collective. Default value is 1048576. "<value>" : ">=0"
 *
 */
constexpr const char* CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD = "CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD";

/**
 * @brief Enable the use of persistent intermediate buffer in allreduce
 *
 * @details Setting this environment variable to 1 enables the use of a persistent intermediate
 * buffer to perform the allreduce operation. This implementation makes the collective fully
 * asynchronous but adds some additional overhead due to the extra copy of the user buffer
 * to a persistent intermediate buffer.
 *
 * "<value>" : "0", "1"
 *
 * By-default: "0 (disabled)"
 */
constexpr const char* CCL_SYCL_ALLREDUCE_TMP_BUF = "CCL_SYCL_ALLREDUCE_TMP_BUF";

/**
 * @brief Specify the threshold for the small size algorithm in allreduce
 *
 * @details Set the threshold in bytes to specify the small size algorithm in the allreduce
 * collective. Default value is 524288. "<value>" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD = "CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD";

// CCL_SYCL_ALLREDUCE_MEDIUM_THRESHOLD
/**
 * @brief Specify the threshold for the medium size algorithm in allreduce
 *
 * @details Set the threshold in bytes to specify the medium size algorithm in the allreduce
 * collective. Default value is 16777216. "<value>" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_ALLREDUCE_MEDIUM_THRESHOLD = "CCL_SYCL_ALLREDUCE_MEDIUM_THRESHOLD";

/**
 * @brief Specify the maximum threshold for the Allreduce Sycl scale-out algorithm
 *
 * @details Set the threshold in bytes to specify the Sycl scaleout algorithm in the allreduce
 * collective. Default value is 1048576. "<value>" : ">=0"
 *
 */
constexpr const char* CCL_SYCL_ALLREDUCE_SCALEOUT_THRESHOLD = "CCL_SYCL_ALLREDUCE_SCALEOUT_THRESHOLD";

/**
 * @brief Specify allreduce SYCL scale-out algorithm
 *
 * @details Set the algorithm string from a list of available algorithms to set 
 * a specific algorithm for scale-out phase.
 * ALLREDUCE algorithms
 * - auto           Automatic selection. Default vaue.
 * - direct         Based on MPI_Iallreduce
 * - rabenseifner   Rabenseifner’s algorithm
 * - ring           Reduce_scatter + allgather ring
 *
 */
constexpr const char* CCL_SYCL_ALLREDUCE_SCALEOUT = "CCL_SYCL_ALLREDUCE_SCALEOUT";

constexpr const char* CCL_SYCL_ALLREDUCE_ARC = "CCL_SYCL_ALLREDUCE_ARC";
constexpr const char* CCL_SYCL_ALLREDUCE_LL_THRESHOLD = "CCL_SYCL_ALLREDUCE_LL_THRESHOLD";
constexpr const char* CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD = "CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD";
constexpr const char* CCL_SYCL_ALLREDUCE_SIMPLE_READ = "CCL_SYCL_ALLREDUCE_SIMPLE_READ";
constexpr const char* CCL_SYCL_ALLREDUCE_CHUNKING_THRESHOLD = "CCL_SYCL_ALLREDUCE_CHUNKING_THRESHOLD";
/**
 * @brief Specify allreduce SYCL scale-up LL transmit algorithm per message-size range
 *
 * @details Selects the LL256 transmit kernel used on the SYCL allreduce
 * scale-up path. Allows per-message-size selection between ring, twoshots
 * and oneshot transmits.
 *
 * Supported algorithms:
 * - ring       Pipelined ring transmit. Best throughput for large messages.
 * - twoshots   Two-shot reduce-scatter + allgather. Balanced for medium sizes.
 * - oneshot    One-shot transmit. Lowest latency for small messages. Falls
 *              back to ring when the per-rank chunk reaches the safe oneshot
 *              limit (ONESHOT_MAX_MSGSIZE), so requesting oneshot at large
 *              sizes is always safe.
 *
 * Syntax (whole range):
 *   CCL_SYCL_ALLREDUCE_LL=<algo_name>
 * Syntax (per size range, in bytes):
 *   CCL_SYCL_ALLREDUCE_LL="<algo_1>:<lo_1>-<hi_1>;<algo_2>:<lo_2>-<hi_2>;..."
 * Use the reserved word "max" for the upper bound of the largest range.
 *
 * Example:
 *   CCL_SYCL_ALLREDUCE_LL="oneshot:0-4096;twoshots:4097-65536;ring:65537-max"
 *
 * Default: "oneshot:0-256;twoshots:257-2048;ring:2049-max" (empirically tuned
 * on 4x Intel Arc B580 over PCIe). If set to a single token (e.g. "ring") the
 * same algorithm is used for all sizes.
 */
constexpr const char* CCL_SYCL_ALLREDUCE_LL = "CCL_SYCL_ALLREDUCE_LL";

/**
 * @brief Path to a JSON file describing the SYCL fast-path selection policy.
 * @details When set, the file at this path is loaded once at first use and
 * consulted by the sycl_policy engine for scaleup/scaleout algorithm choice.
 * When the variable is unset or the file is unreadable, the engine stays empty and
 * the fast path falls through to env-var-driven defaults
 */
constexpr const char* CCL_SELECTION_POLICY = "CCL_SELECTION_POLICY";

/**
 * @brief Enable the use of persistent intermediate buffer in reduce
 */
constexpr const char* CCL_SYCL_REDUCE_TMP_BUF = "CCL_SYCL_REDUCE_TMP_BUF";

/**
 * @brief The threshold for simple protocol based reduce
 */
constexpr const char* CCL_SYCL_REDUCE_SIMPLE_THRESHOLD = "CCL_SYCL_REDUCE_SIMPLE_THRESHOLD";

/**
 * @brief The threshold for LL based ring algorithm for reduce
 */
constexpr const char* CCL_SYCL_REDUCE_LL_THRESHOLD = "CCL_SYCL_REDUCE_LL_THRESHOLD";

/**
 * @brief Enable read protocol for reduce simple algorithm
 */
constexpr const char* CCL_SYCL_REDUCE_SIMPLE_READ = "CCL_SYCL_REDUCE_SIMPLE_READ";

/**
 * @brief Chunking size for reduce operations
 */
constexpr const char* CCL_SYCL_REDUCE_CHUNKING_THRESHOLD = "CCL_SYCL_REDUCE_CHUNKING_THRESHOLD";

/**
 * @brief Specify the threshold for the Sycl scaleout algorithm in reduce
 *
 * @details Set the threshold in bytes to specify the Sycl scaleout algorithm in the reduce
 * collective. Default value is 4294967296. "<value>" : ">=0"
 *
 */
constexpr const char* CCL_SYCL_REDUCE_SCALEOUT_THRESHOLD = "CCL_SYCL_REDUCE_SCALEOUT_THRESHOLD";

constexpr const char* CCL_SYCL_REDUCE_LL = "CCL_SYCL_REDUCE_LL";

/**
 * @brief Enable the use of persistent intermediate buffer in reduce_scatter
 *
 * @details Setting this environment variable to 1 enables the use of a persistent intermediate
 * buffer to perform the reduce_scatter operation. This implementation makes the collective
 * fully asynchronous but adds some additional overhead due to the extra copy of the user
 * buffer to a persistent intermediate buffer.
 *
 * "<value>" : "0", "1"
 *
 * By-default: "0 (disabled)"
 */
constexpr const char* CCL_SYCL_REDUCE_SCATTER_TMP_BUF = "CCL_SYCL_REDUCE_SCATTER_TMP_BUF";

/**
 * @brief Specify the threshold for the small size algorithm in reduce_scatter
 *
 * @details Set the threshold in bytes to specify the small size algorithm in the reduce_scatter
 * collective. Default value is 2097152. "<value>" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_REDUCE_SCATTER_SMALL_THRESHOLD = "CCL_SYCL_REDUCE_SCATTER_SMALL_THRESHOLD";

/**
 * @brief Specify the threshold for the medium size algorithm in reduce_scatter
 *
 * @details Set the threshold in bytes to specify the medium size algorithm in the reduce_scatter
 * collective. Default value is 67108864. "<value>" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_REDUCE_SCATTER_MEDIUM_THRESHOLD = "CCL_SYCL_REDUCE_SCATTER_MEDIUM_THRESHOLD";

/**
 * @brief Specify the threshold for the Sycl scaleout algorithm in reduce-scatter
 *
 * @details Set the threshold in bytes to specify the Sycl scaleout algorithm in the reduce-scatter
 * collective. Default value is 4294967296. "<value>" : ">=0"
 *
 */
constexpr const char* CCL_SYCL_REDUCE_SCATTER_SCALEOUT_THRESHOLD = "CCL_SYCL_REDUCE_SCATTER_SCALEOUT_THRESHOLD";

/**
 * @brief Specify reduce-scatter SYCL scale-out algorithm
 *
 * @details Set the algorithm string from a list of available algorithms to set
 * a specific algorithm for scale-out phase.
 * REDUCE_SCATTER algorithms
 * - auto           Automatic selection. Default vaue.
 * - direct         Based on MPI_Ireduce_scatter
 * - ring           Ring algorithm
 *
 */
constexpr const char* CCL_SYCL_REDUCE_SCATTER_SCALEOUT = "CCL_SYCL_REDUCE_SCATTER_SCALEOUT";

constexpr const char* CCL_SYCL_REDUCE_SCATTER_LL_THRESHOLD = "CCL_SYCL_REDUCE_SCATTER_LL_THRESHOLD";
constexpr const char* CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD = "CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD";

/**
 * @addtogroup OneCCLvars
 * @{
 *
 * @brief Selects the LL256 transmit kernel for the SYCL reduce-scatter scale-up path
 *
 * @details Allows per-message-size selection between ring and oneshot transmits.
 *
 * Supported algorithms:
 * - ring       Pipelined ring transmit. Best throughput for large messages.
 * - oneshot    One-shot scatter+reduce. Lowest latency for small messages. Falls
 *              back to ring above the compile-time ONESHOT_MAX_MSGSIZE cap, so
 *              requesting oneshot at large sizes is always safe.
 *
 * Syntax (whole range):
 *   CCL_SYCL_REDUCE_SCATTER_LL=<algo_name>
 * Syntax (per size range, in bytes):
 *   CCL_SYCL_REDUCE_SCATTER_LL="<algo_1>:<lo_1>-<hi_1>;<algo_2>:<lo_2>-<hi_2>;..."
 * Use the reserved word "max" for the upper bound of the largest range.
 *
 * Example:
 *   CCL_SYCL_REDUCE_SCATTER_LL="oneshot:0-2048;ring:2049-max"
 *
 * Default: "oneshot:0-256;ring:257-max". If set to a single token (e.g. "ring")
 * the same algorithm is used for all sizes.
 */
constexpr const char* CCL_SYCL_REDUCE_SCATTER_LL = "CCL_SYCL_REDUCE_SCATTER_LL";

/**
 * @addtogroup OneCCLvars
 * @{
 *
 * @brief Selects the LL256 transmit kernel for the SYCL allgather(v) scale-up path
 *
 * @details Allows per-message-size selection between ring and oneshot transmits.
 *
 * Supported algorithms:
 * - ring       Pipelined ring transmit. Best throughput for large messages.
 * - oneshot    One-shot broadcast. Lowest latency for small messages. Falls back
 *              to ring above the compile-time ONESHOT_MAX_MSGSIZE cap, so
 *              requesting oneshot at large sizes is always safe.
 *
 * Syntax (whole range):
 *   CCL_SYCL_ALLGATHERV_LL=<algo_name>
 * Syntax (per size range, in bytes):
 *   CCL_SYCL_ALLGATHERV_LL="<algo_1>:<lo_1>-<hi_1>;<algo_2>:<lo_2>-<hi_2>;..."
 * Use the reserved word "max" for the upper bound of the largest range.
 *
 * Example:
 *   CCL_SYCL_ALLGATHERV_LL="oneshot:0-2048;ring:2049-max"
 *
 * Default: "oneshot:0-256;ring:257-max". If set to a single token (e.g. "ring")
 * the same algorithm is used for all sizes.
 */
constexpr const char* CCL_SYCL_ALLGATHERV_LL = "CCL_SYCL_ALLGATHERV_LL";

/**
 * @brief Specify allgatherv SYCL scale-out algorithm
 *
 * @details Set the algorithm string from a list of available algorithms to set
 * a specific algorithm for scale-out phase.
 * ALLGATHERV algorithms
 * - auto           Automatic selection. Default vaue.
 * - direct         Based on MPI_Iallgatherv
 * - ring           Ring algorithm
 *
 */
constexpr const char* CCL_SYCL_ALLGATHERV_SCALEOUT = "CCL_SYCL_ALLGATHERV_SCALEOUT";

constexpr const char* CCL_SYCL_ALLGATHERV_LL_THRESHOLD = "CCL_SYCL_ALLGATHERV_LL_THRESHOLD";
constexpr const char* CCL_SYCL_ALLGATHERV_LL_ENABLE = "CCL_SYCL_ALLGATHERV_LL_ENABLE";
constexpr const char* CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD = "CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD";
constexpr const char* CCL_SYCL_ALLGATHERV_SCALEOUT_OVERLAP = "CCL_SYCL_ALLGATHERV_SCALEOUT_OVERLAP";
constexpr const char* CCL_SYCL_ALLGATHERV_SCALEOUT_COMM_SIZE = "CCL_SYCL_ALLGATHERV_SCALEOUT_COMM_SIZE";
constexpr const char* CCL_SYCL_ALLGATHERV_OVERLAP_BUF_SIZE = "CCL_SYCL_ALLGATHERV_OVERLAP_BUF_SIZE";

/**
 * @brief Enable the use of persistent temporary buffer in broadcast
 *
 * @details Setting this environment variable to 1 enables the use of a persistent temporary
 * buffer to perform the broadcast operation. This implementation makes the collective
 * fully asynchronous but adds some additional overhead due to the extra copy of the user
 * buffer to a persistent temporary buffer.
 *
 * "<value>" : "0", "1"
 *
 * By-default: "0 (disabled)"
 */
constexpr const char* CCL_SYCL_BROADCAST_TMP_BUF = "CCL_SYCL_BROADCAST_TMP_BUF";

/**
 * @brief Specify the threshold for the small size algorithm in broadcast
 *
 * @details Set the threshold in bytes to specify the small size algorithm in the broadcast
 * collective. Default value is 2097152. "<value>" : ">=0"
 *
 */
constexpr const char* CCL_SYCL_BROADCAST_SMALL_THRESHOLD = "CCL_SYCL_BROADCAST_SMALL_THRESHOLD";

/**
 * @brief Specify the threshold for the Sycl scaleout algorithm in broadcast
 *
 * @details Set the threshold in bytes to specify the Sycl scaleout algorithm in the broadcast
 * collective. Default value is 2097152. "<value>" : ">=0"
 *
 */
constexpr const char* CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD = "CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD";

/**
 * @brief Enable the pipelined-ring algorithm for large broadcast scale-up.
 *
 * @details When set to 1, the SYCL large-broadcast path uses a hop-by-hop pipelined
 * ring instead of the default direct (all-read-from-root) IPC path.  Each rank
 * writes one slice per step to its right neighbour's recv_buf via a pre-exchanged
 * IPC pointer and then participates in a p2p barrier.  Targets PCIe-switch
 * topologies (CRI/BMG) where the one-shot path is bottlenecked by N-1 ranks all
 * reading from root simultaneously.  Requires CCL_SYCL_BROADCAST_TMP_BUF=0
 * (the default); MT communicators always force tmp-buf and therefore never
 * use the ring.  On ARC/BMG, also bypasses broadcast_rt_ring (RT256 LL) for
 * large messages so that large traffic reaches this path.  Small messages are
 * unaffected.
 *
 * "<value>" : "0", "1"
 *
 * By-default: "0 (disabled)"
 */
constexpr const char* CCL_SYCL_BROADCAST_RING = "CCL_SYCL_BROADCAST_RING";

/**
 * @brief Enable empirical auto-selection between ring and direct broadcast paths.
 *
 * @details Applies only when CCL_SYCL_BROADCAST_RING=1.  When set to 1, the
 * runtime ignores the user-configured CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD and
 * consults a built-in (world_size, message_size) lookup table instead.  The
 * table can both decline the ring (route the call back to the baseline IPC
 * path for small messages where the one-shot still wins) and pick a tuned
 * slice-floor threshold when it elects the ring.  See
 * broadcast_ring_auto_policy.hpp for the table itself; re-tune there if the
 * target hardware changes.
 *
 * "<value>" : "0", "1"
 *
 * By-default: "1 (enabled)"
 */
constexpr const char* CCL_SYCL_BROADCAST_RING_AUTO = "CCL_SYCL_BROADCAST_RING_AUTO";

/**
 * @brief Minimum slice size in bytes for the broadcast ring algorithm.
 *
 * @details Applies only when CCL_SYCL_BROADCAST_RING=1.  Acts as a floor on the
 * auto-computed slice size:
 *   desired_slice_bytes = total_bytes / (world * broadcast_ring_auto_chunks_per_rank)
 *   real_slice_bytes    = max(desired_slice_bytes, threshold)   // threshold > 0
 *   real_slice_bytes    = desired_slice_bytes                   // threshold == 0
 * Use this to prevent excessive barrier overhead on small messages without
 * giving up auto-sizing on large ones.
 *
 * "<value>" : ">=0"
 *
 * By-default: "0 (no minimum, pure auto)"
 */
constexpr const char* CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD = "CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD";

/**
 * @brief Enable the copy-engine backend for the broadcast pipelined ring.
 *
 * @details Applies only when CCL_SYCL_BROADCAST_RING=1.
 *   0 = use copy-kernel: SYCL kernel reusing the shared copy_kernel_wrap from
 *       reduce_scatter_large_sycl_ring.hpp (L1WB_L3WB stores, in-order queue
 *       ordering, no manual event chaining).
 *   1 = use copy-engine: q.memcpy (hardware DMA copy engine).
 *
 * "<value>" : "0", "1"
 *
 * By-default: "0 (copy-kernel)"
 */
constexpr const char* CCL_SYCL_BROADCAST_RING_COPY_ENGINE =
    "CCL_SYCL_BROADCAST_RING_COPY_ENGINE";

/**
 * @brief Master switch for Scatter-AllGather (SAG) broadcast on ARC.
 *
 * @details When 0 (default) the code path is disabled entirely and the
 *          legacy chunked broadcast runs for every size. When 1, the
 *          Scatter-AllGather algorithm is enabled on ARC-family cards
 *          and gated by
 *          CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_THRESHOLD below.
 *          Non-ARC families are always unaffected.
 *          "<value>" : "0|1"
 */
constexpr const char* CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_ENABLE =
    "CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_ENABLE";

/**
 * @brief Threshold (bytes) at or above which broadcast scaleout uses the
 *        Scatter-AllGather (SAG) algorithm on ARC-family cards.
 *
 * @details ONLY consulted when CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_ENABLE=1.
 *          Below this size the scatter+allgather setup dominates and the
 *          chunked broadcast wins. At or above it, each rank sends its 1/P-th
 *          chunk over its own NIC in parallel, giving about P times the
 *          effective cross-node bandwidth. Default 67108864 (64 MiB).
 *          "<value>" : ">=0"
 */
constexpr const char* CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_THRESHOLD =
    "CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_THRESHOLD";

/**
 * @brief Force-mode switch for broadcast Scatter-AllGather (SAG).
 *
 * @details When 0 (default), unsupported shapes return done=false and the
 *          caller may use the legacy broadcast path. When 1, unsupported
 *          shapes throw so benchmarks cannot silently measure fallback.
 *          "<value>" : "0|1"
 */
constexpr const char* CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_FORCE =
    "CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_FORCE";

/**
 * @brief Maximum local P2P-domain size allowed for normal
 *        Scatter-AllGather (SAG) selection.
 *
 * @details The local P2P-domain size is the number of ranks that will take
 *          part in the local scatter/allgather phases of the algorithm.
 *          When this value is 0 (default), this guard is disabled.
 *
 *          When this value is greater than 0:
 *          - if local P2P-domain size <= value, this guard allows SAG to
 *            continue evaluating the remaining selection checks;
 *          - if local P2P-domain size > value, normal mode returns done=false
 *            so the caller uses the legacy broadcast path instead.
 *
 *          This guard exists because larger local domains do more local
 *          scatter/allgather work and synchronization. For those shapes the
 *          SAG setup cost can dominate or flatten performance, even when the
 *          cross-domain rail broadcast itself is useful. Without this guard,
 *          normal ENABLE+THRESHOLD selection could choose SAG for such shapes
 *          and silently benchmark a slower path as the default candidate.
 *          MAX_PPN says: do not automatically use SAG when the local group is
 *          too big. Example: if SAG is good for ppn 2/4/8 but bad or unstable
 *          for ppn 16, set:
 *          CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_MAX_PPN=8
 *          Then normal runs protect themselves from choosing SAG at ppn 16,
 *          while FORCE=1 can still test ppn 16 deliberately.
 *
 *          Force=1 mode ignores this guard so experiments can measure the raw
 *          SAG behavior instead of the guarded production-style selection.
 *          "<value>" : ">=0"
 */
constexpr const char* CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_MAX_PPN =
    "CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_MAX_PPN";

/**
 * @brief Toggle the tmp-buffer staged variant of the scatter large kernel.
 *
 * @details When set to 1, the SYCL scatter large kernel uses a staging buffer
 * for the data path instead of direct IPC peer reads. The staged path is
 * required on platforms where L0 IPC peer pointers are unavailable or do not
 * meet the alignment required by the IPC fast path.
 *
 * "<value>" : "0", "1"
 *
 * By-default: "0 (disabled)"
 */
constexpr const char* CCL_SYCL_SCATTER_LARGE_TMP_BUF = "CCL_SYCL_SCATTER_LARGE_TMP_BUF";

/**
 * @brief Threshold (bytes) below which scatter uses the small SYCL kernel.
 *
 * @details Set the threshold in bytes for the scatter small-kernel selection.
 * "<value>" : ">=0"
 *
 * By-default: "524288"
 */
constexpr const char* CCL_SYCL_SCATTER_SMALL_THRESHOLD = "CCL_SYCL_SCATTER_SMALL_THRESHOLD";

/**
 * @brief Byte threshold above which scatter falls back from SYCL scaleout.
 *
 * "<value>" : ">=0"; default "4294967296"
 */
constexpr const char* CCL_SYCL_SCATTER_SCALEOUT_THRESHOLD = "CCL_SYCL_SCATTER_SCALEOUT_THRESHOLD";

/**
 * @brief Per-rank byte threshold selecting the multi-node scatter variant:
 * at/below picks the unchunked small-msg path, above picks the chunked
 * large-msg path (with optional cross-chunk overlap).
 *
 * "<value>" : ">=0"; default "131072"
 */
constexpr const char* CCL_SYCL_SCATTER_SCALEOUT_SMALL_THRESHOLD =
    "CCL_SYCL_SCATTER_SCALEOUT_SMALL_THRESHOLD";

/**
 * @brief Enable pipelined overlap in the large-msg multi-node scatter by
 * using two in-order queues so per-chunk device<->host copies overlap the
 * previous chunk's intra-node scaleup.
 *
 * "<value>" : "0", "1"; default "1"
 */
constexpr const char* CCL_SYCL_SCATTER_SCALEOUT_OVERLAP = "CCL_SYCL_SCATTER_SCALEOUT_OVERLAP";

/**
 * @brief Cap on the host-scratch region used by the overlapped large-msg
 * multi-node scatter. A smaller cap yields more chunks and more overlap.
 *
 * "<value>" : ">0"; default "134217728"
 */
constexpr const char* CCL_SYCL_SCATTER_SCALEOUT_OVERLAP_BUF_SIZE = "CCL_SYCL_SCATTER_SCALEOUT_OVERLAP_BUF_SIZE";

constexpr const char* CCL_SYCL_ALLGATHERV_CHUNKING_THRESHOLD = "CCL_SYCL_ALLGATHERV_CHUNKING_THRESHOLD";

/**
 * @brief Select the alltoall SYCL scaleout algorithm.
 *
 * "auto" picks by message size and transport. Every algorithm other than "direct"
 * requires GPU RDMA.
 *
 * "<value>" : "auto", "direct", "scatter", "gdr-only-pairwise", "numa-gdr-split";
 * default "auto"
 */
constexpr const char* CCL_SYCL_ALLTOALL_SCALEOUT = "CCL_SYCL_ALLTOALL_SCALEOUT";

constexpr const char* CCL_SYCL_ALLTOALL_ARC_LL = "CCL_SYCL_ALLTOALL_ARC_LL";

constexpr const char* CCL_SYCL_ALLTOALL_TMP_BUF = "CCL_SYCL_ALLTOALL_TMP_BUF";

constexpr const char* CCL_SYCL_ALLTOALL_LL_THRESHOLD = "CCL_SYCL_ALLTOALL_LL_THRESHOLD";

constexpr const char* CCL_SYCL_ALLTOALL_CHUNKING_THRESHOLD = "CCL_SYCL_ALLTOALL_CHUNKING_THRESHOLD";
constexpr const char* CCL_SYCL_ALLTOALL_SINGLE_NODE_ALGORITHM = "CCL_SYCL_ALLTOALL_SINGLE_NODE_ALGORITHM";
constexpr const char* CCL_SYCL_ALLTOALL_USE_BARRIER = "CCL_SYCL_ALLTOALL_USE_BARRIER";
constexpr const char* CCL_SYCL_ALLTOALL_SIMPLE_THRESHOLD = "CCL_SYCL_ALLTOALL_SIMPLE_THRESHOLD";
constexpr const char* CCL_SYCL_ALLTOALL_COPY_ENGINE = "CCL_SYCL_ALLTOALL_COPY_ENGINE";

/** @} */
/** @} */

constexpr const char* CCL_SYCL_CCL_BARRIER = "CCL_SYCL_CCL_BARRIER";
/**
 * @brief Specify whether to disable single-kernel mode.
 *
 * @details single-kernel mode includes local synchronization of all threads.
 * We can disable this and split the single kernel into multiple kernels.
 *
 * "<value>" : "0", "1"
 *
 * By-default: "1 (enabled)"
 */
constexpr const char* CCL_SYCL_KERNEL_SYNC = "CCL_SYCL_KERNEL_SYNC";
constexpr const char* CCL_SYCL_SINGLE_NODE_ALGORITHM = "CCL_SYCL_SINGLE_NODE_ALGORITHM";
constexpr const char* CCL_SYCL_AUTO_USE_TMP_BUF = "CCL_SYCL_AUTO_USE_TMP_BUF";
constexpr const char* CCL_SYCL_FORCE_USE_TMP_BUF_SCALEOUT = "CCL_SYCL_FORCE_USE_TMP_BUF_SCALEOUT";
constexpr const char* CCL_SYCL_COPY_ENGINE = "CCL_SYCL_COPY_ENGINE";
constexpr const char* CCL_SYCL_KERNEL_COPY = "CCL_SYCL_KERNEL_COPY";
constexpr const char* CCL_SYCL_ESIMD = "CCL_SYCL_ESIMD";
/*
 * @brief Specify whether to disable use of full vectors (>= 8 bytes)
 *
 * @details When data and count are 4 byte aligned, full vectors are used.
 * We can disable that and just use vectors that are <= 4 bytes if possible.
 *
 * "<value>" : "0", "1"
 *
 * By-default: "1 (enabled)"
 */
constexpr const char* CCL_SYCL_FULL_VECTOR = "CCL_SYCL_FULL_VECTOR";

/*
 * @brief Specify whether to force the usage of the path used for sycl_graph recording
 *
 * @details When client software records oneCCL collective on sycl graph, the recording
 * is detected and a specific path inside oneCCL is taken. This environment forces the usage
 * of this path without recording a collective. This can be used for debugging
 * or performance tuning.
 *
 * "<value>" : "0", "1"
 *
 * By-default: "0 (disabled)"
 */
constexpr const char* CCL_SYCL_FORCE_RECORDING_PATH = "CCL_SYCL_FORCE_RECORDING_PATH";
/*
 * @brief Specify what type of graph is being recorded
 *
 * @details Currently, there is no API to distinguish native graph from sycl graph
 * but the functionality of the two varies; this env variable shows what graph type
 * is currently used, for reasons related to functional coverage and performance
 *
 * "<value>" : "auto", "native", "sycl"
 *
 * By-default: "auto (internal logic is used to distinguish the two)"
 */
constexpr const char* CCL_SYCL_GRAPH_TYPE = "CCL_SYCL_GRAPH_TYPE";
/*
 * @brief Disable oneshot allgather ll with graph recording
 *
 * @details This is a workaround for a hang in some cases
 *
 * "<value>" : "1", "0"
 *
 * By-default: "1 (enable)"
 */
constexpr const char* CCL_SYCL_ONESHOT_ALLGATHER_WA = "CCL_SYCL_ONESHOT_ALLGATHER_WA";
/* @brief Specify what host_task API is used in graph-related collectives
 *
 * @details Selects between legacy API for host tasks, with explicit events,
 * and an experimental event-free API; the "auto" option may use any of the
 * two in any possible configuration and users should not rely on any particular
 * behavior
 *
 * "<value>" : "auto", "event", "eventless"
 *
 * By-default: "auto (internal logic is used to choose the API used)"
 */
constexpr const char* CCL_SYCL_HOST_TASK_TYPE = "CCL_SYCL_HOST_TASK_TYPE";
constexpr const char* CCL_SYCL_KERNEL_MEMCPY_UPSIZE = "CCL_SYCL_KERNEL_MEMCPY_UPSIZE ";
constexpr const char* CCL_SYCL_TMP_BUF_SIZE = "CCL_SYCL_TMP_BUF_SIZE";
constexpr const char* CCL_SYCL_SCALEOUT_HOST_BUF_SIZE = "CCL_SYCL_SCALEOUT_HOST_BUF_SIZE";
constexpr const char* CCL_SYCL_SCALEOUT_DEVICE_BUF_SIZE = "CCL_SYCL_SCALEOUT_DEVICE_BUF_SIZE";
constexpr const char* CCL_SYCL_KERNELS_LINE_SIZE = "CCL_SYCL_KERNELS_LINE_SIZE";
constexpr const char* CCL_SYCL_SCALEOUT_BUF_ALLOC_MODE = "CCL_SYCL_SCALEOUT_BUF_ALLOC_MODE";
constexpr const char* CCL_SYCL_PT2PT_READ = "CCL_SYCL_PT2PT_READ";
constexpr const char* CCL_SYCL_PT2PT_ENABLE = "CCL_SYCL_PT2PT_ENABLE";
constexpr const char* CCL_SYCL_PT2PT_SIMPLE_THRESHOLD = "CCL_SYCL_PT2PT_SIMPLE_THRESHOLD";
constexpr const char* CCL_SYCL_PT2PT_COPY_ENGINE = "CCL_SYCL_PT2PT_COPY_ENGINE";
constexpr const char* CCL_SYCL_MAX_PIPELINE_CHUNK_SIZE = "CCL_SYCL_MAX_PIPELINE_CHUNK_SIZE";
constexpr const char* CCL_SYCL_PIPELINE_CHUNK_SIZE = "CCL_SYCL_PIPELINE_CHUNK_SIZE";
constexpr const char* CCL_SYCL_SIMPLE_SINGLE_KERNEL = "CCL_SYCL_SIMPLE_SINGLE_KERNEL";
constexpr const char* CCL_SYCL_NUM_THREADS = "CCL_SYCL_NUM_THREADS";
constexpr const char* CCL_SYCL_WORK_GROUP_SIZE = "CCL_SYCL_WORK_GROUP_SIZE";
constexpr const char* CCL_FINALIZE_EVENT_GC_THRESHOLD = "CCL_FINALIZE_EVENT_GC_THRESHOLD";

/**
 * @brief specify P2P domain levels
 *
 * @details specify P2P domain levels for doing scale-up. The cooresponding
 * r2r_comm will be chosen for scaleout.
 *
 * "<value>" : "node", "numa", "switch", "none"
 *
 * By-default: "node"
 */
constexpr const char* CCL_SYCL_P2P_LEVEL = "CCL_SYCL_P2P_LEVEL";
/**
 * @brief specify threshold for P2P domain levels auto selection
 *
 * @details specify the threshold for each collective to switch
 * P2P domain based on the message sizes
 *
 * "<value>" : >=0
 *
 * By-default: default is chosen based on a particular customer configuration
 */
constexpr const char* CCL_SYCL_P2P_LEVEL_THRESHOLD = "CCL_SYCL_P2P_LEVEL_THRESHOLD";
constexpr const char* CCL_SYCL_P2P_LEVEL_ALLREDUCE_THRESHOLD = "CCL_SYCL_P2P_LEVEL_ALLREDUCE_THRESHOLD";
constexpr const char* CCL_SYCL_P2P_LEVEL_ALLGATHERV_THRESHOLD = "CCL_SYCL_P2P_LEVEL_ALLGATHERV_THRESHOLD";
constexpr const char* CCL_SYCL_P2P_LEVEL_REDUCE_SCATTER_THRESHOLD = "CCL_SYCL_P2P_LEVEL_REDUCE_SCATTER_THRESHOLD";
constexpr const char* CCL_SYCL_P2P_LEVEL_ALLTOALL_SMALL_THRESHOLD = "CCL_SYCL_P2P_LEVEL_ALLTOALL_SMALL_THRESHOLD";
constexpr const char* CCL_SYCL_P2P_LEVEL_ALLTOALL_THRESHOLD = "CCL_SYCL_P2P_LEVEL_ALLTOALL_THRESHOLD";
constexpr const char* CCL_SYCL_P2P_LEVEL_BROADCAST_THRESHOLD = "CCL_SYCL_P2P_LEVEL_BROADCAST_THRESHOLD";
constexpr const char* CCL_SYCL_GPUS_PER_SWITCH = "CCL_SYCL_GPUS_PER_SWITCH";

/**
 * @addtogroup ExpOneCCLvars
 * @{
 */

/**
 * @brief Specify the number of NUMA nodes per host for NUMA-aware sub-communicators
 *
 * @details Set the number of NUMA nodes per host to enable creation of NUMA-aware
 * sub-communicators. This allows ranks to be grouped by their NUMA node affinity,
 * improving performance by reducing cross-NUMA memory access. The ranks are divided
 * into groups based on this count, creating separate communicators for ranks within
 * the same NUMA node and for cross-NUMA communication.
 *
 * "<value>" : ">= 1"
 *
 * By-default: "1" (disabled)
 */
constexpr const char* CCL_SYCL_NUMA_NODES = "CCL_SYCL_NUMA_NODES";
/** @} */
constexpr const char* CCL_SYCL_NUMA_NODES_SPLIT = "CCL_SYCL_NUMA_NODES_SPLIT";
constexpr const char* CCL_SYCL_SPLIT_NUMA = "CCL_SYCL_SPLIT_NUMA";
constexpr const char* CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA = "CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA";
constexpr const char* CCL_SYCL_ENABLE_DIRECT_GPU_RDMA = "CCL_SYCL_ENABLE_DIRECT_GPU_RDMA";
constexpr const char* CCL_SYCL_PIPELINE_GPU_RDMA = "CCL_SYCL_PIPELINE_GPU_RDMA";

constexpr const char* CCL_SYCL_ENABLE_ALLREDUCE_SCALEOUT_PROXY =
    "CCL_SYCL_ENABLE_ALLREDUCE_SCALEOUT_PROXY";
constexpr const char* CCL_SYCL_ENABLE_REDUCE_SCATTER_SCALEOUT_PROXY =
    "CCL_SYCL_ENABLE_REDUCE_SCATTER_SCALEOUT_PROXY";
constexpr const char* CCL_SYCL_ENABLE_ALLGATHERV_SCALEOUT_PROXY =
    "CCL_SYCL_ENABLE_ALLGATHERV_SCALEOUT_PROXY";
constexpr const char* CCL_SYCL_ENABLE_ALLTOALL_SCALEOUT_PROXY =
    "CCL_SYCL_ENABLE_ALLTOALL_SCALEOUT_PROXY";

/*
 * @brief Specify whether to enable sycl kenels with sub-communicators
 *
 * @details While using sub-communicators provide an option to disable sycl kernels
 *
 * "<value>" : "0", "1"
 *
 * By-default: "1 (enabled)"
 */
constexpr const char* CCL_SYCL_SUB_COMMUICATOR = "CCL_SYCL_SUB_COMMUICATOR";

constexpr const char* CCL_SYCL_FORCE_PCIE = "CCL_SYCL_FORCE_PCIE";

constexpr const char* CCL_SYCL_LL_BUFFER_GLOBAL = "CCL_SYCL_LL_BUFFER_GLOBAL";

/**
 * @brief Rate limiter environment parameters for SYCL communication
 *
 * @details Configure rate limiting for GPU RDMA traffic to control bandwidth consumption
 * and prevent congestion. These parameters work together to implement a token bucket
 * algorithm that limits data transfer rates.
 *
 * - CCL_SYCL_RATE_LIMIT_BYTES_PER_SEC: Target bandwidth in bytes per second (>= 0)
 * - CCL_SYCL_RATE_LIMIT_BURST_BYTES: Maximum burst size in bytes for token bucket
 * - CCL_SYCL_RATE_LIMIT_MAX_ACQUIRE_BYTES: Maximum bytes that can be acquired per request
 * - CCL_SYCL_RATE_LIMIT_REFILL_INTERVAL_NS: Token bucket refill interval in nanoseconds
 * - CCL_SYCL_RATE_LIMIT_DEVICE_CLOCK_HZ: Device clock frequency in Hz for timing calculations
 * - CCL_SYCL_RATE_LIMIT_BURST_DIVISOR: Derive burst as rate/divisor when burst bytes are unset
 * - CCL_SYCL_RATE_LIMIT_BACKOFF_CYCLES: Device spin-wait cycles between retry attempts
 * - CCL_SYCL_RATE_LIMIT_STATS: Enable statistics collection ("0" or "1")
 */
constexpr const char* CCL_SYCL_RATE_LIMIT_BYTES_PER_SEC = "CCL_SYCL_RATE_LIMIT_BYTES_PER_SEC";
constexpr const char* CCL_SYCL_RATE_LIMIT_BURST_BYTES = "CCL_SYCL_RATE_LIMIT_BURST_BYTES";
constexpr const char* CCL_SYCL_RATE_LIMIT_MAX_ACQUIRE_BYTES = "CCL_SYCL_RATE_LIMIT_MAX_ACQUIRE_BYTES";
constexpr const char* CCL_SYCL_RATE_LIMIT_REFILL_INTERVAL_NS = "CCL_SYCL_RATE_LIMIT_REFILL_INTERVAL_NS";
constexpr const char* CCL_SYCL_RATE_LIMIT_DEVICE_CLOCK_HZ = "CCL_SYCL_RATE_LIMIT_DEVICE_CLOCK_HZ";
constexpr const char* CCL_SYCL_RATE_LIMIT_BURST_DIVISOR = "CCL_SYCL_RATE_LIMIT_BURST_DIVISOR";
constexpr const char* CCL_SYCL_RATE_LIMIT_BACKOFF_CYCLES = "CCL_SYCL_RATE_LIMIT_BACKOFF_CYCLES";
constexpr const char* CCL_SYCL_RATE_LIMIT_STATS = "CCL_SYCL_RATE_LIMIT_STATS";

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_UMF)
constexpr const char* CCL_UMF_ENABLE = "CCL_UMF_ENABLE";
constexpr const char* CCL_UMF_LIBRARY_PATH = "CCL_UMF_LIBRARY_PATH";
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE && CCL_ENABLE_UMF
#endif // CCL_ENABLE_SYCL

constexpr const char* CCL_IPC_ALLGATHERV_WA = "CCL_IPC_ALLGATHERV_WA";

/**
 * @brief Enable Level Zero Copy Engine (CE) flow for oneCCL collectives
 *
 * "<value>" : "0", "1"
 *
 * By-default: "0 (disabled)"
 */
constexpr const char* CCL_ENABLE_ZE_CE = "CCL_ENABLE_ZE_CE";

/*
 * @brief Specify whether overlapping collectives should be supported
 *
 * @details This variable allows to disable support for overlapping
 * collectives. It might be useful for debugging of hangs
 * and performance issues.
 *
 * "<value>" : "enable_always", "enable_except_native_graph", "disable"
 *
 * By-default: "enable_except_native_graph"
 */
constexpr const char* CCL_OVERLAPPING_COLLECTIVES = "CCL_OVERLAPPING_COLLECTIVES";
