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
 * collective. Default value is 131072. "<value>"" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD = "CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD";

/**
 * @brief Specify the threshold for the medium size algorithm in allgatherv
 *
 * @details Set the threshold in bytes to specify the medium size algorithm in the allgatherv
 * collective. Default value is 2097152. "<value>"" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_ALLGATHERV_MEDIUM_THRESHOLD = "CCL_SYCL_ALLGATHERV_MEDIUM_THRESHOLD";

/**
 * @brief Specify the threshold for the scaleout algorithm in allgatherv
 *
 * @details Set the threshold in bytes to specify the scaleout algorithm in the allgatherv
 * collective. Default value is 1048576. "<value>"" : ">=0"
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
 * collective. Default value is 524288. "<value>"" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD = "CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD";

// CCL_SYCL_ALLREDUCE_MEDIUM_THRESHOLD
/**
 * @brief Specify the threshold for the medium size algorithm in allreduce
 *
 * @details Set the threshold in bytes to specify the medium size algorithm in the allreduce
 * collective. Default value is 16777216. "<value>"" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_ALLREDUCE_MEDIUM_THRESHOLD = "CCL_SYCL_ALLREDUCE_MEDIUM_THRESHOLD";

/**
 * @brief Specify the maximum threshold for the Allreduce Sycl scale-out algorithm
 *
 * @details Set the threshold in bytes to specify the Sycl scaleout algorithm in the allreduce
 * collective. Default value is 1048576. "<value>"" : ">=0"
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
 * collective. Default value is 2097152."<value>"" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_REDUCE_SCATTER_SMALL_THRESHOLD = "CCL_SYCL_REDUCE_SCATTER_SMALL_THRESHOLD";

/**
 * @brief Specify the threshold for the medium size algorithm in reduce_scatter
 *
 * @details Set the threshold in bytes to specify the medium size algorithm in the reduce_scatter
 * collective. Default value is 67108864. "<value>"" : ">=0"
 * 
 */
constexpr const char* CCL_SYCL_REDUCE_SCATTER_MEDIUM_THRESHOLD = "CCL_SYCL_REDUCE_SCATTER_MEDIUM_THRESHOLD";

/**
 * @brief Specify the threshold for the Sycl scaleout algorithm in reduce-scatter
 *
 * @details Set the threshold in bytes to specify the Sycl scaleout algorithm in the reduce-scatter
 * collective. Default value is 4294967296. "<value>"" : ">=0"
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
 * collective. Default value is 2097152."<value>"" : ">=0"
 *
 */
constexpr const char* CCL_SYCL_BROADCAST_SMALL_THRESHOLD = "CCL_SYCL_BROADCAST_SMALL_THRESHOLD";

/**
 * @brief Specify the threshold for the Sycl scaleout algorithm in broadcast
 *
 * @details Set the threshold in bytes to specify the Sycl scaleout algorithm in the broadcast
 * collective. Default value is 2097152. "<value>"" : ">=0"
 *
 */
constexpr const char* CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD = "CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD";
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
constexpr const char* CCL_SYCL_TMP_BUF_SIZE = "CCL_SYCL_TMP_BUF_SIZE";
constexpr const char* CCL_SYCL_SCALEOUT_HOST_BUF_SIZE = "CCL_SYCL_SCALEOUT_HOST_BUF_SIZE";
constexpr const char* CCL_SYCL_SCALEOUT_DEVICE_BUF_SIZE = "CCL_SYCL_SCALEOUT_DEVICE_BUF_SIZE";
constexpr const char* CCL_SYCL_KERNELS_LINE_SIZE = "CCL_SYCL_KERNELS_LINE_SIZE";
constexpr const char* CCL_SYCL_SCALEOUT_BUF_ALLOC_MODE = "CCL_SYCL_SCALEOUT_BUF_ALLOC_MODE";
constexpr const char* CCL_SYCL_PT2PT_READ = "CCL_SYCL_PT2PT_READ";
constexpr const char* CCL_SYCL_PT2PT_ENABLE = "CCL_SYCL_PT2PT_ENABLE";
constexpr const char* CCL_SYCL_MAX_PIPELINE_CHUNK_SIZE = "CCL_SYCL_MAX_PIPELINE_CHUNK_SIZE";
constexpr const char* CCL_SYCL_PIPELINE_CHUNK_SIZE = "CCL_SYCL_PIPELINE_CHUNK_SIZE";
constexpr const char* CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA = "CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA";
constexpr const char* CCL_SYCL_ENABLE_DIRECT_GPU_RDMA = "CCL_SYCL_ENABLE_DIRECT_GPU_RDMA";
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

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_UMF)
constexpr const char* CCL_UMF_ENABLE = "CCL_UMF_ENABLE";
constexpr const char* CCL_UMF_LIBRARY_PATH = "CCL_UMF_LIBRARY_PATH";
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE && CCL_ENABLE_UMF
#endif // CCL_ENABLE_SYCL

constexpr const char* CCL_IPC_ALLGATHERV_WA = "CCL_IPC_ALLGATHERV_WA";
