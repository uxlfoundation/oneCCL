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

#ifndef ONECCL_C_TYPES_H
#define ONECCL_C_TYPES_H
/**
 * @file types.h
 * @brief Definitions of types and macros for OneAPI Collective Communications Library (oneCCL)
 */

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>

/** @brief Opaque handle to a oneCCL communicator */
typedef struct onecclComm *onecclComm_t;

/** @brief Opaque handle to a registered communication window */
typedef void *onecclWindow_t;

/**
 * @enum onecclWindowFlags_t
 * @brief Flags controlling window registration behavior
 * @ingroup Types
 */
typedef enum {
    ONECCL_WINDOW_DEFAULT = 0,
    ONECCL_WINDOW_COLL_SYMMETRIC = 1 << 0
} onecclWindowFlags_t;

/** @def ONECCL_CONFIG_UNDEF_INT
 *  @brief Macro representing an undefined integer configuration.
 * @ingroup Macros
 */
#define ONECCL_CONFIG_UNDEF_INT INT_MIN

/** 
 * @def ONECCL_CONFIG_UNDEF_PTR 
 * @brief Macro representing an undefined pointer configuration.
 * @ingroup Macros
 */
#define ONECCL_CONFIG_UNDEF_PTR NULL

/** 
 * @def ONECCL_SPLIT_NOCOLOR 
 * @brief Constant used to indicate that a rank should not participate in a color.
 * @ingroup Macros
 */
#define ONECCL_SPLIT_NOCOLOR (-1)

/** 
 * @def ONECCL_UNDEF_FLOAT 
 * @brief Represents an undefined float configuration value.
 * @ingroup Macros
 */
#define ONECCL_UNDEF_FLOAT (-1.0f)

/** 
 * @def ONECCL_COMM_NULL 
 * @brief Macro representing a null communicator.
 * @ingroup Macros
 */
#define ONECCL_COMM_NULL NULL

/**
 * @enum onecclResult_t
 * @brief Enum for possible result codes for oneCCL functions.
 * @ingroup Types
 */
typedef enum {
    onecclSuccess = 0,
    onecclError = 1,
    onecclSystemError = 2,
    onecclInternalError = 3,
    onecclInvalidArgument = 4,
    onecclInvalidUsage = 5,
    onecclInProgress = 6,
    onecclFailureGPU = 7,
    onecclFailureCPU = 8,
    onecclAllocFailureCPU = 9,
    onecclAllocFailureGPU = 10,
    onecclPluginException = 11,
    onecclNotImplemented
} onecclResult_t;

/**
 * @enum onecclDataType_t
 * @brief Enum for different oneCCL data types
 * @ingroup Types
 */
typedef enum {
    onecclInt8 = 0,
    onecclChar = 0,
    onecclUint8 = 1,
    onecclInt32 = 2,
    onecclInt = 2,
    onecclUint32 = 3,
    onecclInt64 = 4,
    onecclUint64 = 5,
    onecclFloat16 = 6,
    onecclHalf = 6,
    onecclFloat32 = 7,
    onecclFloat = 7,
    onecclFloat64 = 8,
    onecclDouble = 8,
    onecclBfloat16 = 9,
} onecclDataType_t;

/* Pre-defined reduction operations */
typedef enum { onecclNumOps_dummy = 5 } onecclRedOp_dummy_t;

/**
 * @enum onecclRedOp_t
 * @brief Enum for reduction operations in oneCCL
 * @ingroup Types
 */
typedef enum {
    onecclSum = 0,
    onecclProd = 1,
    onecclMax = 2,
    onecclMin = 3,
    onecclAvg = 4,
    /* indicates number of predefined operations */
    onecclNumOps = 5,
    /* indicates maximum valid operation */
    onecclMaxRedOp = 0x7fffffff>>(32-8*sizeof(onecclRedOp_dummy_t))
} onecclRedOp_t;

/**
 * @enum onecclScalarResidence_t
 * @brief Scalar residence for user-defined operations
 * @ingroup Types
 */
typedef enum {
    /* onecclScalarDevice: The scalar is in device-visible memory and will be
     * dereferenced while the collective is running. */
    onecclScalarDevice = 0,

    /* onecclScalarHostImmediate: The scalar is in host-visible memory and will be
     * dereferenced and stored before collective call. */
    onecclScalarHostImmediate = 1
} onecclScalarResidence_t;

/**
 * @enum onecclPluginType_t
 * @brief Enum to specify plugin types.
 * @ingroup Types
 */
typedef enum {
    onecclPluginAny = 0,
    onecclNull = 0xDEADBEEF,
    onecclLegacy = 0xBAAAAAAD,
    onecclLegacyCPU = 0xBAAAAAAE,
    onecclUserBackend = 0xF8F8F8F8
} onecclPluginType_t;

/** 
 * @def ONECCL_UNIQUE_ID_BYTES 
 * @brief Size of the unique ID structure.
 * @ingroup Macros
 */
#define ONECCL_UNIQUE_ID_BYTES 4096

/**
 * @struct onecclUniqueId
 * @brief Structure to store unique communicator identifier details.
 * @ingroup Types
 */
typedef struct {
    union {
        struct {
            char legacy[512]; /**< Legacy identifier. */
            char nccl[512];   /**< NCCL identifier. */
            char any[2048];   /**< Additional space for any identifier. */
        };
        char data[ONECCL_UNIQUE_ID_BYTES];
    };
} onecclUniqueId;

/**
 * @struct onecclConfig
 * @brief Configuration structure for oneCCL communicators.
 * @ingroup Types
 */
typedef struct onecclConfig {
    size_t size;
    unsigned int magic;
    unsigned int version;

    int blocking; /**< Should the created communicator be blocking */
    int cgaClusterSize;
    int minCTAs;
    int maxCTAs;
    const char *netName;
    int splitShare;

    int multiThreaded; /**< Is the communicator created for multi-threaded use */

    onecclPluginType_t plugin;
} onecclConfig_t;

/**
 * @def ONECCL_CONFIG_INITIALIZER
 * @brief Initializer for onecclConfig_t structure with default values.
 * @ingroup Macros
 */
#define ONECCL_CONFIG_INITIALIZER                                              \
    {                                                                          \
        sizeof(onecclConfig_t),      /* size */                                \
            0xcafebeef,              /* magic */                               \
            ONECCL_VERSION_CODE,     /* version */                             \
            ONECCL_CONFIG_UNDEF_INT, /*blocking*/                              \
            ONECCL_CONFIG_UNDEF_INT, /*cgaClusterSize*/                        \
            ONECCL_CONFIG_UNDEF_INT, /*minCTAs*/                               \
            ONECCL_CONFIG_UNDEF_INT, /*maxCTAs*/                               \
            ONECCL_CONFIG_UNDEF_PTR, /*netName*/                               \
            ONECCL_CONFIG_UNDEF_INT, /*splitShare*/                            \
            ONECCL_CONFIG_UNDEF_INT, /*multiThreaded*/                         \
            onecclPluginAny,         /* plugin */                              \
    }

#ifdef __cplusplus
}
#endif

#endif // ONECCL_C_TYPES_H
