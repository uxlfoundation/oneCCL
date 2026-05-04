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

#include <cstring>
#include <map>
#include <memory>
#include <vector>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include "common/log/log.hpp"

#ifndef container_of
#define container_of(ptr, type, field) ((type*)((char*)ptr - offsetof(type, field)))
#endif

#ifndef gettid
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#define gettid() syscall(SYS_gettid)
#endif

#define SIZEOFARR(arr) (sizeof(arr) / sizeof(arr[0]))

#define ATL_CACHELINE_LEN     64
#define ATL_REQ_SIZE          16
#define ATL_EP_SIZE           16
#define ATL_PROGRESS_MODE_ENV "ATL_PROGRESS_MODE"
#define ATL_MAX_HOSTNAME_LEN  64

#define DIR_SEP  '/'
#define FILENAME (strrchr(__FILE__, DIR_SEP) ? strrchr(__FILE__, DIR_SEP) + 1 : __FILE__)

/*
 * Dynamically loaded transports must export the following entry point.
 * This is invoked by the ATL framework when the transport library is loaded.
 */

#define ATL_CHECK_STATUS(expr, str) \
    do { \
        if (expr != ATL_STATUS_SUCCESS) { \
            LOG_ERROR(str); \
            return ATL_STATUS_FAILURE; \
        } \
    } while (0)

#define KVS_2_ATL_CHECK_STATUS(expr, str) \
    do { \
        if (expr != KVS_STATUS_SUCCESS) { \
            LOG_ERROR(str); \
            return ATL_STATUS_FAILURE; \
        } \
    } while (0)

#define ATL_CHECK_PTR(ptr, str) \
    do { \
        if (!ptr) { \
            LOG_ERROR(str, ", errno: ", strerror(errno)); \
            return ATL_STATUS_FAILURE; \
        } \
    } while (0)

#define ATL_SET_STR(dst, size, ...) \
    do { \
        if (snprintf(dst, size, __VA_ARGS__) > size) { \
            printf("line too long (must be shorter %d)\n", size); \
            printf(__VA_ARGS__); \
            return ATL_STATUS_FAILURE; \
        } \
    } while (0)

#define ATL_CALL(func, err_action) \
    do { \
        atl_status_t status = func; \
        if (status != FI_SUCCESS) { \
            LOG_ERROR(#func "\n fails with status: ", status); \
            err_action; \
        } \
    } while (0)
// TODO: Why are we using FI_SUCCESS? Shouldn't this be ATL_STATUS_SUCCESS?

#define ATL_CALL_THROW_IF_ERROR(func) \
    do { \
        atl_status_t status = func; \
        if (unlikely(status != ATL_STATUS_SUCCESS)) { \
            CCL_THROW(#func "\n fails with status: ", status); \
        } \
    } while (0)

class ipmi;

typedef struct atl_ctx atl_ctx_t;

typedef enum { ATL_PROGRESS_POLL, ATL_PROGRESS_CHECK } atl_progress_mode_t;

typedef enum { ATL_RA_WAIT, ATL_RA_RUN, ATL_RA_FINALIZE } atl_resize_action_t;

typedef atl_resize_action_t (*atl_resize_fn_t)(int size);

typedef enum {
    ATL_STATUS_SUCCESS,
    ATL_STATUS_FAILURE,
    ATL_STATUS_AGAIN,
    ATL_STATUS_UNSUPPORTED
} atl_status_t;

inline const char* atl_status_to_str(atl_status_t status) {
    switch (status) {
        case ATL_STATUS_SUCCESS: return "SUCCESS";
        case ATL_STATUS_FAILURE: return "FAILURE";
        case ATL_STATUS_UNSUPPORTED: return "UNSUPPORTED";
        default: return "UNKNOWN";
    }
}

typedef enum {
    ATL_DTYPE_INT8,
    ATL_DTYPE_UINT8,
    ATL_DTYPE_INT16,
    ATL_DTYPE_UINT16,
    ATL_DTYPE_INT32,
    ATL_DTYPE_UINT32,
    ATL_DTYPE_INT64,
    ATL_DTYPE_UINT64,

    ATL_DTYPE_FLOAT16,
    ATL_DTYPE_FLOAT32,
    ATL_DTYPE_FLOAT64,

    ATL_DTYPE_BFLOAT16
} atl_datatype_t;

typedef enum {
    ATL_REDUCTION_SUM,
    ATL_REDUCTION_PROD,
    ATL_REDUCTION_MIN,
    ATL_REDUCTION_MAX,
    ATL_REDUCTION_AVG,
    ATL_REDUCTION_CUSTOM
} atl_reduction_t;

typedef enum { ATL_MNIC_NONE, ATL_MNIC_LOCAL, ATL_MNIC_GLOBAL } atl_mnic_t;
typedef enum { ATL_MNIC_OFFSET_NONE, ATL_MNIC_OFFSET_LOCAL_PROC_IDX } atl_mnic_offset_t;

extern std::map<atl_mnic_t, std::string> mnic_type_names;
extern std::map<atl_mnic_offset_t, std::string> mnic_offset_names;

typedef struct atl_attr {
    struct {
        bool enable_shm;
        bool enable_rma;
        bool enable_hmem;
        bool enable_sync_coll;
        bool enable_extra_ep;
        size_t ep_count;
        atl_mnic_t mnic_type;
        std::string mnic_name;
        size_t mnic_count;
        atl_mnic_offset_t mnic_offset;
    } in;
    struct {
        bool enable_shm;
        bool enable_rma;
        bool enable_hmem;
        atl_mnic_t mnic_type;
        size_t mnic_count;
        size_t tag_bits;
        uint64_t max_tag;
        size_t max_order_waw_size;
    } out;
} atl_attr_t;

typedef struct {
    void* buf;
    size_t len;
    uintptr_t local_key;
    uintptr_t remote_key;
} atl_mr_t;

typedef struct atl_proc_coord {
    int global_idx;
    int global_count;
    int local_idx;
    int local_count;
    std::vector<int> global2local_map{};
    size_t hostname_hash;
    void reset() {
        global_idx = 0;
        global_count = 0;
        local_idx = 0;
        local_count = 0;
        hostname_hash = 0;
    }
    void validate(int rank = -1, int size = -1);
    atl_proc_coord() {
        reset();
    }
} atl_proc_coord_t;

typedef struct atl_req {
    int is_completed;
    void* internal[ATL_REQ_SIZE];
    atl_req() : is_completed(0) {
        memset(internal, 0, ATL_REQ_SIZE * sizeof(void*));
    }
} atl_req_t;

typedef struct atl_ep {
    size_t idx;
    atl_proc_coord_t coord;
    void* internal[ATL_EP_SIZE];
    atl_ep() : idx(0) {
        memset(internal, 0, ATL_EP_SIZE * sizeof(void*));
    }
} atl_ep_t;

std::string to_string(atl_mnic_t type);
std::string to_string(atl_mnic_offset_t offset);
std::string to_string(atl_proc_coord_t& coord);
std::string to_string(atl_attr_t& attr);
std::ostream& operator<<(std::ostream& str, const atl_req_t& req);

namespace ccl {
enum class kvs_mode : int { pmi, mpi, pmix_ofi, pmix_ofi_shm };
static std::map<kvs_mode, std::string> kvs_mode_names = {
    std::make_pair(kvs_mode::pmi, "pmi"),
    std::make_pair(kvs_mode::mpi, "mpi"),
    std::make_pair(kvs_mode::pmix_ofi, "pmix_ofi"),
    std::make_pair(kvs_mode::pmix_ofi_shm, "pmix_ofi_shm")
};
} // namespace ccl
