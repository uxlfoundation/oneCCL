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
#include <atomic>
#include <unordered_map>
#include <optional>

#include "atl/atl_base_comm.hpp"
#include "comm/comm_common_attr.hpp"
#include "comm/comm_interface.hpp"
#include "comm/atl_tag.hpp"
#include "common/log/log.hpp"
#include "common/stream/stream.hpp"

#include "common/utils/tree.hpp"
#include "common/utils/utils.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/event.hpp"
#include "oneapi/ccl/coll_attr_ids.hpp"
#include "oneapi/ccl/coll_attr_ids_traits.hpp"
#include "oneapi/ccl/coll_attr.hpp"
#include "coll/algorithms/algorithm_utils.hpp"
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/global/ze/ze_fd_manager.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/ze_handle_exchange_entry.hpp"
#include "coll/algorithms/utils/comm_barrier_data.hpp"
#include "coll/algorithms/utils/consts.hpp"
#include "comm/windows.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
#ifdef CCL_ENABLE_UMF
#include "umf/ipc.hpp"
#endif
#include "types_generator_defines.hpp"
#include "topology/topo_manager.hpp"
#include "unordered_coll/unordered_coll.hpp"

#define ARC_MAX_NUM (32)

enum class pattern_type { collective, send, recv };

// index = local_rank, value = global_rank
using ccl_rank2rank_map = std::vector<int>;

class ikvs_wrapper;

using ll_pattern_t = uint32_t;
using ll_storage_pattern_t = uint16_t;

inline ccl_stream* get_stream_ptr(const ccl::stream::impl_value_t& stream) {
    if (stream.get() && stream->is_sycl_device_stream())
        return stream.get();
    else
        return nullptr;
}

using ccl_rank2rank_map = std::vector<int>;

class ccl_comm;
namespace ccl {
namespace v1 {
class kvs_interface;
}
} // namespace ccl

void comm_barrier(const std::shared_ptr<ccl_comm> comm);

#ifdef CCL_ENABLE_SYCL
sycl::event invoke_barrier(const std::shared_ptr<ccl_comm> comm,
                           sycl::queue q,
                           const std::vector<sycl::event>& dep_events,
                           bool use_cpu);

class SerializedStream {

    void initializeEvent() {
        sync_event = main_queue.value().ext_oneapi_submit_barrier();
    }

    enum class InitState {
        Blank,
        InitStarted,
        InitDone
    };

    std::atomic<InitState> init_state = InitState::Blank;

public:
    std::optional<sycl::queue> main_queue;
    sycl::event sync_event;

    void setMainQueue(sycl::queue q) {
        // initialize in-order queue on the given device, use user-supplied context
        // NOTE: only default context is supported in general
        main_queue = std::optional(sycl::queue(q.get_context(), q.get_device(), {sycl::property::queue::in_order{}}));
        initializeEvent();
    }

    void setMainQueueOnce(sycl::queue q) {
        // lockless, thread-safe init
        InitState expected_found = InitState::Blank;
        bool exchanged = init_state.compare_exchange_strong(expected_found,
                                                            InitState::InitStarted);
        if (exchanged) {
            // initialize within this rank
            setMainQueue(q);
            // release: make the results visible to other threads
            init_state.store(InitState::InitDone, std::memory_order_release);
            LOG_DEBUG("SerializedStream initialized once");
        } else {
            // state is not initial
            // wait until other rank initializes stuff
            // acquire to see the results of the init
            while (expected_found != InitState::InitDone) {
                expected_found = init_state.load(std::memory_order_acquire);
            }
        }
    }

    bool update_check_ever_recorded(sycl::queue q) {
        bool is_recording = ccl::global_data::env().enable_op_sycl_serialize
                            || q.ext_oneapi_get_state() ==
                              sycl::ext::oneapi::experimental::queue_state::recording;

        if (is_recording) {
            this->setMainQueueOnce(q);
            return true;
        } else {
            return this->init_state.load(std::memory_order_relaxed) != InitState::Blank;
        }
    }
    // pre: setMainQueueOnce needs to be called before the first call to `getMainQueue`
    sycl::queue &getMainQueue() {
        return main_queue.value();
    }
};

extern SerializedStream global_serialize_stream;

#endif

// comm-specific environment
// based on global environment
// and adjusted according to comm parameters
class ccl_comm_env {
public:
    ccl_comm_env(std::shared_ptr<ccl::device> device);
    ccl_comm_env(const ccl_comm_env& other) = delete;
    ccl_comm_env& operator=(const ccl_comm_env& other) = delete;
    ~ccl_comm_env() = default;

    std::string to_string() const;

#ifdef CCL_ENABLE_SYCL
    bool get_enable_topo_algo() const {
        return enable_topo_algo;
    }

    ccl::ze::copy_engine_mode get_ze_copy_engine() const {
        return ze_copy_engine;
    }
#endif // CCL_ENABLE_SYCL

private:
    std::shared_ptr<ccl::device> device;

#ifdef CCL_ENABLE_SYCL
    bool enable_topo_algo;
    ccl::ze::copy_engine_mode ze_copy_engine;
    ccl::ze::h2d_copy_engine_mode ze_h2d_copy_engine;

#endif // CCL_ENABLE_SYCL
};

#ifdef CCL_ENABLE_SYCL

struct LLPatternDataInternal {
    ll_storage_pattern_t pattern_array[ARC_MAX_NUM + 1];
    ll_pattern_t tmp_pattern_counter{}; // tb used by the kernel, val before inc
    int32_t pattern_reset_due = 0;
    size_t buf_count;
    bool zero_now;
};

struct LLPatternData {
    LLPatternDataInternal* ll_pattern_data_internal = nullptr;
    // WARNING this class only stores `LLPatternDataInternal`
    // the memory has to be managed externally !
    void init(size_t buf_count) const {
        for (int i = 0; i < ARC_MAX_NUM + 1; i++) {
            ll_pattern_data_internal->pattern_array[i] = 0xa770;
        }
        ll_pattern_data_internal->tmp_pattern_counter = 0;
        ll_pattern_data_internal->pattern_reset_due = 0;
        ll_pattern_data_internal->buf_count = buf_count;
    }

    // This function can be called either from kernel or from CPU
    // but the data placement of ll_pattern_data_internal
    // should correspond to
    // it should be called from a single thread, regardless
    // if it's cpu or gpu
    // once the function finishes, users can query `should_zero_now`
    void process_update(pattern_type type,
                        uint32_t global_current_id,
                        uint32_t own_rank,
                        uint32_t peer_rank,
                        uint32_t nSteps) const {
        // should be compatible with device-side code
        calc_rt_pattern(type, own_rank, peer_rank);
        // sycl::ext::oneapi::experimental::printf(
        //     "type %u, own_rank: %u, peer_rank: %u, nSteps: %u, tmp: %ld\n",
        //  type, own_rank, peer_rank, nSteps, tmp);

        update_rt_pattern(type, peer_rank, nSteps);
    }

    // can be called from cpu or gpu, any number of threads
    // if threre are multiple threads, make sure that all finished
    // the `process_update` function - either a kernel barrier
    // or it.barrier() wth single workgroup is required
    bool should_zero_now() {
        return ll_pattern_data_internal->zero_now;
    }

    uint32_t get_tmp_pattern() {
        return this->ll_pattern_data_internal->tmp_pattern_counter;
    }

    // this function can be called from both: cpu and gpu
    // regardless od data placement
    uint32_t* get_tmp_pattern_ptr() {
        return &this->ll_pattern_data_internal->tmp_pattern_counter;
    }

    // pattern: XYYY xxxx xxxx xxxx
    // X: 1 is collective, 0 is pt2pt
    // YYY: is the source rank of the pt2pt
    void calc_rt_pattern(pattern_type type, uint32_t own_rank, uint32_t peer_rank) const {
        uint32_t pattern{};
        if (type == pattern_type::collective) {
            const uint32_t mask = (1U << (32 - 1)) - 1;
            uint32_t counter = ll_pattern_data_internal->pattern_array[ARC_MAX_NUM];
            pattern = (counter & mask) | (1U << (32 - 1));
        }
        else if (type == pattern_type::send || type == pattern_type::recv) {
            const int pof2 =
                sizeof(unsigned int) * 8 - __builtin_clz((unsigned int)ARC_MAX_NUM) - 1;
            const uint32_t mask = (1 << (32 - 1 - pof2)) - 1;
            // this is GPU code, cannot throw from here
            // former check left for reference
            // CCL_THROW_IF_NOT(peer_rank < ARC_MAX_NUM, "invalid rank: ", peer_rank);
            uint32_t counter = ll_pattern_data_internal->pattern_array[peer_rank];
            int src_rank = type == pattern_type::send ? own_rank : peer_rank;
            pattern = (counter & mask) | src_rank << (32 - 1 - pof2);
        }

        ll_pattern_data_internal->tmp_pattern_counter = pattern;
    }

    void update_rt_pattern(pattern_type type, int peer_rank, uint32_t nSteps) const {
        uint32_t pattern = ll_pattern_data_internal->tmp_pattern_counter;

        uint32_t new_pattern{};
        if (type == pattern_type::collective) {
            const uint32_t mask = (1U << (32 - 1)) - 1;
            // cannot be called within a kernel, left for reference
            // CCL_ASSERT(nSteps <= mask);
            uint32_t counter = pattern & mask;
            counter = (counter + nSteps) & mask;
            // only one thread should handle the pattern
            // this only works because we have a single subgroup:
            // one subgroup -> no races between subgroups
            ll_pattern_data_internal->pattern_array[ARC_MAX_NUM] = counter;
            new_pattern = (counter & mask) | (1U << (32 - 1));
        }
        else if (type == pattern_type::send || type == pattern_type::recv) {
            const int pof2 =
                sizeof(unsigned int) * 8 - __builtin_clz((unsigned int)ARC_MAX_NUM) - 1;
            const int mask = (1 << (32 - 1 - pof2)) - 1;
            uint32_t counter = pattern & mask;
            counter = (counter + nSteps) & mask;
            // cannot be called within a kernel, left for reference
            // CCL_THROW_IF_NOT(peer_rank < ARC_MAX_NUM, "invalid rank: ", peer_rank);
            // only one thread should handle the pattern
            // this only works because we have a single subgroup:
            // one subgroup -> no races between subgroups
            ll_pattern_data_internal->pattern_array[peer_rank] = counter;
            new_pattern = (counter & mask) | (1U << (32 - 1));
        }
        rt_check_pattern(pattern, new_pattern);
    }

private:
    void pattern_reset_set_due() const {
        ll_pattern_data_internal->pattern_reset_due = ll_pattern_data_internal->buf_count;
    }

    int pattern_reset_is_due() const {
        return ll_pattern_data_internal->pattern_reset_due;
    }

    void pattern_reset_performed() const {
        ll_pattern_data_internal->pattern_reset_due--;
    }

    inline void rt_check_pattern(uint32_t seqNo, uint32_t newSeqNo) const {
        if (newSeqNo < seqNo || pattern_reset_is_due()) {
            if (newSeqNo < seqNo) {
                pattern_reset_set_due();
            }
            pattern_reset_performed();
            this->ll_pattern_data_internal->zero_now = true;
        }
        else {
            this->ll_pattern_data_internal->zero_now = false;
        }
    }
};

struct alignas(CACHELINE_SIZE) ccl_large_tmp_bufs {
    // three tmp buffers - 1: work_buf, 2: tmp_send_buf, 3: tmp_recv_buf
    static constexpr int buf_count = 3;
    // tmp_bufs are used as work buf and to copy input/output
    std::array<void*, buf_count> tmp_bufs;
    // ipc exchanged pointers to remote tmp buffers
    std::array<void*, MAX_NODE_RANKS> remote_tmp_bufs[buf_count] = {};
    std::array<void*, MAX_NODE_RANKS> remote_numa_tmp_bufs[buf_count] = {};
    std::array<void*, MAX_GPUS> remote_even_tmp_bufs[buf_count] = {};
    std::array<void*, MAX_TILES> remote_pair_tmp_bufs[buf_count] = {};

    ccl_large_tmp_bufs() {
        // Explicitly zero-initialize all pointers
        tmp_bufs.fill(nullptr);
        for (int i = 0; i < buf_count; i++) {
            remote_tmp_bufs[i].fill(nullptr);
            remote_numa_tmp_bufs[i].fill(nullptr);
            remote_even_tmp_bufs[i].fill(nullptr);
            remote_pair_tmp_bufs[i].fill(nullptr);
        }
    }
};

class alignas(CACHELINE_SIZE) ccl_tmp_bufs {
public:
    ccl_tmp_bufs() {
        set_ptrs_to_null();
        buf_size = 2097152;
    }

    ccl_tmp_bufs(ccl_tmp_bufs& other) = delete;

    ccl_tmp_bufs& operator=(ccl_tmp_bufs& other) = delete;

    ~ccl_tmp_bufs() {
        if (this->free_q) {
            sycl::queue& free_q_ref = *this->free_q.get();

#ifdef CCL_ENABLE_UMF
            if (umf_allocated) {
                // Free memory allocated via UMF
                if (tmp_bufs[0])
                    umf_free(tmp_bufs[0]);
                if (tmp_bufs_gpu[0])
                    umf_free(tmp_bufs_gpu[0]);
                if (tmp_bufs_gpu_index)
                    umf_free(tmp_bufs_gpu_index);
                if (tmp_bufs_gpu_secondary_index)
                    umf_free(tmp_bufs_gpu_secondary_index);
                if (pattern_data_gpu.ll_pattern_data_internal)
                    umf_free(pattern_data_gpu.ll_pattern_data_internal);
            }
            else
#endif
            {
                // sycl::free accepts nullptr, no need to check
                sycl::free(tmp_bufs[0], free_q_ref);
                sycl::free(tmp_bufs_gpu[0], free_q_ref);
                sycl::free(tmp_bufs_gpu_index, free_q_ref);
                sycl::free(tmp_bufs_gpu_secondary_index, free_q_ref);
                sycl::free(pattern_data_gpu.ll_pattern_data_internal, free_q_ref);
            }
            set_ptrs_to_null();
        }
        else if (tmp_bufs[0] || tmp_bufs_gpu[0] || tmp_bufs_gpu_index ||
                 tmp_bufs_gpu_secondary_index) {
            LOG_WARN("internal error, memory leak occurs");
        }
    }
    // to avoid data race towards the end of a collective and starting of
    // next collective we use different buffers on consecutive collectives.
    static constexpr int buf_count = 2;
    // use largest threshold among all the small buffers algorithms
    size_t buf_size;

    int* get_tmp_buf_idx() {
        return tmp_bufs_gpu_index;
    }

    int* get_tmp_buf_secondary_idx() {
        return tmp_bufs_gpu_secondary_index;
    }

    void allocate_buffers(sycl::queue& q, bool tmp_in_device, bool umf_enable = false) {
        this->free_q = std::make_shared<sycl::queue>(q);
        sycl::queue& q_ref = *this->free_q.get();

        char* tmp_buf = nullptr;
        char* tmp_buf_gpu = nullptr;

#ifdef CCL_ENABLE_UMF
        if (umf_enable && tmp_in_device) {
            this->umf_allocated = true;
            tmp_buf = static_cast<char*>(umf_alloc_device_aligned(
                q_ref, CCL_REG_MSG_ALIGNMENT, buf_size * ccl_tmp_bufs::buf_count));
            tmp_buf_gpu = static_cast<char*>(umf_alloc_device_aligned(
                q_ref, CCL_REG_MSG_ALIGNMENT, buf_size * ccl_tmp_bufs::buf_count));
        }
        else
#endif
            if (tmp_in_device) {
            tmp_buf = sycl::aligned_alloc_device<char>(
                CCL_REG_MSG_ALIGNMENT, buf_size * ccl_tmp_bufs::buf_count, q_ref);
            tmp_buf_gpu = sycl::aligned_alloc_device<char>(
                CCL_REG_MSG_ALIGNMENT, buf_size * ccl_tmp_bufs::buf_count, q_ref);
        }
        else {
            // when no p2p, use USM host memory for cross card communication
            tmp_buf = sycl::aligned_alloc_host<char>(
                CCL_REG_MSG_ALIGNMENT, buf_size * ccl_tmp_bufs::buf_count, q_ref);
            tmp_buf_gpu = sycl::aligned_alloc_host<char>(
                CCL_REG_MSG_ALIGNMENT, buf_size * ccl_tmp_bufs::buf_count, q_ref);
        }

        int* tmp_bufs_gpu_index;
        int* tmp_bufs_gpu_secondary_index;
#ifdef CCL_ENABLE_UMF
        if (umf_enable && tmp_in_device) {
            tmp_bufs_gpu_index = static_cast<int*>(umf_alloc_device(q_ref, sizeof(int)));
            tmp_bufs_gpu_secondary_index = static_cast<int*>(umf_alloc_device(q_ref, sizeof(int)));
        }
        else
#endif
        {
            tmp_bufs_gpu_index = sycl::malloc_device<int>(1, q_ref);
            tmp_bufs_gpu_secondary_index = sycl::malloc_device<int>(1, q_ref);
        }

        auto t = q_ref.memset(tmp_bufs_gpu_index, 0, sizeof(*tmp_bufs_gpu_index));
        q_ref.memset(tmp_bufs_gpu_secondary_index, 0, sizeof(*tmp_bufs_gpu_secondary_index), t)
            .wait();

#ifdef CCL_ENABLE_UMF
        if (umf_enable) {
            this->pattern_data_gpu.ll_pattern_data_internal = static_cast<LLPatternDataInternal*>(
                umf_alloc_device(q_ref, sizeof(LLPatternDataInternal)));
        }
        else
#endif
        {
            this->pattern_data_gpu.ll_pattern_data_internal =
                sycl::malloc_device<LLPatternDataInternal>(1, q_ref);
        }
        LLPatternData temp_pattern_data = this->pattern_data_gpu;

        q_ref
            .submit([&](auto& h) { // TODO add deps ?
                h.single_task([temp_pattern_data]() {
                    temp_pattern_data.init(buf_count);
                });
            })
            .wait();
        this->pattern_data.ll_pattern_data_internal = &this->pattern_data_internal_cpu;
        this->pattern_data.init(buf_count);

        this->tmp_bufs_gpu_index = tmp_bufs_gpu_index;
        this->tmp_bufs_gpu_secondary_index = tmp_bufs_gpu_secondary_index;

        for (int i = 0; i < ccl_tmp_bufs::buf_count; i++) {
            void* tmp_buf_ptr = tmp_buf + i * buf_size;
            void* tmp_buf_ptr_gpu = tmp_buf_gpu + i * buf_size;

            this->tmp_bufs[i] = tmp_buf_ptr;
            this->tmp_bufs_gpu[i] = tmp_buf_ptr_gpu;
        }
    }

    void set_remote_tmp_bufs(std::array<void*, MAX_NODE_RANKS> ptrs, int idx) {
        remote_tmp_bufs[idx] = ptrs;
    }

    void set_remote_tmp_bufs_gpu(std::array<void*, MAX_NODE_RANKS> ptrs, int idx) {
        remote_tmp_bufs_gpu[idx] = ptrs;
    }

    int get_next_index() {
        return ++index % buf_count;
    }

    void* get_tmp_buf(int idx) const {
        return tmp_bufs[idx];
    }

    void* get_tmp_buf_gpu(int idx) const {
        return tmp_bufs_gpu[idx];
    }

    size_t get_tmp_buf_size() const {
        return buf_size;
    }

    void set_tmp_buf_size(size_t size) {
        buf_size = size;
    }

    std::array<void*, MAX_NODE_RANKS> get_remote_tmp_buf(int idx) const {
        return remote_tmp_bufs[idx];
    }

    std::array<void*, MAX_NODE_RANKS> get_remote_tmp_buf_gpu(int idx) const {
        return remote_tmp_bufs_gpu[idx];
    }

    std::pair<void*, std::array<void*, MAX_NODE_RANKS>> get_all_tmp_bufs(bool is_next) {
        int idx = is_next ? get_next_index() : index;
        return { get_tmp_buf(idx), get_remote_tmp_buf(idx) };
    }
    std::pair<void*, std::array<void*, MAX_NODE_RANKS>> get_all_tmp_bufs_gpu(bool is_next) {
        // int idx = is_next ? get_next_index() : index;
        int idx = 0;
        return { get_tmp_buf_gpu(idx), get_remote_tmp_buf_gpu(idx) };
    }

    LLPatternData get_pattern_data() {
        return this->pattern_data;
    }

    LLPatternData get_pattern_data_gpu() {
        return this->pattern_data_gpu;
    }

    LLPatternData get_pattern_data_auto(sycl::queue& q) {
        if (use_recording_path(q)) {
            return this->pattern_data_gpu;
        }
        else {
            return this->pattern_data;
        }
    }

private:
    void set_ptrs_to_null() {
        for (int i = 0; i < buf_count; i++) {
            tmp_bufs[i] = NULL;
            tmp_bufs_gpu[i] = NULL;
        }
        tmp_bufs_gpu_index = NULL;
        tmp_bufs_gpu_secondary_index = NULL;
        pattern_data_gpu.ll_pattern_data_internal = NULL;
    }

    std::shared_ptr<sycl::queue> free_q;
    void* tmp_bufs[buf_count];
    std::array<void*, MAX_NODE_RANKS> remote_tmp_bufs[buf_count];
    int *tmp_bufs_gpu_index, *tmp_bufs_gpu_secondary_index;
    void* tmp_bufs_gpu[buf_count];
    std::array<void*, MAX_NODE_RANKS> remote_tmp_bufs_gpu[buf_count];
    LLPatternData pattern_data;
    LLPatternData pattern_data_gpu;
    LLPatternDataInternal pattern_data_internal_cpu;

    int index = 0;
    bool umf_allocated = false;
};

class alignas(CACHELINE_SIZE) ccl_scaleout_host_bufs {
public:
    ccl_scaleout_host_bufs() = default;
    ~ccl_scaleout_host_bufs();
    ccl_scaleout_host_bufs(const ccl_scaleout_host_bufs&) = delete;
    ccl_scaleout_host_bufs& operator=(const ccl_scaleout_host_bufs&) = delete;
    void* get_scaleout_host_buf();
    void put_scaleout_host_buf(const void* buf);
    size_t get_scaleout_host_buf_size();

private:
    static constexpr int buf_count = 3;
    size_t buf_size = 0;
    void* host_bufs[buf_count] = { nullptr, nullptr, nullptr };
    int index = 0;
};

class alignas(CACHELINE_SIZE) ccl_scaleout_device_bufs {
public:
    ccl_scaleout_device_bufs() = default;
    ~ccl_scaleout_device_bufs();
    ccl_scaleout_device_bufs(const ccl_scaleout_device_bufs& other);
    ccl_scaleout_device_bufs& operator=(const ccl_scaleout_device_bufs& other);
    void* get_scaleout_device_buf(sycl::queue& q);
    void put_scaleout_device_buf(void* buf);
    size_t get_scaleout_device_buf_size();

private:
    sycl::device& get_device() const;

    static constexpr int buf_count = 2;
    size_t buf_size = 0;
    void* device_bufs[buf_count] = { nullptr, nullptr };
    int index = 0;
};

class alignas(CACHELINE_SIZE) ccl_scaleout_pipeline_bufs {
public:
    static constexpr int num_chunk_buffs = 3;

    ccl_scaleout_pipeline_bufs() = default;
    ~ccl_scaleout_pipeline_bufs();
    ccl_scaleout_pipeline_bufs(const ccl_scaleout_pipeline_bufs& other);
    ccl_scaleout_pipeline_bufs& operator=(const ccl_scaleout_pipeline_bufs& other);
    void** get_scaleout_send_pipeline_bufs(int num_bufs) {
        allocate_pipe_chunks(num_bufs);
        return (void**)send_pipe_chunks;
    }

    void** get_scaleout_recv_pipeline_bufs(int num_bufs) {
        allocate_pipe_chunks(num_bufs);
        return (void**)recv_pipe_chunks;
    }

private:
    void allocate_pipe_chunks(int num_bufs);

    void* send_pipe_buffer = NULL;
    void* send_pipe_chunks[num_chunk_buffs] = { NULL };
    void* recv_pipe_buffer = NULL;
    void* recv_pipe_chunks[num_chunk_buffs] = { NULL };
};
#endif // CCL_ENABLE_SYCL

// the main purpose of internal comm is to hold
// shareable parts of ccl_comm which don't need to
// be copied/reset on ccl_comm's copy
class alignas(CACHELINE_SIZE) ccl_internal_comm {
public:
    ccl_internal_comm() = delete;
    ccl_internal_comm(const ccl_internal_comm& other) = delete;
    ccl_internal_comm& operator=(const ccl_internal_comm& other) = delete;
    ccl_internal_comm(int comm_id, int rank, int size, std::shared_ptr<atl_base_comm> comm);
    // needed for multithreading (single process multiple devices) approach:
    ccl_internal_comm(int comm_id, int rank, int size);
    ~ccl_internal_comm() = default;

#ifdef CCL_ENABLE_SYCL
    // Group API send/recv structs
    struct group_recv_request {
        void* recv_user_buf;
        atl_req_t atl_recv_req;
        void* recv_host_buf;
        size_t recv_size;
        ccl_comm* comm;
    };

    struct group_send_request {
        atl_req_t atl_send_req;
        void* send_host_buf;
        size_t send_size;
        ccl_comm* comm;
    };
#endif // CCL_ENABLE_SYCL

    int rank() const noexcept {
        return m_rank;
    }

    int size() const noexcept {
        return m_size;
    }

    int pof2() const noexcept {
        return m_pof2;
    }

    const ccl_double_tree& dtree() const {
        return m_dtree;
    }

    void reset(int rank, int size);

#ifdef CCL_ENABLE_SYCL
    bool use_remote_atomics() const {
        return m_barrier_data.use_remote_atomics();
    }

    void set_remote_atomics(bool use_remote_atomics) {
        m_barrier_data.set_remote_atomics(use_remote_atomics);
    }

    ccl_comm_barrier_data barrier_data() const {
        return m_barrier_data;
    }

    ccl_comm_barrier_data barrier_inc(const size_t n) {
        m_barrier_data.inc(n);
        return m_barrier_data;
    };

    void set_barrier_ptrs(std::array<size_t*, MAX_NODE_RANKS> ptrs0,
                          std::array<size_t*, MAX_NODE_RANKS> ptrs1,
                          size_t* count) {
        m_barrier_data.set_remote_ptrs(ptrs0, ptrs1);
        m_barrier_data.set_count_gpu(count);
    }

    ccl_comm_flag_data flag_data() const {
        return m_flag_data;
    }

    void set_flag_ptrs(std::array<size_t*, MAX_NODE_RANKS> ptrs, size_t* count) {
        m_flag_data.set_remote_ptrs(ptrs);
        m_flag_data.set_count(count);
    }

    void* get_tmp_buf(int idx) const {
        return m_tmp_buf.get_tmp_buf(idx);
    }

    void* get_tmp_buf_gpu(int idx) const {
        return m_tmp_buf.get_tmp_buf_gpu(idx);
    }

    size_t get_tmp_buf_size() const {
        return m_tmp_buf.get_tmp_buf_size();
    }

    void set_tmp_buf_size(size_t size) {
        m_tmp_buf.set_tmp_buf_size(size);
    }

    std::pair<void*, std::array<void*, MAX_NODE_RANKS>> get_all_tmp_bufs(bool is_next) {
        return m_tmp_buf.get_all_tmp_bufs(is_next);
    }

    std::pair<void*, std::array<void*, MAX_NODE_RANKS>> get_all_tmp_bufs_gpu(bool is_next) {
        return m_tmp_buf.get_all_tmp_bufs_gpu(is_next);
    }

    void allocate_buffers(sycl::queue& q, bool tmp_in_device, bool umf_enable = false) {
        m_tmp_buf.allocate_buffers(q, tmp_in_device, umf_enable);
    }

    int* get_tmp_buf_idx() {
        return m_tmp_buf.get_tmp_buf_idx();
    }

    int* get_tmp_buf_secondary_idx() {
        return m_tmp_buf.get_tmp_buf_secondary_idx();
    }

    void set_remote_tmp_bufs(std::array<void*, MAX_NODE_RANKS> ptrs, int idx) {
        m_tmp_buf.set_remote_tmp_bufs(ptrs, idx);
    }

    void set_remote_tmp_bufs_gpu(std::array<void*, MAX_NODE_RANKS> ptrs, int idx) {
        m_tmp_buf.set_remote_tmp_bufs_gpu(ptrs, idx);
    }

    ccl_large_tmp_bufs& get_large_tmp_bufs() {
        return m_large_tmp_buf;
    }

    void* get_scaleout_host_buf() {
        return m_scaleout_host_bufs.get_scaleout_host_buf();
    }

    void put_scaleout_host_buf(const void* buf) {
        return m_scaleout_host_bufs.put_scaleout_host_buf(buf);
    }

    size_t get_scaleout_host_buf_size() {
        return m_scaleout_host_bufs.get_scaleout_host_buf_size();
    }

    void* get_scaleout_device_buf(sycl::queue& q) {
        return m_scaleout_device_bufs.get_scaleout_device_buf(q);
    }

    void put_scaleout_device_buf(void* buf) {
        return m_scaleout_device_bufs.put_scaleout_device_buf(buf);
    }

    size_t get_scaleout_device_buf_size() {
        return m_scaleout_device_bufs.get_scaleout_device_buf_size();
    }

    void** get_scaleout_send_pipeline_bufs(int num_bufs) {
        return m_scaleout_pipeline_bufs.get_scaleout_send_pipeline_bufs(num_bufs);
    }

    void** get_scaleout_recv_pipeline_bufs(int num_bufs) {
        return m_scaleout_pipeline_bufs.get_scaleout_recv_pipeline_bufs(num_bufs);
    }

    atl_req_t& get_pipeline_send_req() {
        return pipeline_send_req;
    }

    atl_req_t& get_pipeline_recv_req() {
        return pipeline_recv_req;
    }

    // Group API send/recv request accessors
    std::vector<ccl_internal_comm::group_send_request>& get_group_send_requests() {
        return group_send_requests;
    }

    std::vector<ccl_internal_comm::group_recv_request>& get_group_recv_requests() {
        return group_recv_requests;
    }

    void clear_group_requests() {
        group_send_requests.clear();
        group_recv_requests.clear();
    }

    LLPatternData get_pattern_data() {
        return m_tmp_buf.get_pattern_data();
    }

    LLPatternData get_pattern_data_gpu() {
        return m_tmp_buf.get_pattern_data_gpu();
    }

    LLPatternData get_pattern_data_auto(sycl::queue& q) {
        return m_tmp_buf.get_pattern_data_auto(q);
    }

    ccl_window* window_register(ccl_comm* comm, void* ptr, size_t size) {
        ccl_window* window = new ccl_window(comm);
        int ret = window->register_buf(ptr, size);
        if (ret) {
            windows.push_back(window);
            return window;
        }
        else {
            delete window;
            return nullptr;
        }
    }

    void window_deregister(ccl_window* window) {
        window->deregister_buf();
        windows.erase(std::remove(windows.begin(), windows.end(), window), windows.end());
        delete window;
    }

    void buffer_register(void* buffer, size_t size, void** handle) {
        *handle = NULL;
    }

    void buffer_deregister(void* handle) {}

    std::vector<void*> get_registered_ptrs(const void* ptr, size_t size) {
        std::vector<void*> ptrs;
        size_t offset;
        for (auto window : windows) {
            if (window->is_registered(ptr, size, offset)) {
                ptrs = window->get_ptrs(offset);
                break;
            }
        }
        return ptrs;
    }
#endif // CCL_ENABLE_SYCL

    std::shared_ptr<atl_base_comm> atl_comm;
    std::unique_ptr<ccl_unordered_coll_manager> unordered_coll_manager;

private:
    int m_rank;
    int m_size;
    int m_pof2;

    ccl_double_tree m_dtree;
#ifdef CCL_ENABLE_SYCL
    ccl_comm_barrier_data m_barrier_data;
    ccl_comm_flag_data m_flag_data;
    ccl_tmp_bufs m_tmp_buf;
    ccl_large_tmp_bufs m_large_tmp_buf{};
    ccl_scaleout_host_bufs m_scaleout_host_bufs;
    ccl_scaleout_device_bufs m_scaleout_device_bufs;
    // for sycl pipeline sendrecv
    ccl_scaleout_pipeline_bufs m_scaleout_pipeline_bufs;
    atl_req_t pipeline_send_req, pipeline_recv_req;

    std::vector<group_send_request> group_send_requests;
    std::vector<group_recv_request> group_recv_requests;

    std::vector<ccl_window*> windows;
#endif // CCL_ENABLE_SYCL
};

class alignas(CACHELINE_SIZE) ccl_comm : public ccl::comm_interface {
public:
    static constexpr int invalid_rank = ccl::utils::invalid_rank;
    static constexpr int invalid_size = -1;
    static constexpr int invalid_id = -1;
    static constexpr uint64_t invalid_uniq_id = -1;

    void init(int comm_id,
              const std::shared_ptr<atl_base_comm>& atl_comm,
              bool share_resources = false,
              bool is_sub_communicator = false);
    // common usage: support processes and threads
    ccl_comm(int comm_id,
             std::shared_ptr<atl_base_comm> atl_comm,
             bool share_resources,
             bool is_sub_communicator,
             bool is_Ext = false,
             int size = invalid_size,
             int rank = invalid_rank,
             int group_id = 0);
    ccl_comm(std::shared_ptr<atl_base_comm> atl_comm,
             bool share_resources = false,
             bool is_sub_communicator = false);
    ccl_comm();
    ccl_comm(ccl::ccl_comm_attr_impl& attr);
    // needed for multithreading (single process multiple devices) approach:
    ccl_comm(int size, int rank);

    ccl_comm(ccl_comm& src) = delete;
    ccl_comm(ccl_comm&& src) = default;
    ccl_comm& operator=(ccl_comm& src) = delete;
    ccl_comm& operator=(ccl_comm&& src) = default;
    ~ccl_comm() = default;

    void set_parent_comm(ccl_comm* comm) {
        parent_comm = comm;
    }

    ccl_comm* get_parent_comm() {
        return parent_comm;
    }

    static ccl_comm* create(device_t device,
                            context_t context,
                            int size,
                            int rank,
                            ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                            ccl::ccl_comm_attr_impl& attr);
    static ccl_comm* create(int size,
                            int rank,
                            ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                            ccl::ccl_comm_attr_impl& attr);
    static ccl_comm* create(int size,
                            ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                            ccl::ccl_comm_attr_impl& attr);

    // needed for multithreading (single process multiple devices) approach:
    void initExt(int size,
                 int rank,
                 int comm_id,
                 std::shared_ptr<atl_base_comm> atl_comm,
                 bool share_resources = false,
                 bool is_sub_communicator = false,
                 int group_id = 0);
    static ccl_comm* createExt(device_t device,
                               context_t context,
                               int size,
                               int rank,
                               ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                               ccl::ccl_comm_attr_impl& attr);
    static ccl_comm* createExt(int size,
                               int rank,
                               ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                               ccl::ccl_comm_attr_impl& attr);
    static ccl_comm* createExt(int size,
                               ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                               ccl::ccl_comm_attr_impl& attr);

private:
    // common usage: support processes and threads
    ccl_comm(device_t device,
             context_t context,
             std::shared_ptr<atl_base_comm> atl_comm,
             bool is_Ext = false,
             int size = invalid_size,
             int rank = invalid_rank,
             int group_id = 0);
    ccl_comm(int size,
             int rank,
             ccl::shared_ptr_class<ikvs_wrapper> kvs,
             ccl::ccl_comm_attr_impl& attr);
    ccl_comm(int size, ccl::shared_ptr_class<ikvs_wrapper> kvs, ccl::ccl_comm_attr_impl& attr);

    // copy-constructor with explicit comm_id
    ccl_comm(const ccl_comm& src, int comm_id);

    void create_topo_subcomms(std::shared_ptr<atl_base_comm> atl_comm);
    // needed for multithreading (single process multiple devices) approach:
    void create_topo_subcommsExt(int size, int rank);
    int get_numa_node_for_gpu(std::shared_ptr<ccl_comm> node_comm,
                              std::shared_ptr<ccl::device> device_ptr,
                              std::shared_ptr<ccl::context> context_ptr);

    ccl_comm* get_impl() {
        return this;
    }

    static std::shared_ptr<ikvs_wrapper> get_kvs_wrapper(std::shared_ptr<ccl::kvs_interface> kvs);

    void pre_coll_serialize(const ccl::stream::impl_value_t& stream);
    void post_coll_serialize(const ccl::stream::impl_value_t& stream);

    const ccl::vector_class<ccl::event>& pre_coll_events(const ccl::stream::impl_value_t& stream,
                                                         const ccl::vector_class<ccl::event>& deps,
                                                         ccl::vector_class<ccl::event>& newdeps);
    void post_coll_events(const ccl::stream::impl_value_t& stream, ccl::event& e);
    void post_coll_for_finalize(const ccl::stream::impl_value_t& stream, ccl::event& e);

    const ccl::vector_class<ccl::event>& pre_coll(const ccl::stream::impl_value_t& stream,
                                                  const ccl::vector_class<ccl::event>& deps,
                                                  ccl::vector_class<ccl::event>& newdeps);
    void post_coll(const ccl::stream::impl_value_t& stream, ccl::event& e);


public:
    ccl_comm* create_subcomm(int color, int key = 0) const;
    ccl_comm* create_subcomm_split_independent(int color, int key);
    // needed for multithreading (single process multiple devices) approach:
    ccl_comm* create_subcommExt(int size, int rank) const;
    ccl_comm* create_subcommExt(const std::vector<int>& colors, int rank, int key) const;

    std::shared_ptr<ccl_comm> clone_with_new_id(int comm_id);

    void allocate_resources();

    ccl::comm_interface_ptr split(int color, int key, bool split_external_use = false) override;

    std::string to_string() const;
    std::string to_string_ext() const;

    /**
     * Returns the number of @c rank in the global communicator
     * @param rank a rank which is part of the current communicator
     * @return number of @c rank in the global communicator
     */
    int get_global_rank(int rank) const;

    int get_rank_from_global(int global_rank) const;
    bool try_get_rank_from_global(int global_rank) const;

    /**
     * Returns the number of @c rank in the node communicator
     * @param rank a rank which is part of the current communicator
     * @return number of @c rank in the node communicator
     */
    int get_node_rank(int rank) const;

    ccl_sched_id_t get_sched_id(bool use_internal_space, bool is_pt2pt);

    device_ptr_t get_device() const override {
        return device_ptr;
    }

    context_ptr_t get_context() const override {
        return context_ptr;
    }

    std::shared_ptr<atl_base_comm> get_atl_comm() const {
        if (!comm_impl) {
            LOG_ERROR("get_atl_comm: comm_impl is null, this=", this);
        }
        CCL_THROW_IF_NOT(comm_impl, "comm_impl is null");
        if (!comm_impl->atl_comm) {
            LOG_ERROR("get_atl_comm: comm_impl->atl_comm is null, this=",
                      this,
                      ", comm_impl=",
                      (void*)comm_impl.get());
        }
        CCL_THROW_IF_NOT(comm_impl->atl_comm, "comm_impl->atl_comm is null");
        return comm_impl->atl_comm;
    }

    int get_comm_id() const {
        return comm_impl->atl_comm->get_comm_id();
    }

    // needed for multithreading (single process multiple devices) approach:
    int get_comm_idExt(int idx) const {
        return idx;
    }

    std::shared_ptr<ccl_comm> get_r2r_comm() const {
        if (parent_comm) {
            return parent_comm->get_r2r_comm();
        }
        CCL_ASSERT(r2r_comm, "no r2r_comm");
        return r2r_comm;
    }

    std::shared_ptr<ccl_comm> get_node_comm() const {
        if (parent_comm) {
            return parent_comm->get_node_comm();
        }
        CCL_ASSERT(node_comm, "no node_comm");
        return node_comm;
    }

    std::shared_ptr<ccl_comm> get_numa_comm() const {
        if (parent_comm) {
            return parent_comm->get_numa_comm();
        }
        CCL_ASSERT(numa_comm, "no numa_comm");
        return numa_comm;
    }

    std::shared_ptr<ccl_comm> get_numa_r2r_comm() const {
        if (parent_comm) {
            return parent_comm->get_numa_r2r_comm();
        }
        CCL_ASSERT(numa_r2r_comm, "no numa_r2r_comm");
        return numa_r2r_comm;
    }

    std::shared_ptr<ccl_comm> get_even_comm() const {
        if (parent_comm) {
            return parent_comm->get_even_comm();
        }
        CCL_ASSERT(even_comm, "no even_comm");
        return even_comm;
    }

    std::shared_ptr<ccl_comm> get_pair_comm() const {
        if (parent_comm) {
            return parent_comm->get_pair_comm();
        }
        CCL_ASSERT(pair_comm, "no pair_comm");
        return pair_comm;
    }

    const ccl_rank2rank_map& get_local2global_map() const {
        return local2global_map;
    }

    const ccl::topo_manager& get_topo_manager() const {
        if (parent_comm) {
            return parent_comm->get_topo_manager();
        }
        else {
            return topo_manager;
        }
    }

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    std::shared_ptr<ccl::ze::fd_manager> get_fd_manager() const {
        if (parent_comm) {
            return parent_comm->get_fd_manager();
        }
        else {
            return fd_manager;
        }
    }

    void set_handle_exchange_data(std::shared_ptr<void> handle_exchange_data) {
        this->handle_exchange_data = handle_exchange_data;
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    std::shared_ptr<ccl_comm_env> get_env() const {
        return env;
    }

    std::unique_ptr<ccl_unordered_coll_manager>& get_unordered_coll_manager() const {
        return comm_impl->unordered_coll_manager;
    }

    int rank() const override {
        return comm_rank;
    }

    int size() const override {
        return comm_size;
    }

    int pof2() const noexcept {
        return comm_impl->pof2();
    }

    int id() const noexcept {
        return get_comm_id();
    }

    uint64_t unique_id() const noexcept {
        if (parent_comm) {
            return parent_comm->unique_id();
        }
        return unique_comm_id;
    }

    void update_unique_id(uint64_t id) {
        unique_comm_id = id;
    }

    const ccl_double_tree& dtree() const {
        return comm_impl->dtree();
    }

#ifdef CCL_ENABLE_SYCL
    bool use_remote_atomics() const {
        return comm_impl->use_remote_atomics();
    }

    void set_remote_atomics(bool use_remote_atomics) {
        comm_impl->set_remote_atomics(use_remote_atomics);
    }

    ccl_comm_barrier_data barrier_data() const {
        return comm_impl->barrier_data();
    }

    ccl_comm_barrier_data barrier_inc(const size_t n = 1) {
        return comm_impl->barrier_inc(n);
    }

    void set_barrier_ptrs(std::array<size_t*, MAX_NODE_RANKS> ptrs0,
                          std::array<size_t*, MAX_NODE_RANKS> ptrs1,
                          size_t* count) {
        comm_impl->set_barrier_ptrs(ptrs0, ptrs1, count);
    }

    ccl_comm_flag_data flag_data() const {
        return comm_impl->flag_data();
    }

    void set_flag_ptrs(std::array<size_t*, MAX_NODE_RANKS> ptrs, size_t* count) {
        comm_impl->set_flag_ptrs(ptrs, count);
    }

    void allocate_buffers(sycl::queue& q, bool tmp_in_device, bool umf_enable = false) {
        comm_impl->allocate_buffers(q, tmp_in_device, umf_enable);
    }

    void* get_tmp_buf(int idx) const {
        return comm_impl->get_tmp_buf(idx);
    }

    void* get_tmp_buf_gpu(int idx) const {
        return comm_impl->get_tmp_buf_gpu(idx);
    }

    void set_tmp_buf_size(size_t size) {
        return comm_impl->set_tmp_buf_size(size);
    }

    size_t get_tmp_buf_size() const {
        return comm_impl->get_tmp_buf_size();
    }

    std::pair<void*, std::array<void*, MAX_NODE_RANKS>> get_all_tmp_bufs(bool is_next) {
        return comm_impl->get_all_tmp_bufs(is_next);
    }

    std::pair<void*, std::array<void*, MAX_NODE_RANKS>> get_all_tmp_bufs_gpu(bool is_next) {
        return comm_impl->get_all_tmp_bufs_gpu(is_next);
    }

    int* get_tmp_buf_idx() {
        return comm_impl->get_tmp_buf_idx();
    }

    int* get_tmp_buf_secondary_idx() {
        return comm_impl->get_tmp_buf_secondary_idx();
    }

    void set_remote_tmp_bufs(std::array<void*, MAX_NODE_RANKS> ptrs, int idx) {
        comm_impl->set_remote_tmp_bufs(ptrs, idx);
    }

    void set_remote_tmp_bufs_gpu(std::array<void*, MAX_NODE_RANKS> ptrs, int idx) {
        comm_impl->set_remote_tmp_bufs_gpu(ptrs, idx);
    }

    ccl_large_tmp_bufs& get_large_tmp_bufs() {
        return comm_impl->get_large_tmp_bufs();
    }

    void* get_scaleout_host_buf() {
        return comm_impl->get_scaleout_host_buf();
    }
    void put_scaleout_host_buf(const void* buf) {
        return comm_impl->put_scaleout_host_buf(buf);
    }
    size_t get_scaleout_host_buf_size() {
        return comm_impl->get_scaleout_host_buf_size();
    }

    void* get_scaleout_device_buf(sycl::queue& q) {
        return comm_impl->get_scaleout_device_buf(q);
    }
    void put_scaleout_device_buf(void* buf) {
        return comm_impl->put_scaleout_device_buf(buf);
    }
    size_t get_scaleout_device_buf_size() {
        return comm_impl->get_scaleout_device_buf_size();
    }

    // for sycl pipeline sendrecv
    void** get_scaleout_send_pipeline_bufs(int num_bufs) {
        return comm_impl->get_scaleout_send_pipeline_bufs(num_bufs);
    }

    void** get_scaleout_recv_pipeline_bufs(int num_bufs) {
        return comm_impl->get_scaleout_recv_pipeline_bufs(num_bufs);
    }

    atl_req_t& get_pipeline_send_req() {
        return comm_impl->get_pipeline_send_req();
    }

    atl_req_t& get_pipeline_recv_req() {
        return comm_impl->get_pipeline_recv_req();
    }

    // Group API send/recv request accessors
    std::vector<ccl_internal_comm::group_send_request>& get_group_send_requests() {
        return comm_impl->get_group_send_requests();
    }

    std::vector<ccl_internal_comm::group_recv_request>& get_group_recv_requests() {
        return comm_impl->get_group_recv_requests();
    }

    void clear_group_requests() {
        return comm_impl->clear_group_requests();
    }

    bool is_multi_thread_instance() {
        return enable_multi_thread_instance;
    }

    LLPatternData get_pattern_data() {
        return comm_impl->get_pattern_data();
    }

    ccl_window* window_register(void* ptr, size_t size) {
        return comm_impl->window_register(this, ptr, size);
    }

    void window_deregister(ccl_window* window) {
        comm_impl->window_deregister(window);
    }

    void buffer_register(void* buffer, size_t size, void** handle) {
        comm_impl->buffer_register(buffer, size, handle);
    }

    void buffer_deregister(void* handle) {
        comm_impl->buffer_deregister(handle);
    }

    std::vector<void*> get_registered_ptrs(const void* ptr, size_t size) {
        return comm_impl->get_registered_ptrs(ptr, size);
    }

    LLPatternData get_pattern_data_gpu() {
        return comm_impl->get_pattern_data_gpu();
    }

    LLPatternData get_pattern_data_auto(sycl::queue& q) {
        return comm_impl->get_pattern_data_auto(q);
    }

#endif // CCL_ENABLE_SYCL

    // collectives operation declarations
    ccl::event barrier(const ccl::stream::impl_value_t& stream,
                       const ccl::barrier_attr& attr,
                       const ccl::vector_class<ccl::event>& deps = {}) override;
    ccl::event barrier_impl(const ccl::stream::impl_value_t& stream,
                            const ccl::barrier_attr& attr,
                            const ccl::vector_class<ccl::event>& deps = {});

    void finalize() override;

    COMM_INTERFACE_COLL_METHODS(DEFINITION);
#ifdef CCL_ENABLE_SYCL
    SYCL_COMM_INTERFACE_COLL_METHODS(DEFINITION);
#endif // CCL_ENABLE_SYCL

    COMM_IMPL_DECLARATION;
    COMM_IMPL_CLASS_DECLARATION
    int global_current_id = invalid_id;
    uint64_t unique_comm_id = invalid_uniq_id;

private:
    // this is an internal part of the communicator
    // we store there only the fields which should be shared
    // across ccl_comm copies/clones
    // everything else must go to ccl_comm
    std::shared_ptr<ccl_internal_comm> comm_impl;

    ccl_comm* parent_comm = nullptr;
    // ccl::device/context hasn't got a default c-tor
    // that's why we use shared_ptr<ccl::device/context>
    device_ptr_t device_ptr;
    context_ptr_t context_ptr;

    // TODO: double check if these can be moved to comm_impl as shared fields
    std::shared_ptr<ccl_comm> r2r_comm;
    std::shared_ptr<ccl_comm> node_comm;
    std::shared_ptr<ccl_comm> numa_comm;
    std::shared_ptr<ccl_comm> numa_r2r_comm;
    std::shared_ptr<ccl_comm> even_comm;
    std::shared_ptr<ccl_comm> pair_comm;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    std::shared_ptr<void> handle_exchange_data;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    // these fields are duplicate with the ones in ccl_internal_comm
    // but having them here allows to get them without going
    // through the shared_ptr indirection
    int comm_rank;
    int comm_size;
    bool enable_multi_thread_instance{ false };

    ccl_rank2rank_map local2global_map{};
    ccl::topo_manager topo_manager;
    std::shared_ptr<ccl_comm_env> env;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    std::shared_ptr<ccl::ze::fd_manager> fd_manager;
    void init_ipc_exchange_mode(std::shared_ptr<ccl_comm> comm);
    sycl::event last_event;
    ccl_stream* last_stream = nullptr;
    // Map of streams to their most recent events for finalize tracking
    std::unordered_map<ccl_stream*, sycl::event> finalize_events;
    // Flag to track if comm was used during graph recording (invalid for finalize)
    bool used_in_graph_recording = false;

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    ccl_sched_id_t next_sched_id_internal{};
    ccl_sched_id_t next_sched_id_external{};

}; // class ccl_comm

void coll_init(ccl_comm* comm, ccl_stream* stream);
void coll_initExt(ccl_comm* comm,
                  std::unordered_map<int, std::unordered_map<int, std::vector<void*>>>& hash_table,
                  ccl_stream* global_stream);
