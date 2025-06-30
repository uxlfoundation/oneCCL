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
#include <atomic>
#include <unordered_map>

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
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/global/ze/ze_fd_manager.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
#include "types_generator_defines.hpp"
#include "topology/topo_manager.hpp"
#include "unordered_coll/unordered_coll.hpp"

// index = local_rank, value = global_rank
using ccl_rank2rank_map = std::vector<int>;

class ikvs_wrapper;

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
constexpr size_t MAX_TILES = ccl::topo_manager::max_ranks_per_card;
constexpr size_t MAX_GPUS = ccl::topo_manager::max_ranks_per_plane;
constexpr size_t MAX_NODE_RANKS =
    ccl::topo_manager::max_ranks_per_card * ccl::topo_manager::max_ranks_per_plane;
class alignas(CACHELINE_SIZE) ccl_comm_barrier_data {
private:
    int m_rank;
    int m_size;
    size_t m_count = slots - 1;
    size_t* d_count = nullptr;
    bool m_is_set = false;
    std::array<size_t*, MAX_NODE_RANKS> m_remote_ptrs{};
    std::array<size_t*, MAX_NODE_RANKS> d_remote_ptrs{};

public:
    static constexpr int slots = 3;

    ccl_comm_barrier_data(int rank, int size) : m_rank(rank), m_size(size) {}

    int rank() const {
        return m_rank;
    }

    int size() const {
        return m_size;
    }

    bool is_set() const {
        return m_is_set;
    }

    size_t inc(size_t n) {
        m_count += n;
        return m_count;
    }

    size_t count() const {
        return m_count / slots;
    }

    int slot() const {
        return m_count % slots;
    }

    std::array<size_t*, MAX_NODE_RANKS> remote_ptrs() const {
        return m_remote_ptrs;
    }

    size_t inc_gpu(size_t n) {
        *d_count = *d_count + n;
        return *d_count;
    }

    size_t count_gpu() const {
        return *d_count / slots;
    }

    int slot_gpu() const {
        return *d_count % slots;
    }

    std::array<size_t*, MAX_NODE_RANKS> remote_ptrs_gpu() const {
        return d_remote_ptrs;
    }

    void set_count_gpu(size_t* count) {
        d_count = count;
    }

    void set_remote_ptrs(std::array<size_t*, MAX_NODE_RANKS> ptrs,
                         std::array<size_t*, MAX_NODE_RANKS> gpu_barrier_ptrs) {
        assert(!is_set());
        m_is_set = true;
        m_remote_ptrs = ptrs;
        d_remote_ptrs = gpu_barrier_ptrs;
    }
};

struct alignas(CACHELINE_SIZE) ccl_large_tmp_bufs {
    // three tmp buffers - 1: work_buf, 2: tmp_send_buf, 3: tmp_recv_buf
    static constexpr int buf_count = 3;
    // tmp_bufs are used as work buf and to copy input/output
    std::array<void*, buf_count> tmp_bufs;
    // ipc exchanged pointers to remote tmp buffers
    std::array<void*, MAX_NODE_RANKS> remote_tmp_bufs[buf_count] = {};
    std::array<void*, MAX_GPUS> remote_even_tmp_bufs[buf_count] = {};
    std::array<void*, MAX_TILES> remote_pair_tmp_bufs[buf_count] = {};

    ccl_large_tmp_bufs() {
        // Explicitly zero-initialize all pointers
        tmp_bufs.fill(nullptr);
        for (int i = 0; i < buf_count; i++) {
            remote_tmp_bufs[i].fill(nullptr);
            remote_even_tmp_bufs[i].fill(nullptr);
            remote_pair_tmp_bufs[i].fill(nullptr);
        }
    }
};

class alignas(CACHELINE_SIZE) ccl_tmp_bufs {
public:
    // to avoid data race towards the end of a collective and starting of
    // next collective we use different buffers on consecutive collectives.
    static constexpr int buf_count = 2;
    // use largest threshold among all the small buffers algorithms
    static constexpr size_t buf_size = 2097152;

    void set_tmp_buf(void* ptr, int idx) {
        tmp_bufs[idx] = ptr;
    }

    void set_tmp_buf_gpu(void* ptr, int idx) {
        tmp_bufs_gpu[idx] = ptr;
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

private:
    void* tmp_bufs[buf_count];
    std::array<void*, MAX_NODE_RANKS> remote_tmp_bufs[buf_count];
    void* tmp_bufs_gpu[buf_count];
    std::array<void*, MAX_NODE_RANKS> remote_tmp_bufs_gpu[buf_count];

    int index = 0;
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

    std::pair<void*, std::array<void*, MAX_NODE_RANKS>> get_all_tmp_bufs(bool is_next) {
        return m_tmp_buf.get_all_tmp_bufs(is_next);
    }

    std::pair<void*, std::array<void*, MAX_NODE_RANKS>> get_all_tmp_bufs_gpu(bool is_next) {
        return m_tmp_buf.get_all_tmp_bufs_gpu(is_next);
    }

    void set_tmp_buf(void* ptr, int idx) {
        m_tmp_buf.set_tmp_buf(ptr, idx);
    }

    void set_tmp_buf_gpu(void* ptr, int idx) {
        m_tmp_buf.set_tmp_buf_gpu(ptr, idx);
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
    ccl_tmp_bufs m_tmp_buf;
    ccl_large_tmp_bufs m_large_tmp_buf{};
    ccl_scaleout_host_bufs m_scaleout_host_bufs;
    ccl_scaleout_device_bufs m_scaleout_device_bufs;
    // for sycl pipeline sendrecv
    ccl_scaleout_pipeline_bufs m_scaleout_pipeline_bufs;
    atl_req_t pipeline_send_req, pipeline_recv_req;
#endif // CCL_ENABLE_SYCL
};

class alignas(CACHELINE_SIZE) ccl_comm : public ccl::comm_interface {
public:
    static constexpr int invalid_rank = -1;
    static constexpr int invalid_size = -1;
    static constexpr int invalid_id = -1;

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

    ccl_comm* get_impl() {
        return this;
    }

    static std::shared_ptr<ikvs_wrapper> get_kvs_wrapper(std::shared_ptr<ccl::kvs_interface> kvs);

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

    const ccl_double_tree& dtree() const {
        return comm_impl->dtree();
    }

#ifdef CCL_ENABLE_SYCL
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

    std::pair<void*, std::array<void*, MAX_NODE_RANKS>> get_all_tmp_bufs(bool is_next) {
        return comm_impl->get_all_tmp_bufs(is_next);
    }

    std::pair<void*, std::array<void*, MAX_NODE_RANKS>> get_all_tmp_bufs_gpu(bool is_next) {
        return comm_impl->get_all_tmp_bufs_gpu(is_next);
    }

    void set_tmp_buf(void* ptr, int idx) {
        comm_impl->set_tmp_buf(ptr, idx);
    }

    void set_remote_tmp_bufs(std::array<void*, MAX_NODE_RANKS> ptrs, int idx) {
        comm_impl->set_remote_tmp_bufs(ptrs, idx);
    }

    void set_tmp_buf_gpu(void* ptr, int idx) {
        comm_impl->set_tmp_buf_gpu(ptr, idx);
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
    bool is_multi_thread_instance() {
        return enable_multi_thread_instance;
    }

#endif // CCL_ENABLE_SYCL

    // collectives operation declarations
    ccl::event barrier(const ccl::stream::impl_value_t& stream,
                       const ccl::barrier_attr& attr,
                       const ccl::vector_class<ccl::event>& deps = {}) override;
    ccl::event barrier_impl(const ccl::stream::impl_value_t& stream,
                            const ccl::barrier_attr& attr,
                            const ccl::vector_class<ccl::event>& deps = {});

    COMM_INTERFACE_COLL_METHODS(DEFINITION);
#ifdef CCL_ENABLE_SYCL
    SYCL_COMM_INTERFACE_COLL_METHODS(DEFINITION);
#endif // CCL_ENABLE_SYCL

    COMM_IMPL_DECLARATION;
    COMM_IMPL_CLASS_DECLARATION
    int global_current_id = invalid_id;

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
    std::shared_ptr<ccl_comm> even_comm;
    std::shared_ptr<ccl_comm> pair_comm;

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
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    ccl_sched_id_t next_sched_id_internal{};
    ccl_sched_id_t next_sched_id_external{};

}; // class ccl_comm
