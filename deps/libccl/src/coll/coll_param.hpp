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

#include <vector>

#include "coll/algorithms/algorithm_utils.hpp"
#include "common/datatype/datatype.hpp"
#include "common/utils/buffer.hpp"
#include "oneapi/ccl.hpp"

class ccl_comm;

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>

template <class native_type>
using ccl_sycl_typed_buffer_t = sycl::buffer<native_type, 1>;

/* ordering should be aligned with ccl::datatype */
using ccl_sycl_buffer_one_dim_types = std::tuple<ccl_sycl_typed_buffer_t<int8_t>,
                                                 ccl_sycl_typed_buffer_t<uint8_t>,
                                                 ccl_sycl_typed_buffer_t<int16_t>,
                                                 ccl_sycl_typed_buffer_t<uint16_t>,
                                                 ccl_sycl_typed_buffer_t<int32_t>,
                                                 ccl_sycl_typed_buffer_t<uint32_t>,
                                                 ccl_sycl_typed_buffer_t<int64_t>,
                                                 ccl_sycl_typed_buffer_t<uint64_t>,
                                                 ccl_sycl_typed_buffer_t<uint16_t>,
                                                 ccl_sycl_typed_buffer_t<float>,
                                                 ccl_sycl_typed_buffer_t<double>,
                                                 ccl_sycl_typed_buffer_t<uint16_t>>;
#endif // CCL_ENABLE_SYCL

#define CCL_INVALID_GROUP_IDX     (-1)
#define CCL_INVALID_PROC_IDX      (-1)
#define CCL_INVALID_PEER_RANK_IDX (-1)
#define CCL_INVALID_ROOT_RANK_IDX (-1)

struct ccl_coll_attr {
    ccl_coll_attr() = default;
    ccl_coll_attr(const ccl_coll_attr&) = default;
    ccl_coll_attr& operator=(const ccl_coll_attr&) = default;

    ccl_coll_attr(const ccl::allgather_attr& attr);
    ccl_coll_attr(const ccl::allgatherv_attr& attr);
    ccl_coll_attr(const ccl::allreduce_attr& attr);
    ccl_coll_attr(const ccl::alltoall_attr& attr);
    ccl_coll_attr(const ccl::alltoallv_attr& attr);
    ccl_coll_attr(const ccl::barrier_attr& attr);
    ccl_coll_attr(const ccl::broadcast_attr& attr);
    ccl_coll_attr(const ccl::pt2pt_attr& attr);
    ccl_coll_attr(const ccl::reduce_attr& attr);
    ccl_coll_attr(const ccl::reduce_scatter_attr& attr);
    ccl_coll_attr(const ccl::scatter_attr& attr);

    ccl_coll_attr(ccl_coll_attr&&) = default;
    ccl_coll_attr& operator=(ccl_coll_attr&&) = default;

    std::string to_string() const;

    ccl::reduction_fn reduction_fn = nullptr;

    size_t priority = 0;
    int synchronous = 0;
    int to_cache = 0;
    std::string match_id{};

    int group_id = CCL_INVALID_GROUP_IDX;

    /* change how user-supplied buffers have to be interpreted */
    int is_vector_buf = 0;

#ifdef CCL_ENABLE_SYCL
    int is_sycl_buf = 0;
#endif // CCL_ENABLE_SYCL
};

struct ccl_coll_param {
    enum class buf_type { regular, device };

    ccl_coll_type ctype = ccl_coll_last_value;
    ccl_coll_algo hint_algo{};

    // for ccl_coll_build_<coll> of build_sched
    ccl_buffer send_buf{};
    ccl_buffer recv_buf{};

    // in case of: ccl_coll_param::create_<coll>_param
    std::vector<void*> send_bufs{};
    std::vector<void*> recv_bufs{};

    // for host transfer in add_scaleout case
    // of topo algos in coll_param.cpp
    std::vector<ccl_buffer> send_scale_out_bufs{};
    std::vector<ccl_buffer> recv_scale_out_bufs{};

    /*
        filled if pre-post copy is used
        to keep original send/recv buffers
        send_buf and recv_buf fields are replaced by staging buffers
    */
    std::vector<void*> send_dev_bufs{};
    std::vector<void*> recv_dev_bufs{};

    std::vector<size_t> send_counts{};
    std::vector<size_t> recv_counts{};
    size_t send_count{};
    size_t count{};

    ccl_datatype dtype = {};
    ccl::reduction reduction = ccl::reduction::sum;
    int root = CCL_INVALID_ROOT_RANK_IDX, peer_rank = CCL_INVALID_PEER_RANK_IDX;

    int group_id = CCL_INVALID_GROUP_IDX;

    ccl_stream* stream = nullptr;
    ccl_comm* comm = nullptr;

    std::vector<ccl::event> deps{};
    bool is_scaleout{ false };
    bool is_validate{ true };
    bool is_hmem_enabled{ false };
    bool is_pt2pt{ false };

    ccl_coll_param(bool in_is_validate = true);
    ccl_coll_param(const ccl_coll_param& other);
    ccl_coll_param& operator=(const ccl_coll_param& other) {
        if (this != &other && is_validate) {
            copy(other);
        }
        return *this;
    }
    // copy-constructor only adds validation,
    // no need for custom destructor
    ~ccl_coll_param() = default;

    std::string to_string() const;

    void* get_send_buf(size_t idx = 0, buf_type type = buf_type::regular) const;
    void* get_recv_buf(size_t idx = 0, buf_type type = buf_type::regular) const;

    void* get_send_buf_ptr(size_t idx = 0, buf_type type = buf_type::regular) const;
    void* get_recv_buf_ptr(size_t idx = 0, buf_type type = buf_type::regular) const;

    size_t get_send_count(size_t idx = 0) const;
    size_t get_recv_count(size_t idx = 0) const;

    bool is_inplace(buf_type type = buf_type::regular) const;

    std::vector<void*> get_all_non_zero_bufs() const;

    void validate() const;

    void copy_deps(const std::vector<ccl::event>& d);
    void set_common_fields(ccl::datatype dtype,
                           ccl_comm* comm,
                           const ccl_stream* stream,
                           const std::vector<ccl::event>& deps);
    static ccl_coll_param create_allgather_param(const void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 const ccl_coll_attr& attr,
                                                 ccl_comm* comm,
                                                 const ccl_stream* stream,
                                                 const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_allgatherv_param(const void* send_buf,
                                                  size_t send_count,
                                                  void* recv_buf,
                                                  const size_t* recv_counts,
                                                  ccl::datatype dtype,
                                                  const ccl_coll_attr& attr,
                                                  ccl_comm* comm,
                                                  const ccl_stream* stream,
                                                  const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_allreduce_param(const void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 ccl::reduction reduction,
                                                 const ccl_coll_attr& attr,
                                                 ccl_comm* comm,
                                                 const ccl_stream* stream,
                                                 const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_alltoall_param(const void* send_buf,
                                                void* recv_buf,
                                                size_t count,
                                                ccl::datatype dtype,
                                                const ccl_coll_attr& attr,
                                                ccl_comm* comm,
                                                const ccl_stream* stream,
                                                const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_alltoallv_param(const void* send_buf,
                                                 const size_t* send_counts,
                                                 void* recv_buf,
                                                 const size_t* recv_counts,
                                                 ccl::datatype dtype,
                                                 const ccl_coll_attr& attr,
                                                 ccl_comm* comm,
                                                 const ccl_stream* stream,
                                                 const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_barrier_param(ccl_comm* comm,
                                               const ccl_stream* stream,
                                               const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_broadcast_param(void* buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 int root,
                                                 const ccl_coll_attr& attr,
                                                 ccl_comm* comm,
                                                 const ccl_stream* stream,
                                                 const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_broadcast_param(void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 int root,
                                                 const ccl_coll_attr& attr,
                                                 ccl_comm* comm,
                                                 const ccl_stream* stream,
                                                 const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_reduce_param(const void* send_buf,
                                              void* recv_buf,
                                              size_t count,
                                              ccl::datatype dtype,
                                              ccl::reduction reduction,
                                              int root,
                                              const ccl_coll_attr& attr,
                                              ccl_comm* comm,
                                              const ccl_stream* stream,
                                              const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_reduce_scatter_param(const void* send_buf,
                                                      void* recv_buf,
                                                      size_t recv_count,
                                                      ccl::datatype dtype,
                                                      ccl::reduction reduction,
                                                      const ccl_coll_attr& attr,
                                                      ccl_comm* comm,
                                                      const ccl_stream* stream,
                                                      const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_scatter_param(const void* send_buf,
                                               void* recv_buf,
                                               size_t count,
                                               ccl::datatype dtype,
                                               int root,
                                               const ccl_coll_attr& attr,
                                               ccl_comm* comm,
                                               const ccl_stream* stream,
                                               const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_recv_param(void* recv_buf,
                                            size_t recv_count,
                                            ccl::datatype dtype,
                                            int peer_rank,
                                            const ccl_coll_attr& attr,
                                            ccl_comm* comm,
                                            const ccl_stream* stream,
                                            const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_send_param(const void* send_buf,
                                            size_t send_count,
                                            ccl::datatype dtype,
                                            int peer_rank,
                                            const ccl_coll_attr& attr,
                                            ccl_comm* comm,
                                            const ccl_stream* stream,
                                            const std::vector<ccl::event>& deps = {});

private:
    void copy(const ccl_coll_param& other);
};
