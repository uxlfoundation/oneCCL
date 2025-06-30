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
#include <vector>
#include <sstream>
#include <iostream>

#include "common/global/global.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_ll256.hpp"
#include "allreduce_ring_ll256.hpp"
#endif // CCL_ENABLE_SYCL

using namespace std;
using namespace sycl;

#define SG_SZ (16) /* Arc770: Subgroup Sizes Supported: 8;16;32, while 8 threads per EU */
#define LS_SZ (sizeof(message_t)) /* load/store byte size per work-item */

template <typename T>
static inline message_t _sum(message_t dst, message_t src) {
    using math_t = sycl::vec<T, sizeof(message_t) / sizeof(T)>;
    return sycl::bit_cast<message_t>(sycl::bit_cast<math_t>(dst) + sycl::bit_cast<math_t>(src));
}

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
static inline message_t sum_kernel(message_t &dst, message_t &src, const ccl_datatype &dtype) {
    message_t data;

    switch (dtype.idx()) {
        case ccl::datatype::float16: data = _sum<sycl::half>(dst, src); break;
        case ccl::datatype::bfloat16:
            data = _sum<sycl::_V1::ext::oneapi::bfloat16>(dst, src);
            break;
        case ccl::datatype::float32: data = _sum<float>(dst, src); break;
        case ccl::datatype::int32: data = _sum<int32_t>(dst, src); break;
        case ccl::datatype::uint32: data = _sum<uint32_t>(dst, src); break;
        case ccl::datatype::int64: data = _sum<int64_t>(dst, src); break;
        case ccl::datatype::uint64: data = _sum<uint64_t>(dst, src); break;
#if defined(CCL_SYCL_ENABLE_ARCB) || defined(CCL_SYCL_ENABLE_PVC)
        case ccl::datatype::float64: data = _sum<double>(dst, src); break;
#endif
        default:
            /* following code will hurt performance */
            //sycl::ext::oneapi::experimental::printf("Unknow dtype!\n");
            break;
    }

    return data;
}
#endif

static inline void recv_reduce_send(sycl::sub_group &sg,
                                    char *dst,
                                    char *next,
                                    char *src,
                                    int lid,
                                    const ccl_datatype &dtype,
                                    pattern_t pattern) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    ll256_recv_data(data, src + lid * sz, sg, lid, pattern);

    message_t *dst_buf = (message_t *)dst;
    data = sum_kernel(dst_buf[lid], data, dtype);

    ll256_send_data(data, next + lid * sz, pattern);
#endif
}

static inline void recv_reduce_copy_send(sycl::sub_group &sg,
                                         char *dst,
                                         char *next,
                                         char *src,
                                         int lid,
                                         int req_workitems,
                                         const ccl_datatype &dtype,
                                         pattern_t pattern) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    ll256_recv_data(data, src + lid * sz, sg, lid, pattern);

    message_t *dst_buf = (message_t *)dst;
    data = sum_kernel(dst_buf[lid], data, dtype);

    if (lid < req_workitems) {
        LscStoreUnCached(dst + lid * sz, data);
        //dst_buf[lid] = data;
    }

    ll256_send_data(data, next + lid * sz, pattern);
#endif
}

sycl::event arc_ll256_allreduce(const void *src,
                                void *dst,
                                size_t count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                ccl_comm *comm,
                                ccl_stream *global_stream) {
    sycl::event sycl_e;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    //std::cout << "enter " << __func__ << ", rank: " << world_rank <<  ", count: " << count << std::endl;

    sycl::queue q = global_stream->get_native_stream();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t dt_sz = ccl_dtype.size();
    char *recv_buf = static_cast<char *>(dst);
    char *send_buf = static_cast<char *>(const_cast<void *>(src));

    if (send_buf != recv_buf)
        q.memcpy(recv_buf, send_buf, dt_sz * count);

    /*
     * Intel(R) Arc(TM) A770 Graphics:
     *   Number Of Slices:                       1
     *   Number Of Subslices Per Slice:          32
     *   Number Of EU Per Subslice:              16
     *   Number Of Threads Per EU:               8
     *   Total EU Count:                         512
     *   Physical EU SIMD Width:                 8
     *   GRF size                                32B
     */

    /*
     * B580 Graphics (B-series):
     *   Number Of Slices:                       5
     *   Number Of Subslices Per Slice:          4
     *   Number Of EU Per Subslice:              8
     *   Number Of Threads Per EU:               8
     *   Total EU Count:                         1280
     *   GRF size                                64B
     */

    /* 64-byte load/store granularity to HBM, Maximum 128-byte payload can be used by EU store */
    /* Arc770: Subgroup Sizes Supported: 8;16;32, while 8 threads per EU */
    size_t sg_sz = SG_SZ;

    // query total number of hardware threads
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ssize_t dev_id{ ccl::utils::invalid_device_id };
    if (!ccl::ze::get_device_global_id(ze_dev, &dev_id)) {
        CCL_THROW("unable to get global id for device\n");
    }

    size_t l_sz = 1 * sg_sz;
    int ngroups = ccl::global_data::get().ze_data->devices[dev_id].total_threads;
    size_t g_sz = ngroups * l_sz;
    auto max_threads = ngroups * 16;

    /* To avoid pattern not changed when "iters" is 1 */
    pattern_t pattern_prefix = ++pattern_counter << 16;

    sycl_e = q.submit([&](auto &h) {
        //using namespace sycl::ext::intel::experimental::esimd;

        int local_world_rank = comm_rank;
        int local_world_size = comm_size;

        int next_rank = (local_world_rank + 1) % local_world_size;

        char *local_peer_bufs[ARC_NUM];
#if 0
        auto [local_tmp_buf, remote_ptrs] = node_comm->get_all_tmp_bufs(true);
        for (int i = 0; i < local_world_size; i++) {
            local_peer_bufs[i] = (char *)remote_ptrs[i];
        }
#else
        // use large kernel persistent buffers
        for (int i = 0; i < local_world_size; i++) {
            local_peer_bufs[i] = (char *)get_remote_node_tmp_buf(0, comm)[i];
        }
        //char *local_tmp_buf = local_peer_bufs[local_world_rank];
        char *local_tmp_buf = (char *)get_tmp_buf(0, comm);
#endif

        /*
         * In a single subgroup:
         *   a> 1 dedicated work-item to manage a LS_SZ-byte pattern.
         *   b> other work-items to process data, and each of them handle a LS_SZ-byte data.
         */
        auto default_subgroup_capacity =
            sg_sz * LS_SZ; /* bytes: data and pattern  processed by 1 subgroup */
        auto default_workgroup_capacity =
            l_sz * LS_SZ; /* bytes: data and patterns processed by 1 workgroup */
        //auto default_total_capacity = g_sz * LS_SZ;      /* bytes: data and patterns processed by all workgroups in 1 iteration */

        /* In a single workgroup, the available work-items to process data, excluding work-items for patterns */
        auto workgroup_available_items = l_sz - (l_sz / sg_sz);

        auto subgroup_capacity = LS_SZ * (sg_sz - 1); /* bytes: data processed by 1 subgroup */
        // bytes: data processed by 1 workgroup
        auto workgroup_capacity = LS_SZ * workgroup_available_items;

#if 1
        // calculate how many total threads to dispatch
        size_t bytes_per_rank = (count * dt_sz + local_world_size - 1) / local_world_size;
        ngroups = (bytes_per_rank + subgroup_capacity - 1) / subgroup_capacity;
        // round up
        //ngroups = (ngroups + l_sz - 1) / l_sz * l_sz;
        g_sz = ngroups * l_sz;
        if (g_sz > max_threads) {
            ngroups = max_threads / l_sz;
            g_sz = ngroups * l_sz;
        }
#endif

        auto total_available_items = ngroups * workgroup_available_items;
        // bytes: data processed by all workgroups in 1 iteration
        auto total_capacity = ngroups * workgroup_capacity;

        /* div up */
        int iters = (count * dt_sz + (local_world_size * total_available_items * LS_SZ - 1)) /
                    (local_world_size * total_available_items * LS_SZ);

        //sycl::ext::oneapi::experimental::printf("------> rank: %d, group num: %ld, loop count: %zu\n", local_world_rank, g_sz / l_sz, iters);

        h.parallel_for(
            sycl::nd_range<1>(g_sz, l_sz),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
                int idx = 0;
                size_t offset = 0;
                size_t offset_with_pattern = 0;

                auto group_id = item.get_group_linear_id();
                //auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                auto sg = item.get_sub_group();
                auto sg_id = sg.get_group_id()[0];
                auto sg_lid = sg.get_local_id()[0];

                for (int i = 0; i < iters; i++) {
                    pattern_t pattern = pattern_prefix + i;

                    // base offsets of the current subgroup
                    auto base =
                        local_world_size * (i * total_capacity + group_id * workgroup_capacity +
                                            sg_id * subgroup_capacity);
                    auto base_with_pattern =
                        local_world_size *
                        (/* i * default_total_capacity + */
                         group_id * default_workgroup_capacity + sg_id * default_subgroup_capacity);

                    auto finished = i * total_capacity * local_world_size; /* bytes */
                    auto unreduced = count * dt_sz - finished; /* bytes */

                    // required work-items exclude 1 work-item for pattern
                    auto req_workitems = sg_sz - 1;
                    // LS_SZ bytes per work-item
                    auto chunk_sz = req_workitems * LS_SZ;
                    // aligned to 256B
                    auto chunk_with_pattern = sg_sz * LS_SZ;

                    /* items will be assigned to each rank */
                    auto per_rank_items =
                        (unreduced + (local_world_size * LS_SZ - 1)) / (local_world_size * LS_SZ);
                    auto req_workgroups = (per_rank_items + (workgroup_available_items - 1)) /
                                          workgroup_available_items;
                    auto req_subgroups = 0;

                    if (req_workgroups >= ngroups) {
                        req_workgroups = ngroups;
                    }
                    else {
                        if (group_id == (req_workgroups - 1)) {
                            req_subgroups = (per_rank_items + (sg_sz - 1)) / (sg_sz - 1);

                            /* (req_subgroups % (l_sz/sg_sz) - 1) equals to the final subgroup id in a workgroup */
                            /* Note:  req_subgroups % (l_sz/sg_sz) might be 0 */
                            if (((req_subgroups % (l_sz / sg_sz)) == 0) ||
                                (sg_id == (req_subgroups % (l_sz / sg_sz) - 1))) {
                                if ((per_rank_items % (sg_sz - 1)) != 0) {
                                    /* FIXME: */
                                    req_workitems = per_rank_items % (sg_sz - 1);
                                    // LS_SZ bytes per work-item
                                    chunk_sz = req_workitems * LS_SZ;
                                }
                            }
                        }
                    }

                    if (group_id < req_workgroups) {
                        // right neighbor buffer
                        char *next = local_peer_bufs[next_rank];

                        // step 1: push data to next GPU
                        {
                            offset = base + local_world_rank * chunk_sz;
                            offset_with_pattern =
                                base_with_pattern + local_world_rank * chunk_with_pattern;

                            size_t left_size = count * dt_sz - offset;
                            ll256_send(send_buf + offset + sg_lid * LS_SZ,
                                       next + offset_with_pattern + sg_lid * LS_SZ,
                                       sg_lid * LS_SZ < left_size,
                                       pattern);
                        }

                        // step 2: reduce and copy to next GPU
                        for (int j = 2; j < local_world_size; j++) {
                            idx = (local_world_rank + local_world_size + 1 - j) % local_world_size;
                            offset = base + idx * chunk_sz;
                            offset_with_pattern = base_with_pattern + idx * chunk_with_pattern;

                            recv_reduce_send(sg,
                                             recv_buf + offset,
                                             next + offset_with_pattern,
                                             local_tmp_buf + offset_with_pattern,
                                             sg_lid,
                                             ccl_dtype,
                                             pattern);
                        }

                        // step 3: reduce this buffer and data, which will produce the final
                        // result that we store in this data and push to the next GPU
                        {
                            idx = (local_world_rank + 1) % local_world_size;
                            offset = base + idx * chunk_sz;
                            offset_with_pattern = base_with_pattern + idx * chunk_with_pattern;

                            recv_reduce_copy_send(sg,
                                                  recv_buf + offset,
                                                  next + GATHER_BUF_OFFSET + offset_with_pattern,
                                                  local_tmp_buf + offset_with_pattern,
                                                  sg_lid,
                                                  req_workitems,
                                                  ccl_dtype,
                                                  pattern);
                        }

                        // step 4: copy to next GPU
                        for (int j = 1; j < local_world_size - 1; ++j) {
                            idx = (local_world_rank + local_world_size + 1 - j) % local_world_size;
                            offset = base + idx * chunk_sz;
                            offset_with_pattern =
                                GATHER_BUF_OFFSET + base_with_pattern + idx * chunk_with_pattern;

                            ll256_forward(local_tmp_buf + offset_with_pattern + sg_lid * LS_SZ,
                                          recv_buf + offset + sg_lid * LS_SZ,
                                          next + offset_with_pattern + sg_lid * LS_SZ,
                                          sg,
                                          sg_lid,
                                          req_workitems,
                                          pattern);
                        }

                        // step 5: Make final copy from buffer to dest
                        {
                            idx = (local_world_rank + 2) % local_world_size;
                            offset = base + idx * chunk_sz;
                            offset_with_pattern =
                                GATHER_BUF_OFFSET + base_with_pattern + idx * chunk_with_pattern;

                            ll256_recv(recv_buf + offset + sg_lid * LS_SZ,
                                       local_tmp_buf + offset_with_pattern + sg_lid * LS_SZ,
                                       sg,
                                       sg_lid,
                                       req_workitems,
                                       pattern);
                        }
                    }
                }
            });
    });

    if (reduction == ccl::reduction::avg) {
        std::vector<sycl::event> evs;
        evs.push_back(sycl_e);
        sycl_e = sycl_average(q, dst, count, comm_size, dtype, evs);
    }

    return sycl_e;
}

ccl::event arc_allreduce(const void *src,
                         void *dst,
                         size_t count,
                         ccl::datatype dtype,
                         ccl::reduction reduction,
                         ccl_comm *comm,
                         ccl_stream *global_stream) {
    coll_init(comm, global_stream);

    auto e = arc_ll256_allreduce(src, dst, count, dtype, reduction, comm, global_stream);

    return ccl::event::create_from_native(e);
}
