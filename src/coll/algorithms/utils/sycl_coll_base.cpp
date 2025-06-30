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
#include "coll/coll_util.hpp"
#include "comm/comm.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

// sync_ptrs is used for counting in local kernel_barrier
static ccl_kernel_barrier_data kernel_barrier_data;
static thread_local ccl_kernel_barrier_data thread_kernel_barrier_data;

// three tmp buffers - 1: work_buf, 2: tmp_send_buf, 3: tmp_recv_buf
constexpr int tmp_bufs_count = 3;
// tmp_bufs are used as work buf and to copy input/output
static std::array<void *, tmp_bufs_count> tmp_bufs;

static thread_local std::array<void *, tmp_bufs_count> thread_tmp_bufs;

static thread_local bool is_thread_initial_invocation = true;
size_t tmp_buf_size_per_rank = 0;

std::pair<ccl_sched *, ze_handle_exchange_entry *> do_ipc_exchange(ccl_comm *comm,
                                                                   ccl_stream *stream,
                                                                   std::vector<void *> ptrs) {
    sycl::queue q = stream->get_native_stream();
    bool host_found = false;
    for (auto ptr : ptrs) {
        sycl::usm::alloc alloc_type = sycl::get_pointer_type(ptr, q.get_context());
        if (alloc_type == sycl::usm::alloc::host) {
            host_found = true;
            break;
        }
    }

    int cache_status = -1;
    if (host_found) {
        cache_status = ccl::global_data::env().enable_ze_cache_get_ipc_handles;
        ccl::global_data::env().enable_ze_cache_get_ipc_handles = 0;
        LOG_DEBUG("disabling ZE IPC cache because a host USM buffer was found.");
    }

    ccl_comm *node_comm = comm->get_node_comm().get();
    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers;

    for (auto ptr : ptrs) {
        in_buffers.emplace_back(ptr, ccl::ze::ipc_mem_type::memory);
    }

    ccl_coll_param param{};
    param.comm = comm;
    param.stream = stream;
    ccl_coll_attr attr{};
    ccl_sched *sched = ccl_sched::create(param, attr);
    ccl::utils::pt2pt_handle_exchange_info info = {};
    int skip_rank = ccl_comm::invalid_rank;

    ze_handle_exchange_entry *exchange_entry =
        new ze_handle_exchange_entry(sched, node_comm, in_buffers, skip_rank, info);
    // start the entry
    exchange_entry->start();
    while (!exchange_entry->is_completed()) {
        exchange_entry->update(); //    128us
    }

    if (host_found) {
        // restore
        ccl::global_data::env().enable_ze_cache_get_ipc_handles = cache_status;
    }

    return { sched, exchange_entry };
}

static int get_num_queues(sycl::queue q, int ordinal) {
    sycl::device dev = q.get_device();
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);

    uint32_t queue_group_count = 0;
    ze_result_t result =
        zeDeviceGetCommandQueueGroupProperties(ze_dev, &queue_group_count, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        LOG_WARN("zeDeviceGetCommandQueueGroupProperties failed");
        return 0;
    }

    if (ordinal < queue_group_count) {
        ze_command_queue_group_properties_t *queueProperties =
            (ze_command_queue_group_properties_t *)malloc(
                sizeof(ze_command_queue_group_properties_t) * queue_group_count);
        result =
            zeDeviceGetCommandQueueGroupProperties(ze_dev, &queue_group_count, queueProperties);
        if (result != ZE_RESULT_SUCCESS) {
            LOG_WARN("zeDeviceGetCommandQueueGroupProperties failed");
            return 0;
        }
        int n = queueProperties[ordinal].numQueues;
        free(queueProperties);
        return n;
    }
    else {
        return 0;
    }
}

static sycl::queue create_sycl_queue(sycl::queue &q, int ordinal, int index) {
    // TODO: should we use the parameter q or a new queue?
    sycl::device dev = q.get_device();
    sycl::context ctx = q.get_context();
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    // Create Command Queue
    ze_command_queue_desc_t Qdescriptor = {};
    Qdescriptor.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    Qdescriptor.pNext = NULL;
    Qdescriptor.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    Qdescriptor.ordinal = ordinal;
    Qdescriptor.index = index;
    Qdescriptor.flags = 0;
    Qdescriptor.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    //ze_command_queue_handle_t ze_cmd_queue = nullptr;
    //ze_result_t result = zeCommandQueueCreate(ze_ctx, ze_dev, &Qdescriptor, &ze_cmd_queue);

    ze_command_list_handle_t ze_imm_cmd_list = nullptr;
    ze_result_t result =
        zeCommandListCreateImmediate(ze_ctx, ze_dev, &Qdescriptor, &ze_imm_cmd_list);
    if (result != ZE_RESULT_SUCCESS) {
        LOG_WARN("zeCommandQueueCreate (", ordinal, ",", index, ") failed, returning same queue");
        return q;
    }

#if 0
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::device> InteropDeviceInput{
        ze_dev
    };
    sycl::device InteropDevice =
        sycl::make_device<sycl::backend::ext_oneapi_level_zero>(InteropDeviceInput);
#endif

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> InteropContextInput{
        ze_ctx, std::vector<sycl::device>(1, dev), sycl::ext::oneapi::level_zero::ownership::keep
    };
    sycl::context InteropContext =
        sycl::make_context<sycl::backend::ext_oneapi_level_zero>(InteropContextInput);

    //sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInputCQ{
    //  ze_cmd_queue, InteropDevice, sycl::ext::oneapi::level_zero::ownership::keep};

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInputCL{
        ze_imm_cmd_list, dev, sycl::ext::oneapi::level_zero::ownership::keep
    };

    //return sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(InteropQueueInputCQ, InteropContext);
    return sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(InteropQueueInputCL,
                                                                  InteropContext);
}

// number of link copy engines
static int num_lce;

// main copy engine
static sycl::queue &get_q_me() {
    static sycl::queue q_me;
    return q_me;
}

// link copy engine
static std::vector<sycl::queue> &get_q_le() {
    static std::vector<sycl::queue> q_le;
    return q_le;
}

static void create_copy_engine_queues(sycl::queue q) {
    auto q_me = get_q_me();
    auto q_le = get_q_le();

    num_lce = get_num_queues(q, 2);
    LOG_DEBUG("number of link engines : ", num_lce);

    for (int i = 0; i < num_lce; i++) {
        q_le.push_back(create_sycl_queue(q, 2, i));
    }
    q_me = create_sycl_queue(q, 1, 0);
}

static void comm_barrier(const std::shared_ptr<ccl_comm> comm) {
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        ccl::impl_dispatch disp;
        comm->barrier(disp(ccl::default_stream), ccl::default_barrier_attr).wait();
    }
    else {
        // based on atl invocation from allreduce_scaleout_sycl
        // call ccl::wrapper for MPI/OFI.
        int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
        atl_req_t req;
        std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
        ATL_CALL_THROW_IF_ERROR(atl_comm->barrier(ep_idx, req));

        ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
        if (!req.is_completed) {
            // We do not want to call check() in a loop (because we would call MPI_Test repeatedly). Call MPI_Wait() instead.
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
        }
        else {
            // The operation was probably blocking, since it finished really quickly
        }
    }
}

void coll_init(ccl_comm *comm, ccl_stream *global_stream) {
    static bool is_initial_invocation = true;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    ccl_comm_barrier_data bd = node_comm->barrier_data();

    sycl::queue q = global_stream->get_native_stream();

    // if communicator is used for first time then do ipc exchage
    // to get remote ptrs used for barrier counting and remote tmp bufs
    if (!bd.is_set()) {
        std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
        std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
        std::vector<std::shared_ptr<ccl_comm>> sub_comms{ node_comm, even_comm, pair_comm };
        ccl_large_tmp_bufs &comm_large_tmp_bufs = node_comm->get_large_tmp_bufs();

        sycl::queue q = global_stream->get_native_stream();
        sycl::queue q_worker(q.get_device());

        // alloc sync pointers to be used for global comm_barrier across ranks
        constexpr int num_slots = ccl_comm_barrier_data::slots;
        const size_t ptr_count = num_slots * sub_comms.size();
        size_t *ptrs0;
        if (node_comm->get_topo_manager().has_p2p_access()) {
            ptrs0 = sycl::malloc_device<size_t>(ptr_count, q);
        }
        else {
            ptrs0 = sycl::malloc_host<size_t>(ptr_count, q);
        }
        q.memset(ptrs0, 0, ptr_count * sizeof(size_t)).wait();

        size_t *ptrs1;
        if (node_comm->get_topo_manager().has_p2p_access()) {
            ptrs1 = sycl::malloc_device<size_t>(ptr_count, q);
        }
        else {
            ptrs1 = sycl::malloc_host<size_t>(ptr_count, q);
        }
        q.memset(ptrs1, 0, ptr_count * sizeof(size_t)).wait();

        std::vector<void *> ipc_ptrs{ ptrs0, ptrs0 + num_slots, ptrs0 + 2 * num_slots,
                                      ptrs1, ptrs1 + num_slots, ptrs1 + 2 * num_slots };

        // do one time initializations
        if (is_initial_invocation) {
            is_initial_invocation = false;

            create_copy_engine_queues(q);

            // allocate sync_ptrs for local kernel barrier
            size_t *sync_ptrs;
            if (node_comm->get_topo_manager().has_p2p_access()) {
                sync_ptrs = sycl::malloc_device<size_t>(ccl_kernel_barrier_data::slots, q);
            }
            else {
                sync_ptrs = sycl::malloc_host<size_t>(ccl_kernel_barrier_data::slots, q);
            }

            // Initialize memory to zero using memset
            q.memset(sync_ptrs, 0, ccl_kernel_barrier_data::slots * sizeof(size_t)).wait();

            // Set the sync_ptrs for the kernel barrier data
            get_kernel_barrier_data(comm).set_sync_ptrs(sync_ptrs);

            //set up temp buf to be used for large collectives
            // WA : use smaller tmp buffer for client GPUs
            if (is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device())) &&
                ccl::global_data::env().sycl_tmp_buf_size == 3 * 128 * 1024 * 1024) {
                ccl::global_data::env().sycl_tmp_buf_size = 3 * 32 * 1024 * 1024;
            }
            const size_t tmp_buf_size = ccl::global_data::env().sycl_tmp_buf_size / tmp_bufs_count;
            const size_t tmp_buf_size_per_rank_orig =
                tmp_buf_size / ccl::global_data::get().get_local_proc_count();

            // adjust tmp_buf_size_per_rank to align in all ranks
            const size_t align_bytes = ccl::global_data::env().kernel_mem_align;
            tmp_buf_size_per_rank = (tmp_buf_size_per_rank_orig / align_bytes) * align_bytes;

            char *tmp_buf;
            if (node_comm->get_topo_manager().has_p2p_access()) {
                tmp_buf = sycl::aligned_alloc_device<char>(
                    CCL_REG_MSG_ALIGNMENT, tmp_buf_size * tmp_bufs_count, q);
            }
            else {
                tmp_buf = sycl::aligned_alloc_host<char>(
                    CCL_REG_MSG_ALIGNMENT, tmp_buf_size * tmp_bufs_count, q);
            }

            for (int i = 0; i < tmp_bufs_count; i++) {
                tmp_bufs[i] = tmp_buf + i * tmp_buf_size;
            }
        }

        for (int i = 0; i < tmp_bufs_count; i++) {
            comm_large_tmp_bufs.tmp_bufs[i] = tmp_bufs[i];
        }

        // set up temp buf to be used for small collectives
        const int small_buf_ipc_idx = ipc_ptrs.size();
        char *tmp_buf;
        if (node_comm->get_topo_manager().has_p2p_access()) {
            tmp_buf = sycl::aligned_alloc_device<char>(
                CCL_REG_MSG_ALIGNMENT, ccl_tmp_bufs::buf_size * ccl_tmp_bufs::buf_count, q);
        }
        else {
            tmp_buf = sycl::aligned_alloc_host<char>(
                CCL_REG_MSG_ALIGNMENT, ccl_tmp_bufs::buf_size * ccl_tmp_bufs::buf_count, q);
        }
        for (int i = 0; i < ccl_tmp_bufs::buf_count; i++) {
            void *tmp_buf_ptr = tmp_buf + i * ccl_tmp_bufs::buf_size;
            node_comm->set_tmp_buf(tmp_buf_ptr, i);
            ipc_ptrs.push_back(tmp_buf_ptr);
        }

        // add tmp buf pointers of large buffers
        const int large_buf_ipc_idx = ipc_ptrs.size();
        ipc_ptrs.insert(std::end(ipc_ptrs), std::begin(tmp_bufs), std::end(tmp_bufs));

        const int small_buf_ipc_gpu_idx = ipc_ptrs.size();
        char *tmp_buf_gpu;
        if (node_comm->get_topo_manager().has_p2p_access()) {
            tmp_buf_gpu = sycl::aligned_alloc_device<char>(
                CCL_REG_MSG_ALIGNMENT, ccl_tmp_bufs::buf_size * ccl_tmp_bufs::buf_count, q);
        }
        else {
            // when no p2p, use USM host memory for cross card communication
            tmp_buf_gpu = sycl::aligned_alloc_host<char>(
                CCL_REG_MSG_ALIGNMENT, ccl_tmp_bufs::buf_size * ccl_tmp_bufs::buf_count, q);
        }
        for (int i = 0; i < ccl_tmp_bufs::buf_count; i++) {
            void *tmp_buf_ptr_gpu = tmp_buf_gpu + i * ccl_tmp_bufs::buf_size;
            node_comm->set_tmp_buf_gpu(tmp_buf_ptr_gpu, i);
            ipc_ptrs.push_back(tmp_buf_ptr_gpu);
        }

#ifdef CCL_ENABLE_UMF
        if (ccl::global_data::env().umf_enable) {
            LOG_DEBUG("|UMF|: umf_ipc_exchange");
            umf_ipc_exchange(comm, global_stream, ipc_ptrs);

            // add comm_barrier sync pointers to each communicator
            size_t sub_comms_size = sub_comms.size();
            size_t *counter = sycl::malloc_device<size_t>(sub_comms_size, q);
            q.fill((void *)counter, static_cast<size_t>(num_slots - 1), sub_comms_size).wait();
            for (size_t i = 0, j = sub_comms_size; i < sub_comms.size(); i++, j++) {
                size_t ptrs0_idx = i;
                size_t ptrs1_idx = j;
                auto remote_ptrs0 = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                                         ptrs0_idx,
                                                                         ipc_ptrs[ptrs0_idx],
                                                                         ipc_handle_map,
                                                                         q_worker,
                                                                         q.get_device(),
                                                                         1);
                auto remote_ptrs1 = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                                         ptrs1_idx,
                                                                         ipc_ptrs[ptrs1_idx],
                                                                         ipc_handle_map,
                                                                         q_worker,
                                                                         q.get_device(),
                                                                         1);
                sub_comms[i]->set_barrier_ptrs(remote_ptrs0, remote_ptrs1, counter + i);
            }

            // get ipc pointers for small tmp buffers and add them to node_comm
            for (size_t i = 0, j = small_buf_ipc_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
                auto remote_ptrs = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                    node_comm, j, ipc_ptrs[j], ipc_handle_map, q_worker, q.get_device(), 1);
                node_comm->set_remote_tmp_bufs(remote_ptrs, i);
            }
            for (size_t i = 0, j = small_buf_ipc_gpu_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
                auto remote_ptrs = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                    node_comm, j, ipc_ptrs[j], ipc_handle_map, q_worker, q.get_device(), 1);
                node_comm->set_remote_tmp_bufs_gpu(remote_ptrs, i);
            }

            for (size_t i = 0, j = large_buf_ipc_idx; i < tmp_bufs.size(); i++, j++) {
                comm_large_tmp_bufs.remote_tmp_bufs[i] = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                    node_comm, j, ipc_ptrs[j], ipc_handle_map, q_worker, q.get_device(), 1);
                comm_large_tmp_bufs.remote_even_tmp_bufs[i] = get_ipc_ptrs<void, MAX_GPUS>(
                    even_comm, j, ipc_ptrs[j], ipc_handle_map, q_worker, q.get_device(), 1);
                comm_large_tmp_bufs.remote_pair_tmp_bufs[i] = get_ipc_ptrs<void, MAX_TILES>(
                    pair_comm, j, ipc_ptrs[j], ipc_handle_map, q_worker, q.get_device(), 1);
            }

            q_worker.wait();
        }
        else {
#endif // CCL_ENABLE_UMF

            LOG_DEBUG("|SCHED|: do_ipc_exchange: with sched");
            auto [sched, exchange_entry] = do_ipc_exchange(comm, global_stream, ipc_ptrs);

            // add comm_barrier sync pointers to each communicator
            size_t sub_comms_size = sub_comms.size();
            size_t *counter = sycl::malloc_device<size_t>(sub_comms_size, q);
            q.fill((void *)counter, static_cast<size_t>(num_slots - 1), sub_comms_size).wait();
            for (size_t i = 0, j = sub_comms_size; i < sub_comms.size(); i++, j++) {
                size_t ptrs0_idx = i;
                size_t ptrs1_idx = j;
                auto remote_ptrs0 = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                                         ptrs0_idx,
                                                                         ipc_ptrs[ptrs0_idx],
                                                                         sched,
                                                                         q_worker,
                                                                         1,
                                                                         false /* to_cache */);
                auto remote_ptrs1 = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(sub_comms[i],
                                                                         ptrs1_idx,
                                                                         ipc_ptrs[ptrs1_idx],
                                                                         sched,
                                                                         q_worker,
                                                                         1,
                                                                         false /* to_cache */);
                sub_comms[i]->set_barrier_ptrs(remote_ptrs0, remote_ptrs1, counter + i);
            }
            // get ipc pointers for small tmp buffers and add them to node_comm
            for (size_t i = 0, j = small_buf_ipc_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
                auto remote_ptrs = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                    node_comm, j, ipc_ptrs[j], sched, q_worker, 1, false /* to_cache */);
                node_comm->set_remote_tmp_bufs(remote_ptrs, i);
            }
            // get ipc pointers for small tmp gpu buffers and add them to node_comm
            for (size_t i = 0, j = small_buf_ipc_gpu_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
                auto remote_ptrs = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                    node_comm, j, ipc_ptrs[j], sched, q_worker, 1, false /* to_cache */);
                node_comm->set_remote_tmp_bufs_gpu(remote_ptrs, i);
            }
            // get ipc pointers for large tmp_buffers
            for (size_t i = 0, j = large_buf_ipc_idx; i < tmp_bufs.size(); i++, j++) {
                comm_large_tmp_bufs.remote_tmp_bufs[i] = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                    node_comm, j, ipc_ptrs[j], sched, q_worker, 1, false /* to_cache */);
                comm_large_tmp_bufs.remote_even_tmp_bufs[i] = get_ipc_ptrs<void, MAX_GPUS>(
                    even_comm, j, ipc_ptrs[j], sched, q_worker, 1, false /* to_cache */);
                comm_large_tmp_bufs.remote_pair_tmp_bufs[i] = get_ipc_ptrs<void, MAX_TILES>(
                    pair_comm, j, ipc_ptrs[j], sched, q_worker, 1, false /* to_cache */);
            }

            q_worker.wait();

            delete exchange_entry;
            delete sched;

#ifdef CCL_ENABLE_UMF
        }
#endif // CCL_ENABLE_UMF

        auto evt = q.submit([=](sycl::handler &h) {
            h.host_task([node_comm]() {
                comm_barrier(node_comm);
            });
        });
        evt.wait();
    }
}

void coll_initExt(ccl_comm *comm,
                  std::unordered_map<int, std::unordered_map<int, std::vector<void *>>> &hash_table,
                  ccl_stream *global_stream) {
    // TODO: check all pthread_barrier_wait invokes
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    ccl_comm_barrier_data bd = node_comm->barrier_data();
    pthread_barrier_wait(
        &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);

    // if communicator is used for the first time then do ipc exchange
    // to get remote pointers used for barrier counting and remote tmp bufs
    if (!bd.is_set()) {
        std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
        std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
        std::vector<std::shared_ptr<ccl_comm>> sub_comms{ node_comm, even_comm, pair_comm };
        ccl_large_tmp_bufs &comm_large_tmp_bufs = node_comm->get_large_tmp_bufs();

        sycl::queue q = global_stream->get_native_stream();
        sycl::queue q_worker(q.get_device());

        // alloc sync pointers to be used for global comm_barrier across ranks
        constexpr int num_slots = ccl_comm_barrier_data::slots;
        const size_t ptr_count = num_slots * sub_comms.size();
        size_t *ptrs0 = sycl::malloc_device<size_t>(ptr_count, q);
        q.memset(ptrs0, 0, ptr_count * sizeof(size_t)).wait();

        size_t *ptrs1 = sycl::malloc_device<size_t>(ptr_count, q);
        q.memset(ptrs1, 0, ptr_count * sizeof(size_t)).wait();

        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);

        std::vector<void *> ipc_ptrs{ ptrs0, ptrs0 + num_slots, ptrs0 + 2 * num_slots,
                                      ptrs1, ptrs1 + num_slots, ptrs1 + 2 * num_slots

        };
        // do one-time initializations
        if (is_thread_initial_invocation) {
            is_thread_initial_invocation = false;

            // allocate sync_ptrs for local kernel barrier
            size_t *sync_ptrs = sycl::malloc_device<size_t>(ccl_kernel_barrier_data::slots, q);

            // Initialize memory to zero using memset
            q.memset(sync_ptrs, 0, ccl_kernel_barrier_data::slots * sizeof(size_t)).wait();

            // Set the sync_ptrs for the kernel barrier data
            get_kernel_barrier_data(comm).set_sync_ptrs(sync_ptrs);

            //set up temp buf to be used for large collectives
            // WA : use smaller tmp buffer for client GPUs
            if (is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device())) &&
                ccl::global_data::env().sycl_tmp_buf_size == 3 * 128 * 1024 * 1024) {
                ccl::global_data::env().sycl_tmp_buf_size = 3 * 32 * 1024 * 1024;
            }
            const size_t tmp_buf_size = ccl::global_data::env().sycl_tmp_buf_size / tmp_bufs_count;
            const size_t tmp_buf_size_per_rank_orig =
                tmp_buf_size / comm->size(); //ccl::global_data::get().get_local_proc_count();
            // adjust tmp_buf_size_per_rank to align in all ranks
            const size_t align_bytes = ccl::global_data::env().kernel_mem_align;
            tmp_buf_size_per_rank = (tmp_buf_size_per_rank_orig / align_bytes) * align_bytes;

            char *tmp_buf = sycl::aligned_alloc_device<char>(
                CCL_REG_MSG_ALIGNMENT, tmp_buf_size * tmp_bufs_count, q);

            for (int i = 0; i < tmp_bufs_count; i++) {
                thread_tmp_bufs[i] = tmp_buf + i * tmp_buf_size;
            }
        }
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);

        for (int i = 0; i < tmp_bufs_count; i++) {
            comm_large_tmp_bufs.tmp_bufs[i] = thread_tmp_bufs[i];
        }

        // set up temp buf to be used for small collectives
        const int small_buf_ipc_idx = ipc_ptrs.size();
        char *tmp_buf = sycl::aligned_alloc_device<char>(
            CCL_REG_MSG_ALIGNMENT, ccl_tmp_bufs::buf_size * ccl_tmp_bufs::buf_count, q);
        for (int i = 0; i < ccl_tmp_bufs::buf_count; i++) {
            void *tmp_buf_ptr = tmp_buf + i * ccl_tmp_bufs::buf_size;
            node_comm->set_tmp_buf(tmp_buf_ptr, i);
            ipc_ptrs.push_back(tmp_buf_ptr);
        }
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
        // add tmp buf pointers of large buffers
        const int large_buf_ipc_idx = ipc_ptrs.size();

        ipc_ptrs.insert(std::end(ipc_ptrs),
                        std::begin(comm_large_tmp_bufs.tmp_bufs),
                        std::end(comm_large_tmp_bufs.tmp_bufs));
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
        const int small_buf_ipc_gpu_idx = ipc_ptrs.size();
        char *tmp_buf_gpu = sycl::aligned_alloc_device<char>(
            CCL_REG_MSG_ALIGNMENT, ccl_tmp_bufs::buf_size * ccl_tmp_bufs::buf_count, q);
        for (int i = 0; i < ccl_tmp_bufs::buf_count; i++) {
            void *tmp_buf_ptr_gpu = tmp_buf_gpu + i * ccl_tmp_bufs::buf_size;
            node_comm->set_tmp_buf_gpu(tmp_buf_ptr_gpu, i);
            ipc_ptrs.push_back(tmp_buf_ptr_gpu);
        }
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);

        ccl::global_data::get().shared_data->do_ipc_exchangeExt(
            comm, hash_table, global_stream, ipc_ptrs);

        // add comm_barrier sync pointers to each communicator
        size_t sub_comms_size = sub_comms.size();
        size_t *counter = sycl::malloc_device<size_t>(sub_comms_size, q);
        q.submit([=](sycl::handler &h) {
             h.single_task([=]() {
                 for (int i = 0; i < sub_comms_size; i++) {
                     counter[i] = num_slots - 1;
                 }
             });
         }).wait();
        for (size_t i = 0, j = sub_comms_size; i < sub_comms.size(); i++, j++) {
            size_t ptrs0_idx = i;
            size_t ptrs1_idx = j;
            auto remote_ptrs0 =
                ccl::global_data::get().shared_data->get_ipc_ptrsExt<size_t, MAX_NODE_RANKS>(
                    sub_comms[i],
                    hash_table,
                    i,
                    ptrs0_idx,
                    ipc_ptrs[ptrs0_idx],
                    0,
                    0,
                    even_comm,
                    pair_comm);
            auto remote_ptrs1 =
                ccl::global_data::get().shared_data->get_ipc_ptrsExt<size_t, MAX_NODE_RANKS>(
                    sub_comms[i],
                    hash_table,
                    i,
                    ptrs1_idx,
                    ipc_ptrs[ptrs1_idx],
                    0,
                    0,
                    even_comm,
                    pair_comm);
            sub_comms[i]->set_barrier_ptrs(remote_ptrs0, remote_ptrs1, counter + i);
        }
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
        // get ipc pointers for small tmp buffers and add them to node_comm
        for (size_t i = 0, j = small_buf_ipc_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
            auto remote_ptrs =
                ccl::global_data::get().shared_data->get_ipc_ptrsExt<void, MAX_NODE_RANKS>(
                    node_comm, hash_table, 0 /*node*/, j, ipc_ptrs[j], 0, 0, even_comm, pair_comm);
            node_comm->set_remote_tmp_bufs(remote_ptrs, i);
        }
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
        // get ipc pointers for small tmp gpu buffers and add them to node_comm
        for (size_t i = 0, j = small_buf_ipc_gpu_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
            auto remote_ptrs =
                ccl::global_data::get().shared_data->get_ipc_ptrsExt<void, MAX_NODE_RANKS>(
                    node_comm, hash_table, 0 /*node*/, j, ipc_ptrs[j], 0, 0, even_comm, pair_comm);
            node_comm->set_remote_tmp_bufs_gpu(remote_ptrs, i); // TODO this crashes
        }

        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
        // get ipc pointers for large tmp_buffers
        for (size_t i = 0, j = large_buf_ipc_idx; i < tmp_bufs.size(); i++, j++) {
            comm_large_tmp_bufs.remote_tmp_bufs[i] =
                ccl::global_data::get().shared_data->get_ipc_ptrsExt<void, MAX_NODE_RANKS>(
                    node_comm, hash_table, 0 /*node*/, j, ipc_ptrs[j], 0, 0, even_comm, pair_comm);
            comm_large_tmp_bufs.remote_even_tmp_bufs[i] =
                ccl::global_data::get().shared_data->get_ipc_ptrsExt<void, MAX_GPUS>(
                    even_comm, hash_table, 1 /*even*/, j, ipc_ptrs[j], 0, 0, even_comm, pair_comm);
            comm_large_tmp_bufs.remote_pair_tmp_bufs[i] =
                ccl::global_data::get().shared_data->get_ipc_ptrsExt<void, 2>(
                    pair_comm, hash_table, 2 /*pair*/, j, ipc_ptrs[j], 0, 0, even_comm, pair_comm);
        }
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
        q_worker.wait();
    }
    pthread_barrier_wait(
        &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
}

ccl_kernel_barrier_data &get_kernel_barrier_data(ccl_comm *comm) {
    return comm->is_multi_thread_instance() ? thread_kernel_barrier_data : kernel_barrier_data;
}

void *get_tmp_buf(int index, ccl_comm *comm) {
    ccl_comm *node_comm = comm->get_node_comm().get();
    ccl_large_tmp_bufs &comm_large_tmp_bufs = node_comm->get_large_tmp_bufs();
    return comm_large_tmp_bufs.tmp_bufs[index];
}

std::array<void *, MAX_NODE_RANKS> get_remote_node_tmp_buf(int index, ccl_comm *comm) {
    ccl_comm *node_comm = comm->get_node_comm().get();
    ccl_large_tmp_bufs &comm_large_tmp_bufs = node_comm->get_large_tmp_bufs();
    return comm_large_tmp_bufs.remote_tmp_bufs[index];
}

std::array<void *, MAX_GPUS> get_remote_even_tmp_buf(int index, ccl_comm *comm) {
    ccl_comm *node_comm = comm->get_node_comm().get();
    ccl_large_tmp_bufs &comm_large_tmp_bufs = node_comm->get_large_tmp_bufs();
    return comm_large_tmp_bufs.remote_even_tmp_bufs[index];
}

std::array<void *, MAX_TILES> get_remote_pair_tmp_buf(int index, ccl_comm *comm) {
    ccl_comm *node_comm = comm->get_node_comm().get();
    ccl_large_tmp_bufs &comm_large_tmp_bufs = node_comm->get_large_tmp_bufs();
    return comm_large_tmp_bufs.remote_pair_tmp_bufs[index];
}

size_t get_tmp_buf_size_per_rank() {
    return tmp_buf_size_per_rank;
}

std::vector<sycl::event> get_sycl_events(const ccl::vector_class<ccl::event> &deps) {
    std::vector<sycl::event> ret;
    if (!group_impl::is_group_active) {
        for (auto &dep : deps) {
            ret.push_back(dep.get_native());
        }
    }
    return ret;
}

// invoke the global communication barrier kernel
sycl::event invoke_barrier(const std::shared_ptr<ccl_comm> comm,
                           sycl::queue q,
                           const std::vector<sycl::event> &dep_events,
                           bool use_cpu) {
    bool gpu_increment = use_recording_path(q);
    sycl::event e;
    if (use_cpu) {
        e = q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.host_task([comm]() {
                comm_barrier(comm);
            });
        });
    }
    else {
        ccl_comm_barrier_data barrier_data =
            gpu_increment ? comm->barrier_data() : comm->barrier_inc();
        e = q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.parallel_for(
                sycl::nd_range<1>(MAX_NODE_RANKS, MAX_NODE_RANKS),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    comm_barrier(barrier_data, it, true, gpu_increment);
                });
        });
    }
    return e;
}

int get_num_lce() {
    return num_lce;
}

// get main copy engine
sycl::queue get_mce_queue(sycl::queue q) {
    auto q_me = get_q_me();

    return q_me;
}

// get link copy engine
sycl::queue get_lce_queue(sycl::queue q, int index) {
    auto q_le = get_q_le();

    return q_le[index];
}

sycl::queue &get_default_queue() {
    static sycl::queue default_queue;
    return default_queue;
}

void copy_data(const int dsize,
               const int N,
               std::array<void *, MAX_GPUS> dst,
               std::array<void *, MAX_GPUS> src,
               const size_t count,
               sycl::queue q,
               std::vector<sycl::event> deps,
               std::vector<sycl::event> &out) {
    for (int i = 0; i < N; i++) {
        sycl::event e = q.submit([=](sycl::handler &h) {
            h.depends_on(deps);
            h.memcpy(dst[i], src[i], dsize * count);
        });
        out.push_back(e);
    }
}

sycl::event sycl_average(sycl::queue &q,
                         void *reduce_buf,
                         const size_t reduce_count,
                         const size_t total_ranks,
                         ccl::datatype dtype,
                         std::vector<sycl::event> &dep_events) {
    auto lambda = [&]<typename T>() {
        const size_t dsize = ccl::global_data::get().dtypes->get(dtype).size();
        bool use_full_vector = can_use_full_vector(reduce_buf, reduce_buf, reduce_count, dsize);
        if (use_full_vector) {
            constexpr int vec_size = get_num_elements<T, 8, true>();
            return reduce_average_invoke<T, vec_size, 16>(
                q, reduce_buf, reduce_count, total_ranks, dep_events);
        }
        else {
            constexpr int vec_size = get_num_elements<T, 8, false>();
            return reduce_average_invoke<T, vec_size, 64>(
                q, reduce_buf, reduce_count, total_ranks, dep_events);
        }
    };
    return invoke_scaleout(lambda, dtype);
}

sycl::event pt2pt_pre_sync(sycl::queue &q,
                           const std::vector<sycl::event> &deps,
                           ccl_comm *comm,
                           bool do_send,
                           int peer_rank,
                           uint64_t tag) {
    auto init_fn = [=](atl_req_t &req) -> bool {
        int ep_idx = 0;
        char data[1] = { 1 };
        size_t n = sizeof(data);
        auto atl = comm->get_atl_comm();

        if (do_send) {
            ATL_CALL_THROW_IF_ERROR(atl->send(ep_idx, data, n, peer_rank, tag, req));
            LOG_DEBUG("pt2pt pre-sync SEND init, tag=", tag);
        }
        else {
            LOG_DEBUG("pt2pt pre-sync RECV init, tag=", tag);
            ATL_CALL_THROW_IF_ERROR(atl->recv(ep_idx, data, n, peer_rank, tag, req));
            if (group_impl::is_group_active) {
                ATL_CALL_THROW_IF_ERROR(atl->wait(ep_idx, req));
            }
        }

        if (!group_impl::is_group_active) {
            ATL_CALL_THROW_IF_ERROR(atl->check(ep_idx, req));
            if (!req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl->wait(ep_idx, req));
            }
        }
        return true;
    };

    // wait_fn: performs the blocking wait
    auto wait_fn = [=](atl_req_t &req) {
        if (req.is_completed) {
            return;
        }
        int ep_idx = 0;
        auto atl = comm->get_atl_comm();
        ATL_CALL_THROW_IF_ERROR(atl->wait(ep_idx, req));
        LOG_DEBUG("pt2pt pre-sync wait done, tag=", tag);
    };

    if (group_impl::is_group_active) {
        return q.submit([=](sycl::handler &h) {
            h.depends_on(deps);
            h.host_task([=]() {
                atl_req_t req{};
                init_fn(req);
            });
        });
    }

    atl_req_t req{};
    sycl::event e = q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        h.host_task([=, &req]() {
            init_fn(req);
        });
    });

    // Wait for the host_task to post the send/recv
    e.wait();
    // then do the final blocking wait here
    wait_fn(req);
    return e;
}

sycl::event post_host_task_ack(sycl::queue &q,
                               const std::vector<sycl::event> &deps,
                               ccl_comm *comm,
                               bool do_send,
                               int peer_rank,
                               uint64_t ack_tag) {
    auto ack_driver = [=](atl_req_t &req, bool init) -> bool {
        int ep_idx = 0;
        char data[1] = { 1 };
        size_t n = sizeof(data);
        auto atl = comm->get_atl_comm();
        if (req.is_completed) {
            return true;
        }
        if (init) {
            if (do_send)
                ATL_CALL_THROW_IF_ERROR(atl->send(ep_idx, data, n, peer_rank, ack_tag, req));
            else
                ATL_CALL_THROW_IF_ERROR(atl->recv(ep_idx, data, n, peer_rank, ack_tag, req));
        }

        ATL_CALL_THROW_IF_ERROR(atl->check(ep_idx, req));
        if (!req.is_completed) {
            ATL_CALL_THROW_IF_ERROR(atl->wait(ep_idx, req));
        }

        LOG_DEBUG("post_host_task_ack: ", (do_send ? "send" : "recv"), ", tag=", ack_tag);
        return true;
    };
    if (group_impl::is_group_active) {
        group_impl::add_post_processing_step(ack_driver);
        group_impl::set_sycl_queue(q);
    }

    sycl::event e;
    if (!group_impl::is_group_active) {
        atl_req_t req{};
        e = q.submit([&](sycl::handler &h) {
            h.depends_on(deps);
            h.host_task([=, &req]() {
                ack_driver(req, /*init=*/true);
            });
        });
        e.wait();
        ack_driver(req, /*init=*/false);
    }
    else {
        e = submit_wait_on_events(q, deps);
        e.wait();
    }
    return e;
}
