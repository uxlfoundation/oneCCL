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

#include "common/global/global.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/reduce/sycl/reduce_pcie.hpp"
#endif // CCL_ENABLE_SYCL

template <template <typename, template <typename, int> class, bool, int> class Transmit>
ccl::event reduce_ll(const void *src,
                        void *dst,
                        size_t count,
                        ccl::datatype dtype,
                        int root,
                        ccl::reduction reduction,
                        ccl_comm *comm,
                        ccl_stream *global_stream,
                        bool &done) {
    sycl::event sycl_e;
    sycl::queue q = global_stream->get_native_stream();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    bool recording = use_recording_path(q);
    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t dt_sz = ccl_dtype.size();

    bool p2p = node_comm->get_topo_manager().has_p2p_access();

    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }

    auto lambda = [&]<typename T, template <typename, int> class Proto>(int NRanks) {
        T *peerbuf0[NRanks];
        T *peerbuf1[NRanks];
        T *ipcbuf0;
        T *ipcbuf1;
        size_t ipcbuf_size = 0;
        if (ccl::global_data::env().sycl_ll_buffer_global) {
            // large buffer
            for (int i = 0; i < NRanks; i++) {
                peerbuf0[i] = (T *)get_remote_large_tmp_buf(0, node_comm.get())[i];
                peerbuf1[i] = (T *)get_remote_large_tmp_buf(1, node_comm.get())[i];
            }
            ipcbuf0 = (T *)get_large_tmp_buf(0, comm);
            ipcbuf1 = (T *)get_large_tmp_buf(1, comm);
            ipcbuf_size = ccl::global_data::env().sycl_tmp_buf_size / ccl_large_tmp_bufs::buf_count;
        }
        else {
            // small buffer
            auto [local_tmp_buf, remote_ptrs] = recording ? node_comm->get_all_tmp_bufs_gpu(true)
                                                          : node_comm->get_all_tmp_bufs(true);
            ipcbuf_size = node_comm->get_tmp_buf_size() / 2;
            for (int i = 0; i < NRanks; i++) {
                peerbuf0[i] = (T *)remote_ptrs[i];
                peerbuf1[i] = (T *)((char *)remote_ptrs[i] + ipcbuf_size);
            }
            ipcbuf0 = (T *)local_tmp_buf;
            ipcbuf1 = (T *)((char *)local_tmp_buf + ipcbuf_size);
        }

        sycl::event e;

        if (recording) {
            e = Reduce<T, Proto, Transmit, true>::launch(
                comm->get_node_comm()->get_pattern_data_gpu(),
                NRanks,
                (T *)src,
                (T *)dst,
                ipcbuf0,
                ipcbuf1,
                // TODO it can also be a two-shot transmit !
                ipcbuf_size,
                RingTransmit<int, Proto, true>::ringSize / sizeof(T),
                peerbuf0,
                peerbuf1,
                count,
                reduction,
                root,
                comm_rank,
                -1,
                comm->global_current_id,
                q,
                pattern_type::collective,
                node_comm,
                p2p,
                done);
        }
        else {
            e = Reduce<T, Proto, Transmit, false>::launch(
                comm->get_node_comm()->get_pattern_data(),
                NRanks,
                (T *)src,
                (T *)dst,
                ipcbuf0,
                ipcbuf1,
                // TODO it can also be a two-shot transmit !
                ipcbuf_size,
                RingTransmit<int, Proto, false>::ringSize / sizeof(T),
                peerbuf0,
                peerbuf1,
                count,
                reduction,
                root,
                comm_rank,
                -1,
                comm->global_current_id,
                q,
                pattern_type::collective,
                node_comm,
                p2p,
                done);
        }
        return e;
    };

    if (ccl::global_data::env().sycl_ll_buffer_global) {
        const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
        sycl::event barrier_event = invoke_barrier(node_comm, q, {}, is_cpu_barrier);
    }

    if (count * dt_sz <= ccl::global_data::env().sycl_reduce_ll_threshold) {
        // small ring with LL
        sycl_e = invoke_pcie_type<Rt64_PCIE>(lambda, comm_size, dtype);
    }
    else {
        // simple ring with LL256
        sycl_e = invoke_pcie_type<Rt64_128_PCIE>(lambda, comm_size, dtype);
    }

    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }

    if (reduction == ccl::reduction::avg) {
        std::vector<sycl::event> evs;
        evs.push_back(sycl_e);
        sycl_e = sycl_average(q, dst, count, comm_size, dtype, evs);

        if (comm->is_multi_thread_instance() == true) {
            pthread_barrier_wait(ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        }
    }
    return ccl::event::create_from_native(sycl_e);
}

template ccl::event reduce_ll<RingTransmit>(const void *,
                                               void *,
                                               size_t,
                                               ccl::datatype,
                                               int,
                                               ccl::reduction,
                                               ccl_comm *,
                                               ccl_stream *,
                                               bool &);

template <typename T,
          template <typename, int> class Proto,
          template <typename, template <typename, int> class, bool, int> class Transmit,
          bool LoadSeqNoInRuntime,
          int SubGroupSize>
Reduce<T, Proto, Transmit, LoadSeqNoInRuntime, SubGroupSize>::Reduce(int nranks,
              T* input,
              T* output,
              size_t nelems,
              int root,
              int rank,
              uint32_t* seqNoPtr,
              uint32_t seqNo,
              T* scatterBuf,
              T* gatherBuf,
              size_t ipcbuf_size,
              T* const peerBuf0[],
              T* const peerBuf1[],
              ccl::reduction reduction,
              bool p2p)
            : root(root),
              rank(rank),
              nRanks(nranks),
              ipcbuf_size(ipcbuf_size),
              gr(calcAccessGranularity(input, output, nelems, nranks)),
              workSizeData(calcWorkSize(input, output, nelems * sizeof(T), nranks)),
              reduction(reduction),
              transmit(nranks,
                       input,
                       output,
                       NULL,
                       scatterBuf,
                       gatherBuf,
                       peerBuf0,
                       peerBuf1,
                       workSizeData.workSize,
                       rank,
                       seqNoPtr,
                       seqNo,
                       reduction,
                       p2p) {}

template <typename T,
          template <typename, int> class Proto,
          template <typename, template <typename, int> class, bool, int> class Transmit,
          bool LoadSeqNoInRuntime,
          int SubGroupSize>
std::pair<sycl::nd_range<1>, uint64_t> Reduce<T, Proto, Transmit, LoadSeqNoInRuntime, SubGroupSize>::getLaunchParam(sycl::queue q,
                                                              const std::shared_ptr<ccl_comm> comm,
                                                              T* ipcbuf0,
                                                              T* ipcbuf1) const {
    constexpr uint32_t nThreads = 64; /* TODO: get EU/thread config */
// TODO: can be queried
#if defined(CCL_SYCL_ENABLE_PVC)
    constexpr size_t maxSS = 64;
#elif defined(CCL_SYCL_ENABLE_ARCB)
    constexpr size_t maxSS = 20;
#elif defined(CCL_SYCL_ENABLE_ARCA)
    constexpr size_t maxSS = 32;
#endif
    int w = transmit.parallel_sg;
    size_t wirePerSS = nThreads / w;
    size_t nWire = divUp(workSizeData.workSize, ProtoT::wireCapacity);
    size_t nSS = divUp(nWire, wirePerSS);
    auto actualSS = std::min(nSS, maxSS);
    auto nSteps = divUp(nWire, actualSS * wirePerSS);
    using TransmitT = Transmit<T, Proto, LoadSeqNoInRuntime, SubGroupSize>;
    using ProtoT = Proto<T, SubGroupSize>;
    auto nSlot = TransmitT::nSlot;
    nSteps = (nSteps + nSlot - 1) / nSlot;
    // XXX: we over updated sequence number. Should be nSteps / nSlot
    // No harm, but not nice.
    //

    return std::make_pair(
        sycl::nd_range<1>(actualSS * wirePerSS * w * SubGroupSize, nThreads * SubGroupSize),
        nSteps);
}

template <typename T,
          template <typename, int> class Proto,
          template <typename, template <typename, int> class, bool, int> class Transmit,
          bool LoadSeqNoInRuntime,
          int SubGroupSize>
sycl::event Reduce<T, Proto, Transmit, LoadSeqNoInRuntime, SubGroupSize>::launch(LLPatternData ll_pattern_data,
                            int nranks,
                            T* input,
                            T* output,
                            T* ipcbuf0,
                            T* ipcbuf1,
                            size_t ipcbuf_size,
                            size_t tmp_buf_size,
                            T* const peerbuf0[],
                            T* const peerbuf1[],
                            size_t nelems,
                            ccl::reduction reduction,
                            int root,
                            int rank,
                            int peer_rank,
                            int global_current_id,
                            sycl::queue queue,
                            pattern_type type,
                            const std::shared_ptr<ccl_comm> comm,
                            bool p2p,
                            bool& done) {
    sycl::event e2;
    done = true;

    if (LoadSeqNoInRuntime) {
        Reduce<T, Proto, Transmit, LoadSeqNoInRuntime, SubGroupSize> offload(nranks,
                            input,
                            output,
                            nelems,
                            root,
                            rank,
                            ll_pattern_data.get_tmp_pattern_ptr(),
                            0,
                            ipcbuf0,
                            ipcbuf1,
                            ipcbuf_size,
                            peerbuf0,
                            peerbuf1,
                            reduction,
                            p2p);

        auto [ndrange, n_steps] = offload.getLaunchParam(queue, comm, ipcbuf0, ipcbuf1);
        ccl_comm_barrier_data barrier_data = comm->barrier_data();

        queue.submit([&](sycl::handler& h) { // TODO port to sycl free functions
            h.parallel_for<class oneccl_reduce_counter_pcie<T,
                                                                Proto,
                                                                LoadSeqNoInRuntime,
                                                                SubGroupSize,
                                                                Transmit>>(
                sycl::nd_range<1>({ 16 }, 16),
                [ll_pattern_data,
                    barrier_data,
                    ipcbuf0,
                    ipcbuf1,
                    tmp_buf_size,
                    type,
                    global_current_id,
                    rank,
                    peer_rank,
                    n_steps](sycl::nd_item<1> it) {
                if (it.get_global_linear_id() == 0) {
                    ll_pattern_data.process_update(
                        type, global_current_id, rank, peer_rank, n_steps);
                }
                check_zero_buffers_gpu(ll_pattern_data,
                                        it,
                                        barrier_data,
                                        ipcbuf0,
                                        ipcbuf1,
                                        tmp_buf_size);
                });
        });

        e2 = queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class oneccl_reduce_pcie<T,
                                                            Proto,
                                                            LoadSeqNoInRuntime,
                                                            SubGroupSize,
                                                            Transmit>>(ndrange, offload);
        });
    }
    else {
        ll_pattern_data.calc_rt_pattern(type, rank, peer_rank);

        Reduce<T, Proto, Transmit, LoadSeqNoInRuntime, SubGroupSize> offload(nranks,
                            input,
                            output,
                            nelems,
                            root,
                            rank,
                            nullptr, // ptr is unused
                            ll_pattern_data.get_tmp_pattern(),
                            ipcbuf0,
                            ipcbuf1,
                            ipcbuf_size,
                            peerbuf0,
                            peerbuf1,
                            reduction,
                            p2p);

        auto [ndrange, n_steps] = offload.getLaunchParam(queue, comm, ipcbuf0, ipcbuf1);
        ll_pattern_data.update_rt_pattern(type, peer_rank, n_steps);
        check_zero_buffers_cpu(ll_pattern_data, comm, queue, ipcbuf0, ipcbuf1, tmp_buf_size);

        e2 = queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class oneccl_reduce_pcie<T,
                                                            Proto,
                                                            LoadSeqNoInRuntime,
                                                            SubGroupSize,
                                                            Transmit>>(ndrange, offload);
        });
    }
    return e2;
}

template <typename T,
          template <typename, int> class Proto,
          template <typename, template <typename, int> class, bool, int> class Transmit,
          bool LoadSeqNoInRuntime,
          int SubGroupSize>
void Reduce<T, Proto, Transmit, LoadSeqNoInRuntime, SubGroupSize>::operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (sycl::nd_item<1> pos) const {
    using TransmitT = Transmit<T, Proto, LoadSeqNoInRuntime, SubGroupSize>;
    using ProtoT = Proto<T, SubGroupSize>;

    size_t nWires = pos.get_global_range(0) / SubGroupSize;
    // parallel_sg wires are bundled together to form a cable
    size_t cableId = pos.get_global_id(0) / SubGroupSize / transmit.parallel_sg;

    size_t loopSize = TransmitT::calcLoopSize(nWires, transmit.parallel_sg, ipcbuf_size);
    size_t local_id = pos.get_sub_group().get_local_id()[0];

    for (size_t offset = 0, iter = 0; offset < workSizeData.workSize;
            offset += loopSize, ++iter) {
        // offset is the global offset for the current loop iteration, iter is the loop iteration index
        // wireOff is the offset for the current wire (sub-group) in the global data, calculated based on the cableId and the global offset
        size_t wireOff = offset + cableId * ProtoT::wireCapacity;
        // workLeft is how much data in bytes is left for the current wire (sub-group) to process in this loop iteration
        ssize_t workLeft = std::min(workSizeData.workSize, offset + loopSize) - wireOff;

#if defined(__enable_device_verbose__)
        if (local_id == 0)
            sycl::ext::oneapi::experimental::printf(
                "wireOff %d, workLeft %ld, cableId %d, rank %d, gOff %ld, loopSize: %ld\n",
                wireOff,
                workLeft,
                cableId,
                rank,
                gOff,
                loopSize);
#endif

        const_cast<TransmitT*>(&transmit)->runReduce(
            root, wireOff, iter, workLeft, workSizeData, gr);
    }
}

template <typename T,
          template <typename, int> class Proto,
          template <typename, template <typename, int> class, bool, int> class Transmit,
          bool LoadSeqNoInRuntime,
          int SubGroupSize>
AccessGranularity Reduce<T, Proto, Transmit, LoadSeqNoInRuntime, SubGroupSize>::calcAccessGranularity(T* input,
                                                       T* output,
                                                       size_t count,
                                                       size_t nranks) {
    using ProtoT = Proto<T, SubGroupSize>;
    using message_t = typename ProtoT::message_t;

    if ((uintptr_t)input % sizeof(message_t) == 0 &&
        (uintptr_t)output % sizeof(message_t) == 0) {
        // aligned on message_t read by vector
        return AccessGranularity::Vector;
    }
    else if ((uintptr_t)input % sizeof(T) == 0 && (uintptr_t)output % sizeof(T) == 0) {
        // aligned on underlying type, read by item
        return AccessGranularity::Item;
    }
    // unaligned, read by byte
    return AccessGranularity::Byte;
}

template <typename T,
          template <typename, int> class Proto,
          template <typename, template <typename, int> class, bool, int> class Transmit,
          bool LoadSeqNoInRuntime,
          int SubGroupSize>
typename Reduce<T, Proto, Transmit, LoadSeqNoInRuntime, SubGroupSize>::WorkSizeDataT Reduce<T, Proto, Transmit, LoadSeqNoInRuntime, SubGroupSize>::calcWorkSize(T* input, T* output, size_t size, int NRanks) {
    using TransmitT = Transmit<T, Proto, LoadSeqNoInRuntime, SubGroupSize>;
    using WorkSizeDataT = typename Reduce<T, Proto, Transmit, LoadSeqNoInRuntime, SubGroupSize>::WorkSizeDataT;
    return WorkSizeDataT(NRanks, size, sizeof(T), TransmitT::splitType(TransmitColl::Reduce));
}
