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

#include <sycl/sycl.hpp>

#include "coll/algorithms/utils/tvisa/include/gen_visa_templates.hpp"
#include "coll/algorithms/utils/transmit/transmit.hpp"

template <typename T,
          template <typename, int>
          class Proto,
          template <typename, template <typename, int> class, bool, int>
          class Transmit,
          bool LoadSeqNoInRuntime,
          int SubGroupSize = 16>
struct Recv {
    using TransmitT = Transmit<T, Proto, LoadSeqNoInRuntime, SubGroupSize>;
    using ProtoT = Proto<T, SubGroupSize>;
    using message_t = typename ProtoT::message_t;

    Recv(int nranks,
         T* output,
         size_t nelems,
         int my_rank,
         int peer_rank,
         uint32_t* seqNoPtr,
         uint32_t seqNo,
         T* scatterBuf,
         T* gatherBuf,
         T* const peerBuf0[],
         T* const peerBuf1[],
         bool p2p)
            : workSize(nelems * sizeof(T)),
              gr(calcAccessGranularity(output)),
              my_rank(my_rank),
              peer_rank(peer_rank),
              transmit(nranks,
                       output,
                       scatterBuf,
                       gatherBuf,
                       peerBuf0,
                       peerBuf1,
                       workSize,
                       peer_rank - 1,
                       seqNoPtr,
                       seqNo,
                       p2p) {}

    std::pair<sycl::nd_range<1>, uint64_t> getLaunchParam(sycl::queue q,
                                                          const std::shared_ptr<ccl_comm> comm,
                                                          T* ipcbuf0,
                                                          T* ipcbuf1) const {
        constexpr uint32_t nThreads = 64; /* TODO: get EU/thread config */
#if defined(CCL_SYCL_ENABLE_PVC)
        constexpr size_t maxSS = 64;
#elif defined(CCL_SYCL_ENABLE_ARCB)
        constexpr size_t maxSS = 20;
#elif defined(CCL_SYCL_ENABLE_ARCA)
        constexpr size_t maxSS = 32;
#endif
        int w = transmit.parallel_sg;
        size_t wirePerSS = nThreads / w;
        size_t nWire = divUp(workSize, ProtoT::wireCapacity);
        size_t nSS = divUp(nWire, wirePerSS);
        auto actualSS = std::min(nSS, maxSS);
        auto nSteps = divUp(nWire, actualSS * wirePerSS);
        // XXX: we over updated sequence number. Should be nSteps / nSlot
        // No harm, but not nice.
        //

        return std::make_pair(
            sycl::nd_range<1>(actualSS * wirePerSS * w * SubGroupSize, nThreads * SubGroupSize),
            nSteps);
    }

    static sycl::event launch(LLPatternData ll_pattern_data,
                              int nranks,
                              T* output,
                              T* ipcbuf0,
                              T* ipcbuf1,
                              size_t tmp_buf_size,
                              T* const peerbuf0[],
                              T* const peerbuf1[],
                              size_t nelems,
                              int my_rank,
                              int peer_rank,
                              int global_current_id,
                              sycl::queue queue,
                              pattern_type type,
                              const std::shared_ptr<ccl_comm> comm,
                              std::vector<sycl::event>& dep_events,
                              bool p2p,
                              bool& done) {
        sycl::event e1, e2;

        done = true;

        // peer_rank is actual rank minus 1 because the ringTransmit calculate
        // next neighbor by rank plus 1

        if (LoadSeqNoInRuntime) {
            Recv offload(nranks,
                         output,
                         nelems,
                         my_rank,
                         peer_rank,
                         ll_pattern_data.get_tmp_pattern_ptr(),
                         0,
                         ipcbuf0,
                         ipcbuf1,
                         peerbuf0,
                         peerbuf1,
                         p2p);

            auto [ndrange, n_steps] = offload.getLaunchParam(queue, comm, ipcbuf0, ipcbuf1);
            ccl_comm_barrier_data barrier_data = comm->barrier_data();

            e1 = queue.submit([&](auto& h) { // TODO port to sycl free functions
                h.parallel_for(
                    sycl::nd_range<1>({ 16 }, 16),
                    [ll_pattern_data,
                     barrier_data,
                     ipcbuf0,
                     ipcbuf1,
                     tmp_buf_size,
                     type,
                     global_current_id,
                     my_rank,
                     peer_rank,
                     n_steps](sycl::nd_item<1> it) {
                        if (it.get_global_linear_id() == 0) {
                            ll_pattern_data.process_update(
                                type, global_current_id, my_rank, peer_rank, n_steps);
                        }
                        check_zero_buffers_gpu(
                            ll_pattern_data, it, barrier_data, ipcbuf0, ipcbuf1, tmp_buf_size);
                    });
            });

            size_t global_range_size = ndrange.get_global_range()[0];
            size_t nWires = global_range_size / SubGroupSize;
            auto loopSize = nWires * ProtoT::wireCapacity / offload.transmit.parallel_sg;
            CCL_THROW_IF_NOT(loopSize > 0, "loopSize must be greater than 0");
            e2 = queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(dep_events);
                cgh.parallel_for(ndrange, offload);
            });
        }
        else {
            ll_pattern_data.calc_rt_pattern(type, my_rank, peer_rank);

            Recv offload(nranks,
                         output,
                         nelems,
                         my_rank,
                         peer_rank,
                         nullptr, // ptr is unused
                         ll_pattern_data.get_tmp_pattern(),
                         ipcbuf0,
                         ipcbuf1,
                         peerbuf0,
                         peerbuf1,
                         p2p);

            auto [ndrange, n_steps] = offload.getLaunchParam(queue, comm, ipcbuf0, ipcbuf1);
            ll_pattern_data.update_rt_pattern(type, peer_rank, n_steps);
            check_zero_buffers_cpu(ll_pattern_data, comm, queue, ipcbuf0, ipcbuf1, tmp_buf_size);

            size_t global_range_size = ndrange.get_global_range()[0];
            size_t nWires = global_range_size / SubGroupSize;
            auto loopSize = nWires * ProtoT::wireCapacity / offload.transmit.parallel_sg;
            CCL_THROW_IF_NOT(loopSize > 0, "loopSize must be greater than 0");
            e2 = queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(dep_events);
                cgh.parallel_for(ndrange, offload);
            });
            // FIXME: work-around for vllm test
            // check the commit message for more info
            e2.wait();
        }

        return e2;
    }
    //
    // Found this analogy fascinating:
    //
    // Let correspond sub-group to wire, sequential guaranteed.
    // Bundle sub-groups(wires) into group(cable).
    //
    // Total cables will deliver the full capacity of single loop.
    //
    void operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (sycl::nd_item<1> pos) const {
        auto nWires = pos.get_global_range(0) / SubGroupSize;
        auto wireId_x = pos.get_global_id(0) / SubGroupSize / transmit.parallel_sg;

        auto loopSize = nWires * ProtoT::wireCapacity / transmit.parallel_sg;

        const_cast<TransmitT*>(&transmit)->runRecvStart(my_rank, peer_rank);

        for (size_t gOff = 0, tOff = 0; gOff < workSize; gOff += loopSize, ++tOff) {
            auto wireOff = wireId_x * ProtoT::wireCapacity + gOff;

            ssize_t workLeft = workSize - wireOff;
#if defined(__enable_device_verbose__)
            auto local_id = pos.get_sub_group().get_local_id()[0];
            if (local_id == 0)
                sycl::ext::oneapi::experimental::printf(
                    "wireOff %d, workLeft %ld, wireId %d\n", wireOff, workLeft, wireId_x);
#endif
            const_cast<TransmitT*>(&transmit)->runRecv(
                wireOff, tOff, workLeft, my_rank, peer_rank, gr);
        }
    }

private:
    static AccessGranularity calcAccessGranularity(T* output) {
        if ((uintptr_t)output % sizeof(message_t) == 0) {
            // aligned on message_t read by vector
            return AccessGranularity::Vector;
        }
        else if ((uintptr_t)output % sizeof(T) == 0) {
            // aligned on underlying type, read by item
            return AccessGranularity::Item;
        }
        // unaligned, read by byte
        return AccessGranularity::Byte;
    }

    ssize_t workSize;
    AccessGranularity gr;
    int my_rank;
    int peer_rank;
    TransmitT transmit;
};
