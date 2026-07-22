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

#include <cerrno>
#include <sycl/sycl.hpp>

#include "coll/algorithms/utils/check_zero_buffers.hpp"
#include "coll/algorithms/utils/tvisa/include/gen_visa_templates.hpp"
#include "coll/algorithms/utils/transmit/transmit.hpp"

template <typename T,
          template <typename, int>
          class Proto,
          template <typename, template <typename, int> class, bool, int>
          class Transmit,
          bool LoadSeqNoInRuntime,
          int SubGroupSize = 16>
struct Broadcast {
    using TransmitT = Transmit<T, Proto, LoadSeqNoInRuntime, SubGroupSize>;
    using ProtoT = Proto<T, SubGroupSize>;
    using message_t = typename ProtoT::message_t;

    Broadcast(int nranks,
              T* input,
              T* output,
              size_t nelems,
              int root,
              int rank,
              uint32_t* seqNoPtr,
              uint32_t seqNo,
              T* scatterBuf,
              T* gatherBuf,
              T* const peerBuf0[],
              T* const peerBuf1[],
              bool p2p)
            : root(root),
              gr(calcAccessGranularity(input, output, nelems)),
              workSize(nelems * sizeof(T)),
              transmit(nranks,
                       input,
                       output,
                       NULL, /* offset */
                       scatterBuf,
                       gatherBuf,
                       peerBuf0,
                       peerBuf1,
                       workSize,
                       rank,
                       seqNoPtr,
                       seqNo,
                       p2p) {}

    static int scatterVerify(uint32_t* host, int rank, uint32_t flag, size_t nWorkElemsInInt);
    static int stage2Verify(T* host, int rank, uint32_t flag, size_t nWorkElemsInInt);

    std::pair<sycl::nd_range<1>, uint64_t> getLaunchParam(sycl::queue q,
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
        size_t nWire = divUp(workSize, ProtoT::wireCapacity);
        size_t nSS = divUp(nWire, wirePerSS);
        auto actualSS = std::min(nSS, maxSS);
        auto nSteps = divUp(nWire, actualSS * wirePerSS);
        auto nSlot = Transmit<T, Proto, LoadSeqNoInRuntime, SubGroupSize>::nSlot;
        //
        // XXX: we over updated sequence number. Should be nSteps / nSlot
        // No harm, but not nice.
        //

        return std::make_pair(
            sycl::nd_range<1>(actualSS * wirePerSS * w * SubGroupSize, nThreads * SubGroupSize),
            nSteps);
    }

    static sycl::event launch(LLPatternData ll_pattern_data,
                              int nranks,
                              T* input,
                              T* output,
                              T* ipcbuf0,
                              T* ipcbuf1,
                              size_t tmp_buf_size,
                              T* const peerbuf0[],
                              T* const peerbuf1[],
                              size_t nelems,
                              int root,
                              int rank,
                              int global_current_id,
                              sycl::queue queue,
                              const std::shared_ptr<ccl_comm> comm,
                              bool p2p,
                              bool& done) {
        sycl::event e1, e2;

        if (LoadSeqNoInRuntime) {
            Broadcast offload(nranks,
                              input,
                              output,
                              nelems,
                              root,
                              rank,
                              ll_pattern_data.get_tmp_pattern_ptr(),
                              0,
                              ipcbuf0,
                              ipcbuf1,
                              peerbuf0,
                              peerbuf1,
                              p2p);
            done = true;

            auto [ndrange, n_steps] = offload.getLaunchParam(queue, comm, ipcbuf0, ipcbuf1);
            ccl_comm_barrier_data barrier_data = comm->barrier_data();

            e1 = queue.submit([&](auto& h) {
                h.parallel_for(
                    sycl::nd_range<1>({ 16 }, 16),
                    [ll_pattern_data,
                     barrier_data,
                     ipcbuf0,
                     ipcbuf1,
                     tmp_buf_size,
                     global_current_id,
                     rank,
                     n_steps](sycl::nd_item<1> it) {
                        if (it.get_global_linear_id() == 0) {
                            ll_pattern_data.process_update(
                                pattern_type::collective, global_current_id, rank, -1, n_steps);
                        }
                        check_zero_buffers_gpu(
                            ll_pattern_data, it, barrier_data, ipcbuf0, ipcbuf1, tmp_buf_size);
                    });
            });

            e2 = queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(ndrange, offload);
            });
        }
        else {
            ll_pattern_data.calc_rt_pattern(pattern_type::collective, rank, -1);

            Broadcast offload(nranks,
                              input,
                              output,
                              nelems,
                              root,
                              rank,
                              nullptr, // ptr is unused
                              ll_pattern_data.get_tmp_pattern(),
                              ipcbuf0,
                              ipcbuf1,
                              peerbuf0,
                              peerbuf1,
                              p2p);
            done = true;

            auto [ndrange, n_steps] = offload.getLaunchParam(queue, comm, ipcbuf0, ipcbuf1);
            ll_pattern_data.update_rt_pattern(pattern_type::collective, -1, n_steps);
            check_zero_buffers_cpu(ll_pattern_data, comm, queue, ipcbuf0, ipcbuf1, tmp_buf_size);

            e2 = queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(ndrange, offload);
            });
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

        auto loopSize = nWires / transmit.parallel_sg * ProtoT::wireCapacity;

        for (size_t gOff = 0, tOff = 0; gOff < workSize; gOff += loopSize, ++tOff) {
            auto wireOff = wireId_x * ProtoT::wireCapacity + gOff;

            ssize_t workLeft = workSize - wireOff;
#if defined(__enable_device_verbose__)
            auto local_id = pos.get_sub_group().get_local_id()[0];
            if (local_id == 0)
                sycl::ext::oneapi::experimental::printf(
                    "wireOff %d, workLeft %ld, wireId %d\n", wireOff, workLeft, wireId_x);
#endif
            const_cast<TransmitT*>(&transmit)->runBroadcast(root, wireOff, tOff, workLeft, gr);
        }
    }

private:
    static AccessGranularity calcAccessGranularity(T* input, T* output, size_t count) {
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

    AccessGranularity gr;
    ssize_t workSize;
    int root;
    TransmitT transmit;
};
