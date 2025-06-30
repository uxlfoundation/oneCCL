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
          int NRanks,
          template <typename, int>
          class Proto,
          template <typename, int, template <typename, int> class, int>
          class Transmit,
          int SubGroupSize = 16>
struct AllReduce : public Transmit<T, NRanks, Proto, SubGroupSize> {
    using Super = Transmit<T, NRanks, Proto, SubGroupSize>;
    using message_t = typename Super::message_t;
    constexpr static int wireCapacity = Super::wireCapacity;

    AllReduce(T* input,
              size_t nelems,
              int rank,
              uint32_t seqNo,
              T* scatterBuf,
              T* gatherBuf,
              T* const peerBuf0[],
              T* const peerBuf1[],
              bool p2p)
            : Transmit<T, NRanks, Proto, SubGroupSize>(input,
                                                       scatterBuf,
                                                       gatherBuf,
                                                       peerBuf0,
                                                       peerBuf1,
                                                       calcWorkSize(input, nelems * sizeof(T)),
                                                       rank,
                                                       seqNo,
                                                       p2p),
              workSize(calcWorkSize(input, nelems * sizeof(T))) {}

    static int scatterVerify(uint32_t* host, int rank, uint32_t flag, size_t nWorkElemsInInt);
    static int stage2Verify(T* host, int rank, uint32_t flag, size_t nWorkElemsInInt);

    sycl::nd_range<1> getLaunchParam(uint32_t& updateSeqNo) const {
        constexpr uint32_t nThreads = 64; /* TODO: get EU/thread config */
// TODO: can be queried
#if defined(CCL_SYCL_ENABLE_PVC)
        constexpr size_t maxSS = 64;
#elif defined(CCL_SYCL_ENABLE_ARCB)
        constexpr size_t maxSS = 20;
#elif defined(CCL_SYCL_ENABLE_ARCA)
        constexpr size_t maxSS = 32;
#endif
        int w = Super::parallel_sg;
        size_t wirePerSS = nThreads / w;
        size_t nWire = divUp(workSize, wireCapacity);
        size_t nSS = divUp(nWire, wirePerSS);
        auto actualSS = std::min(nSS, maxSS);
        auto nSteps = divUp(nWire, actualSS * wirePerSS);
        updateSeqNo += nSteps;
        //
        // XXX: we over updated sequence number. Should be nSteps / nSlot
        // No harm, but not nice.
        //

        return sycl::nd_range<1>(actualSS * wirePerSS * w * SubGroupSize, nThreads * SubGroupSize);
    }

    static sycl::event launch(T* input,
                              T* ipcbuf0,
                              T* ipcbuf1,
                              T* const peerbuf0[],
                              T* const peerbuf1[],
                              size_t nelems,
                              int rank,
                              uint32_t& step,
                              sycl::queue queue,
                              bool p2p,
                              bool& done) {
        sycl::event e;
        AllReduce offload(input, nelems, rank, step, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1, p2p);
        if (offload.workSize == 0) {
            done = false;
            return e;
        }
        done = true;

        e = queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(offload.getLaunchParam(step), offload);
        });
        return e;
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
        auto wireId_x = pos.get_global_id(0) / SubGroupSize / Super::parallel_sg;

        auto loopSize = nWires / Super::parallel_sg * wireCapacity;

        for (size_t gOff = 0, tOff = 0; gOff < workSize; gOff += loopSize, ++tOff) {
            auto wireOff = wireId_x * wireCapacity + gOff;

            ssize_t workLeft = workSize - wireOff;
#if defined(__enable_device_verbose__)
            auto local_id = pos.get_sub_group().get_local_id()[0];
            if (local_id == 0)
                sycl::ext::oneapi::experimental::printf(
                    "wireOff %d, workLeft %ld, wireId %d\n", wireOff, workLeft, wireId_x);
#endif
            const_cast<AllReduce*>(this)->runAllreduce(wireOff, tOff, workLeft);
        }
    }

private:
    // TODO: buffer plan and start point calc
    static size_t calcWorkSize(T* input, size_t size) {
        // Input must be message size align
        if ((uintptr_t)input % sizeof(message_t) != 0)
            throw std::logic_error("We only support aligned pointer for now");

        auto nChunks = NRanks;
        auto msgSize = divUp(size, sizeof(message_t));
        auto chunkSize = divUp(msgSize, nChunks);

        if (msgSize * sizeof(message_t) != size || chunkSize * sizeof(message_t) * nChunks > size) {
            //throw std::logic_error("We don't support non-even divide yet");
            return 0;
        }

        // TODO: Production logic needs every rank chunk
        return chunkSize * sizeof(message_t);
    }

    ssize_t workSize;
};
