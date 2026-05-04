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

#include "coll/algorithms/utils/transmit/work_size_data.hpp"

// sub-group level two-shots transmit, minimum latency expected

template <typename T,
          template <typename, int>
          class Proto,
          bool LoadSeqNoInRuntime,
          int SubGroupSize = 16>
class TwoShotsTransmit {
protected:
    using ProtoT = Proto<T, SubGroupSize>;
    using message_t = typename ProtoT::message_t;

public:
    const int parallel_sg;
    constexpr static size_t nSlot = 8;
    constexpr static size_t ringSize = ProtoT::wireTransSize * nSlot;
    constexpr static size_t maxLaunch = 1;

    typedef T (*twoshotsPtr)[nSlot][ProtoT::wireTransElems];

public:
    TwoShotsTransmit(int nranks,
                     T* input,
                     T* output,
                     const size_t* offs, // offset is not supported currently
                     T* scatterBuf,
                     T* gatherBuf,
                     T* const peerBuf0[],
                     T* const peerBuf1[],
                     ssize_t workSize,
                     int rank,
                     uint32_t* seqNoPtr, // Serve as flag for checking
                     uint32_t seqNo, // Serve as flag for checking
                     ccl::reduction reduction,
                     bool p2p)
            : NRanks(nranks),
              parallel_sg(nranks),
              workElems(workSize / sizeof(T)),
              rank(rank),
              seqNoPtr(seqNoPtr),
              seqNo_(seqNo),
              reduction(reduction),
              p2p(p2p) {
        ingress = input;
        egress = output;

        localGatherSink = reinterpret_cast<twoshotsPtr>(gatherBuf);

        for (int i = 0; i < NRanks; ++i)
            scatterSinks[i] =
                reinterpret_cast<twoshotsPtr>((uintptr_t)peerBuf0[i] + rank * ringSize);

        for (int i = 0; i < NRanks - 1; ++i) {
            int next = (rank + i + 1) % NRanks;

            localScatterSinks[i] =
                reinterpret_cast<twoshotsPtr>((uintptr_t)scatterBuf + next * ringSize);

            gatherSinks[i] =
                reinterpret_cast<twoshotsPtr>((uintptr_t)peerBuf1[next] + rank * ringSize);
        }
    }

    inline void runAllreduce(size_t inputOffset,
                             size_t tStep,
                             ssize_t workLeft,
                             WorkSizeData<message_t> workSizeData,
                             AccessGranularity gr) {
        if (workLeft <= 0) {
            return;
        }

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto y_id = wireId % NRanks;
        auto x_id = wireId / NRanks * NRanks;

        auto inputOffInType = inputOffset / sizeof(T);
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo + tStep / nSlot;
        auto nelems = workSizeData.GetNElems(y_id) - inputOffInType;

        const int unroll = 1;
        constexpr auto eltPerPack = unroll * ProtoT::wireCapacityInType;
        constexpr bool accumulateShuffledData = sizeof(T) <= sizeof(seqNo);

        message_t v[unroll];

        auto* ptr = ingress + inputOffInType + workSizeData.GetOffset(y_id);
        auto* o_ptr = egress + inputOffInType + workSizeData.GetOffset(y_id);

        ProtoT::loadInput(v[0], ptr, nelems, gr);

        // scatter and gather
        if (y_id != rank) {
            ProtoT::shuffleData(v[0]);
            ProtoT::insertFlags(v[0], flag);
            ProtoT::sendMessages(scatterSinks[y_id][x_id][tStep % nSlot], v[0]);

            bool retry;
            do {
                retry = false;
                retry |= ProtoT::recvMessages(v[0], localGatherSink[wireId][tStep % nSlot], flag);
            } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));
        }

        // wait reduce bcast
        if (y_id == rank) {
            if constexpr (accumulateShuffledData) {
                ProtoT::shuffleData(v[0]);
            }
            message_t messages[unroll];
            for (int i = 0; i < NRanks - 1; ++i) {
                bool retry;
                do {
                    retry = false;
                    retry |= ProtoT::recvMessages(
                        messages[0], localScatterSinks[i][x_id][tStep % nSlot], flag);
                } while (
                    sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));
                if constexpr (!accumulateShuffledData) {
                    ProtoT::restoreData(messages[0]);
                }
                ProtoT::accumMessages(v[0], messages[0]);
            }

            if constexpr (!accumulateShuffledData) {
                ProtoT::shuffleData(v[0]);
            }
            ProtoT::insertFlags(v[0], flag);

#pragma unroll
            for (int i = 0; i < NRanks - 1; ++i)
                ProtoT::sendMessages(gatherSinks[i][x_id][tStep % nSlot], v[0]);
        }

        ProtoT::restoreData(v[0]);
        ProtoT::storeOutput(o_ptr, v[0], nelems, gr);
    }

protected:
    inline int getSeqNo() {
        // the variable is a template parameter
        // should be optimized out
        if (LoadSeqNoInRuntime) {
            return *seqNoPtr;
        }
        else {
            return seqNo_;
        }
    }

    T* ingress;
    T* egress;

    int NRanks;
    ssize_t workElems;
    int rank;
    uint32_t* seqNoPtr;
    uint32_t seqNo_;
    bool p2p;
    ccl::reduction reduction;

    twoshotsPtr scatterSinks[ARC_MAX_NUM];
    twoshotsPtr gatherSinks[ARC_MAX_NUM];
    twoshotsPtr localScatterSinks[ARC_MAX_NUM];
    twoshotsPtr localGatherSink;
};
