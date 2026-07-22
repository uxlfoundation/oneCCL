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

#include "coll/algorithms/utils/transmit/work_size_data.hpp"
#include "coll/reduction/reduction.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"
#include "common/utils/rounding.hpp"
#include "work_size_data.hpp"

//
// For those do not have sub-group level independent forward progress
// or PCIE connection without switch (remote polling).
//

template <typename T,
          template <typename, int>
          class Proto,
          bool LoadSeqNoInRuntime,
          int SubGroupSize = 16>
class RingTransmit {
protected:
    using ProtoT = Proto<T, SubGroupSize>;
    using message_t = typename ProtoT::message_t;

public:
    const int parallel_sg;
    constexpr static size_t nSlot = 4;
#if defined(CCL_SYCL_ENABLE_ARCB)
    constexpr static size_t maxLaunch = 64 * 20;
#else
    constexpr static size_t maxLaunch = 64 * 64;
#endif
    constexpr static size_t ringSize = maxLaunch * ProtoT::wireTransSize * nSlot;
    static_assert(ringSize <= 4 * 1024 * 1024ull * SubGroupSize / 16);
    constexpr static SplitType splitType = SplitType::NoOverlap;

    typedef T (*ringPtr)[nSlot][maxLaunch][ProtoT::wireTransElems];

    static size_t calcLoopSize(size_t nWires, int parallel_sg, size_t /*ipcbuf_size*/) {
        return nWires * ProtoT::wireCapacity / parallel_sg;
    }

public:
    RingTransmit(int nranks,
                 T* input,
                 T* scatterBuf,
                 T* gatherBuf,
                 T* const peerBuf0[],
                 T* const peerBuf1[],
                 ssize_t workSize,
                 int rank,
                 uint32_t* seqNoPtr, // Serves as gpu-based flag (sycl graph)
                 uint32_t seqNo, // Serves as const flag for checking (not recording)
                 bool p2p)
            : nRanks(nranks),
              parallel_sg(1),
              workElems(workSize / sizeof(T)),
              rank(rank),
              seqNoPtr(seqNoPtr),
              seqNo_(seqNo),
              p2p(p2p) {
        auto next = (rank + 1) % nRanks;
        ingress = input;
        egress = input;
        has_offsets = false;

        scatterSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf0[next]);
        gatherSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf1[next]);

        localScatterSink = reinterpret_cast<ringPtr>((uintptr_t)scatterBuf);
        localGatherSink = reinterpret_cast<ringPtr>((uintptr_t)gatherBuf);
    }

    RingTransmit(int nranks,
                 T* input,
                 T* output,
                 const size_t* offs,
                 T* scatterBuf,
                 T* gatherBuf,
                 T* const peerBuf0[],
                 T* const peerBuf1[],
                 ssize_t workSize,
                 int rank,
                 uint32_t* seqNoPtr, // Serves as gpu-based flag (sycl graph)
                 uint32_t seqNo, // Serves as const flag for checking (not recording)
                 ccl::reduction reduction,
                 bool p2p)
            : nRanks(nranks),
              parallel_sg(1),
              workElems(workSize / sizeof(T)),
              rank(rank),
              seqNoPtr(seqNoPtr),
              seqNo_(seqNo),
              reduction(reduction),
              reduction_data{},
              p2p(p2p) {
        if (ccl_reduction_type_storage::is_custom(reduction)) {
            effective_reduction = ccl::reduction::sum;
            reduction_data = make_reduction_operation(reduction);
        } else {
            effective_reduction = reduction;
        }
        init_ring_buffers(input, output, offs, scatterBuf, gatherBuf, peerBuf0, peerBuf1, nranks);
    }

    RingTransmit(int nranks,
                 T* input,
                 T* output,
                 const size_t* offs,
                 T* scatterBuf,
                 T* gatherBuf,
                 T* const peerBuf0[],
                 T* const peerBuf1[],
                 ssize_t workSize,
                 int rank,
                 uint32_t* seqNoPtr,
                 uint32_t seqNo,
                 bool p2p)
            : RingTransmit(nranks,
                           input,
                           output,
                           offs,
                           scatterBuf,
                           gatherBuf,
                           peerBuf0,
                           peerBuf1,
                           workSize,
                           rank,
                           seqNoPtr,
                           seqNo,
                           ccl::reduction::sum,
                           p2p) {}

    inline void runAllreduce(size_t inputOffset,
                             size_t tStep,
                             ssize_t workLeft,
                             WorkSizeData<message_t> workSizeData,
                             AccessGranularity gr) {
        if (workLeft <= 0) {
            // threads without work paticipate in exactly same number of
            // barrier as those threads with actual work
            ProtoT::sbarrier_signal_compat(p2p);
            for (uint32_t i = 1; i < nRanks - 1; ++i) {
                ProtoT::sbarrier_wait_compat(p2p);
                ProtoT::sbarrier_signal_compat(p2p);
            }
            ProtoT::sbarrier_wait_compat(p2p);
            ProtoT::sbarrier_signal_compat(p2p);
            for (uint32_t i = 1; i < nRanks - 1; ++i) {
                ProtoT::sbarrier_wait_compat(p2p);
                ProtoT::sbarrier_signal_compat(p2p);
            }
            ProtoT::sbarrier_wait_compat(p2p);
            return;
        }

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        size_t offset = inputOffset / sizeof(T);
        uint32_t seqNo = getSeqNo();
        size_t flag = seqNo + tStep / nSlot;
        size_t slot = (seqNo + tStep) % nSlot;

        size_t* offset_ptr = has_offsets ? offsets : NULL;

        uint32_t p_idx = 0;
        ssize_t peer = (rank + p_idx) % nRanks;

        // Step 0
        ProtoT::send(ingress,
                     scatterSink[peer][slot][wireId],
                     wireId,
                     peer,
                     0,
                     workSizeData.GetOffset(peer) + offset,
                     flag,
                     slot,
                     workSizeData.GetNElems(peer) - offset,
                     reduction_data,
                     p2p,
                     gr);

        // Step 1 to N-1
#pragma unroll
        for (int i = 1; i < nRanks - 1; ++i) {
            p_idx = (p_idx + nRanks - 1) % nRanks;
            peer = (rank + p_idx) % nRanks;
            ProtoT::loadRecvReduceSend(ingress,
                                       localScatterSink[peer][slot][wireId],
                                       scatterSink[peer][slot][wireId],
                                       wireId,
                                       peer,
                                       0,
                                       workSizeData.GetOffset(peer) + offset,
                                       flag,
                                       slot,
                                       workSizeData.GetNElems(peer) - offset,
                                       effective_reduction,
                                       reduction_data,
                                       p2p,
                                       gr);
        }

        // Step N
        p_idx = (p_idx + nRanks - 1) % nRanks;
        peer = (rank + p_idx) % nRanks;
        ProtoT::loadRecvReduceSendWrtback(ingress,
                                          localScatterSink[peer][slot][wireId],
                                          gatherSink[peer][slot][wireId],
                                          egress,
                                          wireId,
                                          peer,
                                          0,
                                          workSizeData.GetOffset(peer) + offset,
                                          flag,
                                          slot,
                                          workSizeData.GetNElems(peer) - offset,
                                          effective_reduction,
                                          reduction_data,
                                          p2p,
                                          gr);

        // write back
#pragma unroll
        for (uint32_t i = 1; i < nRanks - 1; ++i) {
            p_idx = (p_idx + nRanks - 1) % nRanks; // 0
            peer = (rank + p_idx) % nRanks;
            ProtoT::recvSendWrtback(localGatherSink[peer][slot][wireId],
                                    gatherSink[peer][slot][wireId],
                                    egress,
                                    offset_ptr,
                                    wireId,
                                    peer,
                                    0,
                                    workSizeData.GetOffset(peer) + offset,
                                    flag,
                                    slot,
                                    workSizeData.GetNElems(peer) - offset,
                                    p2p,
                                    gr);
        }

        p_idx = (p_idx + nRanks - 1) % nRanks;
        peer = (rank + p_idx) % nRanks;
        ProtoT::recvWrtback(localGatherSink[peer][slot][wireId],
                            egress,
                            offset_ptr,
                            wireId,
                            peer,
                            0,
                            workSizeData.GetOffset(peer) + offset,
                            flag,
                            slot,
                            workSizeData.GetNElems(peer) - offset,
                            p2p,
                            gr);
    }

    inline void runAllgather(size_t inputOffset,
                             size_t tStep,
                             ssize_t workLeft,
                             AccessGranularity gr) {
        if (workLeft <= 0) {
            ProtoT::sbarrier_signal_compat(p2p);
            for (uint32_t i = 1; i < nRanks - 1; ++i) {
                ProtoT::sbarrier_wait_compat(p2p);
                ProtoT::sbarrier_signal_compat(p2p);
            }
            ProtoT::sbarrier_wait_compat(p2p);
            return;
        }

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto offset = inputOffset / sizeof(T);
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo + tStep / nSlot;
        auto slot = (seqNo + tStep) % nSlot;
        auto nelems = workLeft / sizeof(T);
        auto offset_ptr = has_offsets ? offsets : NULL;

        message_t v;

        uint32_t p_idx = 0;
        int peer = (rank + p_idx) % nRanks;

        auto* ptr = ingress + offset;
        ProtoT::loadInput(v, ptr, nelems, gr);

        auto* o_ptr = egress + offset;
        if (has_offsets)
            o_ptr = (T*)((char*)o_ptr + offsets[peer]);
        else
            o_ptr = o_ptr + peer * workElems;

        if (ptr != o_ptr)
            ProtoT::storeOutput(o_ptr, v, nelems, gr);

        ProtoT::shuffleData(v);
        ProtoT::insertFlags(v, flag);
        ProtoT::sendMessages(gatherSink[peer][slot][wireId], v);

        ProtoT::sbarrier_signal_compat(p2p);

#pragma unroll
        for (uint32_t i = 1; i < nRanks - 1; ++i) {
            p_idx = (p_idx + nRanks - 1) % nRanks; // 0
            peer = (rank + p_idx) % nRanks;
            ProtoT::recvSendWrtback(localGatherSink[peer][slot][wireId],
                                    gatherSink[peer][slot][wireId],
                                    egress,
                                    offset_ptr,
                                    wireId,
                                    peer,
                                    workElems,
                                    offset,
                                    flag,
                                    slot,
                                    nelems,
                                    p2p,
                                    gr);
        }

        p_idx = (p_idx + nRanks - 1) % nRanks;
        peer = (rank + p_idx) % nRanks;

        ProtoT::recvWrtback(localGatherSink[peer][slot][wireId],
                            egress,
                            offset_ptr,
                            wireId,
                            peer,
                            workElems,
                            offset,
                            flag,
                            slot,
                            nelems,
                            p2p,
                            gr);
    }

    inline void runReduceScatter(size_t inputOffset,
                                 size_t tStep,
                                 ssize_t workLeft,
                                 ssize_t workSizeData,
                                 AccessGranularity gr) {
        if (workLeft <= 0) {
            ProtoT::sbarrier_signal_compat(p2p);
            for (uint32_t i = 1; i < nRanks - 1; ++i) {
                ProtoT::sbarrier_wait_compat(p2p);
                ProtoT::sbarrier_signal_compat(p2p);
            }
            ProtoT::sbarrier_wait_compat(p2p);
            return;
        }

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto offset = inputOffset / sizeof(T);
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo + tStep / nSlot;
        auto slot = (seqNo + tStep) % nSlot;
        auto nelems = workLeft / sizeof(T);

        int p_idx = -1;
        int peer = (rank + nRanks + p_idx) % nRanks;

        // Step 0
        ProtoT::send(ingress,
                     scatterSink[peer][slot][wireId],
                     wireId,
                     peer,
                     workElems,
                     offset,
                     flag,
                     slot,
                     nelems,
                     reduction_data,
                     p2p,
                     gr);

        // Step 1 to N-1
#pragma unroll
        for (int i = 1; i < nRanks - 1; ++i) {
            p_idx = (p_idx + nRanks - 1) % nRanks;
            peer = (rank + p_idx) % nRanks;
            ProtoT::loadRecvReduceSend(ingress,
                                       localScatterSink[peer][slot][wireId],
                                       scatterSink[peer][slot][wireId],
                                       wireId,
                                       peer,
                                       workElems,
                                       offset,
                                       flag,
                                       slot,
                                       nelems,
                                       effective_reduction,
                                       reduction_data,
                                       p2p,
                                       gr);
        }

        // Step N
        p_idx = (p_idx + nRanks - 1) % nRanks;
        peer = (rank + p_idx) % nRanks;
        ProtoT::loadRecvReduceWrtback(ingress,
                                      localScatterSink[peer][slot][wireId],
                                      egress,
                                      wireId,
                                      peer,
                                      workElems,
                                      offset,
                                      flag,
                                      slot,
                                      nelems,
                                      effective_reduction,
                                      reduction_data,
                                      p2p,
                                      gr);
    }

    inline void runBroadcast(int root,
                             size_t inputOffset,
                             size_t tStep,
                             ssize_t workLeft,
                             AccessGranularity gr) {
        if (workLeft <= 0) {
            return;
        }

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto inputOffInType = inputOffset / sizeof(T);
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo + tStep;
        auto slot = (seqNo + tStep) % nSlot;
        auto nelems = workLeft / sizeof(T);

        message_t v;

        int peer = (rank + 1) % nRanks;

        if (rank == root) {
            // recv
            if (tStep >= nSlot) {
                // halt to wait for previous step
                auto prev_flag = seqNo + tStep - nSlot;
                auto prev_slot = (seqNo + tStep - nSlot) % nSlot;
                bool retry;
                do {
                    retry = false;
                    retry |= ProtoT::recvMessages(
                        v, localGatherSink[rank][prev_slot][wireId], prev_flag);
                } while (
                    sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));
            }

            auto* ptr = ingress + inputOffInType;
            ProtoT::loadInput(v, ptr, nelems, gr);

            auto* o_ptr = egress + inputOffInType;
            if (ptr != o_ptr)
                ProtoT::storeOutput(o_ptr, v, nelems, gr);

            ProtoT::shuffleData(v);
            ProtoT::insertFlags(v, flag);
            ProtoT::sendMessages(scatterSink[peer][slot][wireId], v);
        }
        else {
            // recv
            bool retry;
            do {
                retry = false;
                retry |= ProtoT::recvMessages(v, localScatterSink[rank][slot][wireId], flag);
            } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

            // forward
            if (peer != root) {
                ProtoT::sendMessages(scatterSink[peer][slot][wireId], v);
            }
            else {
                ProtoT::sendMessages(gatherSink[peer][slot][wireId], v);
            }

            ProtoT::restoreData(v);
            auto* ptr = egress + inputOffInType;
            ProtoT::storeOutput(ptr, v, nelems, gr);
        }
    }

    inline void runSendStart(int my_rank, int peer_rank) {
        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        uint32_t seqNo = getSeqNo();
        auto flag = seqNo;
        auto slot = seqNo % nSlot;

        message_t v;
        // wait for receiver to be ready
        bool retry;
        do {
            retry = false;
            retry |= ProtoT::recvMessages(v, localGatherSink[peer_rank][slot][wireId], flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));
    }

    inline void runSend(size_t inputOffset,
                        size_t tStep,
                        ssize_t workLeft,
                        int my_rank,
                        int peer_rank,
                        AccessGranularity gr) {
        if (workLeft <= 0)
            return;

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto inputOffInType = inputOffset / sizeof(T);
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo + tStep / nSlot;
        auto slot = (seqNo + tStep) % nSlot;
        auto nelems = workLeft / sizeof(T);

        bool retry;
        message_t v;

        auto* ptr = ingress + inputOffInType;
        ProtoT::loadInput(v, ptr, nelems, gr);

        ProtoT::shuffleData(v);
        ProtoT::insertFlags(v, flag);
        ProtoT::sendMessages(scatterSink[my_rank][slot][wireId], v);

        do {
            retry = false;
            retry |= ProtoT::recvMessages(v, localGatherSink[peer_rank][slot][wireId], flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));
    }

    inline void runRecvStart(int my_rank, int peer_rank) {
        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo;
        auto slot = seqNo % nSlot;
        message_t v;
        ProtoT::insertFlags(v, flag);
        ProtoT::sendMessages(gatherSink[my_rank][slot][wireId], v);
    }

    inline void runRecv(size_t outputOffset,
                        size_t tStep,
                        ssize_t workLeft,
                        int my_rank,
                        int peer_rank,
                        AccessGranularity gr) {
        if (workLeft <= 0)
            return;

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto offset = outputOffset / sizeof(T);
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo + tStep / nSlot;
        auto slot = (seqNo + tStep) % nSlot;
        auto nelems = workLeft / sizeof(T);

        message_t v;

        bool retry;
        do {
            retry = false;
            retry |= ProtoT::recvMessages(v, localScatterSink[peer_rank][slot][wireId], flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

        ProtoT::insertFlags(v, flag);
        ProtoT::sendMessages(gatherSink[my_rank][slot][wireId], v);

        ProtoT::restoreData(v);

        auto* ptr = egress + offset;
        ProtoT::storeOutput(ptr, v, nelems, gr);
    }

    inline void runAllToAll(size_t inputOffset,
                            size_t tStep,
                            ssize_t workLeft,
                            AccessGranularity gr) {
        if (workLeft <= 0) {
            for (int step = 1; step < nRanks; step++) {
                ProtoT::sbarrier_signal_compat(p2p);
                ProtoT::sbarrier_wait_compat(p2p);
            }
            return;
        }

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto offset = inputOffset / sizeof(T);
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo + tStep / nSlot;
        auto slot = (seqNo + tStep) % nSlot;
        auto nelems = workLeft / sizeof(T);

        message_t v;

        for (int step = 1; step < nRanks; step++) {
            int send_peer = (rank + step) % nRanks;
            int recv_peer = (rank + nRanks - step) % nRanks;
            gatherSink = reinterpret_cast<ringPtr>((uintptr_t)gatherSinkArray[send_peer]);

            auto* ptr = ingress + offset;
            if (has_offsets)
                ptr = (T*)((char*)ptr + offsets[send_peer]);
            else
                ptr = ptr + send_peer * workElems;
            ProtoT::loadInput(v, ptr, nelems, gr);
            ProtoT::shuffleData(v);
            ProtoT::insertFlags(v, flag);
            ProtoT::sendMessages(gatherSink[rank][slot][wireId], v);

            ProtoT::sbarrier_signal_compat(p2p);

            // recv data
            ProtoT::sbarrier_wait_compat(p2p);
            bool retry;
            do {
                retry = false;
                retry |= ProtoT::recvMessages(v, localGatherSink[recv_peer][slot][wireId], flag);
            } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

            ProtoT::restoreData(v);

            ptr = egress + offset;
            if (has_offsets)
                ptr = (T*)((char*)ptr + offsets[recv_peer]);
            else
                ptr = ptr + recv_peer * workElems;
            ProtoT::storeOutput(ptr, v, nelems, gr);
        }
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

    void init_ring_buffers(T* input,
                           T* output,
                           const size_t* offs,
                           T* scatterBuf,
                           T* gatherBuf,
                           T* const peerBuf0[],
                           T* const peerBuf1[],
                           int nranks) {
        auto next = (rank + 1) % nRanks;
        ingress = input;
        egress = output;

        has_offsets = false;
        if (offs) {
            has_offsets = true;
            for (int i = 0; i < nranks; i++)
                offsets[i] = offs[i];
        }

        if (peerBuf0 == nullptr) {
            for (int i = 0; i < nranks; i++)
                gatherSinkArray[i] = reinterpret_cast<ringPtr>((uintptr_t)peerBuf1[i]);
            localGatherSink = reinterpret_cast<ringPtr>((uintptr_t)gatherBuf);
        }
        else {
            scatterSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf0[next]);
            gatherSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf1[next]);

            localScatterSink = reinterpret_cast<ringPtr>((uintptr_t)scatterBuf);
            localGatherSink = reinterpret_cast<ringPtr>((uintptr_t)gatherBuf);
        }
    }

    T* ingress;
    T* egress;
    size_t offsets[ARC_MAX_NUM];
    bool has_offsets;

    int nRanks;
    ssize_t workElems;
    int rank;
    uint32_t* seqNoPtr;
    uint32_t seqNo_;
    bool p2p;
    ccl::reduction reduction;

    // Effective reduction is reduction performed
    // in functions as ProtoT::loadRecvReduceSendWrtback
    // to perform in flight simple reduction operations
    ccl::reduction effective_reduction;

    // Stores data required to perform complex reductions
    // such as average or premul sum
    ccl_reduction_data reduction_data;

    ringPtr scatterSink;
    ringPtr gatherSink;

    ringPtr localScatterSink;
    ringPtr localGatherSink;

    ringPtr gatherSinkArray[ARC_MAX_NUM];
};
