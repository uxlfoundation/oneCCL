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

// sub-group level one-shot transmit, minimum latency

// set the maximum message size to 16 KB for one-shot transmit
#define ONESHOT_MAX_MSGSIZE 16384

template <typename T,
          template <typename, int>
          class Proto,
          bool LoadSeqNoInRuntime,
          int SubGroupSize = 16>
class OneShotTransmit {
protected:
    using ProtoT = Proto<T, SubGroupSize>;
    using message_t = typename ProtoT::message_t;

public:
    const int parallel_sg;
    constexpr static size_t nSlot = 8;
    constexpr static size_t ringSize = ProtoT::wireTransSize * nSlot;
    constexpr static size_t maxLaunch = 1;
    // Per-collective data split. Allreduce broadcasts the full buffer to every
    // wire (FullBufferEach); reduce-scatter partitions it across wires (NoOverlap).
    static SplitType splitType(TransmitColl coll) {
        switch (coll) {
            case TransmitColl::ReduceScatter:
            case TransmitColl::Allgather:
                return SplitType::NoOverlap;
            case TransmitColl::Allreduce:
            case TransmitColl::Reduce:
                return SplitType::FullBufferEach;
            default:
                CCL_THROW("OneShotTransmit::splitType: unhandled TransmitColl");
        }
    }
    typedef T (*oneshotPtr)[nSlot][ProtoT::wireTransElems];

    static size_t calcLoopSize(size_t nWires, int parallel_sg, size_t ipcbuf_size) {
        size_t loopSize = nWires * ProtoT::wireCapacity / parallel_sg;
        size_t bufCap = ipcbuf_size / (parallel_sg * ringSize) * ProtoT::wireCapacity;
        if (bufCap < loopSize)
            loopSize = bufCap;
        return loopSize;
    }

public:
    OneShotTransmit(int nranks,
                    T* input,
                    T* output,
                    const size_t* offs, // offset is not supported currently
                    T* scatterBuf,
                    T* gatherBuf,
                    T* const peerBuf0[],
                    T* const peerBuf1[],
                    ssize_t workSize,
                    int rank,
                    uint32_t* seqNoPtr, // Serves as a flag for checking
                    uint32_t seqNo, // Serves as a flag for checking
                    ccl::reduction reduction,
                    bool p2p)
            : nranks(nranks),
              parallel_sg(nranks),
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
        }
        else {
            effective_reduction = reduction;
        }
        ingress = input;
        egress = output;
        has_offsets = offs != nullptr;
        if (has_offsets) {
            for (int i = 0; i < nranks; ++i)
                offsets[i] = offs[i];
        }
        // See memory layout diagram in runAllreduce()
        for (int i = 0; i < nranks; ++i)
            scatterSinks[i] =
                reinterpret_cast<oneshotPtr>((uintptr_t)peerBuf0[i] + rank * ringSize);

        // Indexed by the writing peer, not by distance from us, so the reduce loops
        // can fold in ascending rank order. Entry [rank] is our own input, unused.
        for (int p = 0; p < nranks; ++p) {
            if (p == rank)
                continue;

            localScatterSinks[p] =
                reinterpret_cast<oneshotPtr>((uintptr_t)scatterBuf + p * ringSize);
        }

        // Gather sinks mirror the allgather (broadcast) phase of the two-shots
        // allreduce: the owner of a chunk writes its data into every peer's gather
        // buffer at the owner's rank offset; each rank receives into its own
        // gatherBuf (localGatherSink). See runAllgather().
        localGatherSink = reinterpret_cast<oneshotPtr>(gatherBuf);
        for (int i = 0; i < nranks - 1; ++i) {
            int next = (rank + i + 1) % nranks;

            gatherSinks[i] =
                reinterpret_cast<oneshotPtr>((uintptr_t)peerBuf1[next] + rank * ringSize);
        }
    }

    // Allgather-shaped constructor (no reduction): mirrors RingTransmit's
    // output+offsets overload so OneShotTransmit can drive the allgather path.
    // The reduction is irrelevant for allgather, so it delegates with sum.
    OneShotTransmit(int nranks,
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
            : OneShotTransmit(nranks,
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

    inline void printData(const char* c_str, message_t v) {
        sycl::ext::oneapi::experimental::printf(
            "%s [rank %d global_id %d local_id %d group_id %d subgroup_id %d subgroup_local_id %d]: %x,%x,%x,%x\n",
            c_str,
            rank,
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0),
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_local_id(0),
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_group(0),
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_sub_group().get_group_id()[0],
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_sub_group().get_local_id()[0],
            v[0],
            v[1],
            v[2],
            v[3]);
        return;
    }

    inline void runAllreduce(size_t inputOffset,
                             size_t iter,
                             ssize_t workLeft,
                             WorkSizeData<message_t> workSizeData,
                             AccessGranularity gr) {
        if (workLeft <= 0) {
            return;
        }

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto y_id = wireId % nranks; // represents the rank that the current wire is going to handle
        auto x_id = wireId / nranks * nranks; // cableId * nranks

        auto inputOffInType = inputOffset / sizeof(T);
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo + iter / nSlot;
        size_t slot = (seqNo + iter) % nSlot;
        auto nelems = workSizeData.GetNElems(y_id) - inputOffInType;

        const int unroll = 1;
        constexpr auto eltPerPack = unroll * ProtoT::wireCapacityInType;
        constexpr bool accumulateShuffledData = sizeof(T) <= sizeof(seqNo);

        message_t v[unroll];

        auto* ptr = ingress + inputOffInType + workSizeData.GetOffset(y_id);
        auto* o_ptr = egress + inputOffInType + workSizeData.GetOffset(y_id);

        ProtoT::loadInput(v[0], ptr, nelems, gr);
        ProtoT::applyPreOpMessages(v[0], reduction_data);

        if (y_id != rank) { // bcast
            ProtoT::shuffleData(v[0]);
            ProtoT::insertFlags(v[0], flag);

            /*
            * Memory layout
            * each - represents one block of (nSlot * ProtoT::wireTransElems * sizeof(T)) bytes (2048 bytes for Rt64_PCIE):
            * Suppose there are 4 ranks in total.
            * wireId = 0,1,2,3 => x_id = 0
            * On rank 0:
            * peerBuf0[0] ----------------......------
            *             ^ scatterSinks[0][0] points here
            * peerBuf0[1] ----------------......------
            *             ^ scatterSinks[1][0] points here
            * peerBuf0[2] ----------------......------
            *             ^ scatterSinks[2][0] points here
            * peerBuf0[3] ----------------......------ 
            *             ^ scatterSinks[3][0] points here
            * On rank 1:
            * peerBuf0[0] ----------------......------
            *              ^ scatterSinks[0][0] points here
            * peerBuf0[1] ----------------......------
            *              ^ scatterSinks[1][0] points here
            * peerBuf0[2] ----------------......------
            *              ^ scatterSinks[2][0] points here
            * peerBuf0[3] ----------------......------ 
            *              ^ scatterSinks[3][0] points here
            * On rank 2:
            * peerBuf0[0] ----------------......------
            *               ^ scatterSinks[0][0] points here
            * peerBuf0[1] ----------------......------
            *               ^ scatterSinks[1][0] points here
            * peerBuf0[2] ----------------......------
            *               ^ scatterSinks[2][0] points here
            * peerBuf0[3] ----------------......------
            *               ^ scatterSinks[3][0] points here
            * On rank 3:
            * peerBuf0[0] ----------------......------
            *                ^ scatterSinks[0][0] points here
            * peerBuf0[1] ----------------......------
            *                ^ scatterSinks[1][0] points here
            * peerBuf0[2] ----------------......------
            *                ^ scatterSinks[2][0] points here
            * peerBuf0[3] ----------------......------ 
            *                ^ scatterSinks[3][0] points here
            * wireId = 4,5,6,7 => x_id = 4
            * peerBuf0[0] ----------------......------
            *                 ^ scatterSinks[0][1] points here
            * peerBuf0[1] ----------------......------
            *                  ^ scatterSinks[1][1] points here
            * peerBuf0[2] ----------------......------
            *                  ^ scatterSinks[2][1] points here
            * peerBuf0[3] ----------------......------ 
            *                  ^ scatterSinks[3][1] points here
            * On rank 1:
            * peerBuf0[0] ----------------......------
            *                   ^ scatterSinks[0][1] points here
            * peerBuf0[1] ----------------......------
            *                   ^ scatterSinks[1][1] points here
            * peerBuf0[2] ----------------......------
            *                   ^ scatterSinks[2][1] points here
            * peerBuf0[3] ----------------......------ 
            *                   ^ scatterSinks[3][1] points here
            * On rank 2:
            * peerBuf0[0] ----------------......------
            *                    ^ scatterSinks[0][1] points here
            * peerBuf0[1] ----------------......------
            *                    ^ scatterSinks[1][1] points here
            * peerBuf0[2] ----------------......------
            *                    ^ scatterSinks[2][1] points here
            * peerBuf0[3] ----------------......------
            *                    ^ scatterSinks[3][1] points here
            * On rank 3:
            * peerBuf0[0] ----------------......------
            *                     ^ scatterSinks[0][1] points here
            * peerBuf0[1] ----------------......------
            *                     ^ scatterSinks[1][1] points here
            * peerBuf0[2] ----------------......------
            *                     ^ scatterSinks[2][1] points here
            * peerBuf0[3] ----------------......------ 
            *                     ^ scatterSinks[3][1] points here
            * 
            */
            ProtoT::sendMessages(scatterSinks[y_id][x_id][slot], v[0]);
        }

        if (y_id == rank) { // reduce
            if constexpr (accumulateShuffledData) {
                ProtoT::shuffleData(v[0]);
            }
            // Fold in ascending rank order, ours spliced in at position rank. Float
            // addition is not associative, so a rank-dependent order would give every
            // rank a different sum here and break DDP/FSDP replica sync.
            message_t messages[unroll];
            message_t acc{};
            for (int p = 0; p < nranks; ++p) {
                if (p != rank) {
                    bool retry;
                    do {
                        retry = false;
                        retry |= ProtoT::recvMessages(
                            messages[0], localScatterSinks[p][x_id][slot], flag);
                    } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(),
                                                retry));

                    if constexpr (!accumulateShuffledData) {
                        ProtoT::restoreData(messages[0]);
                    }
                }
                message_t& contribution = (p == rank) ? v[0] : messages[0];
                if (p == 0) {
                    acc = contribution;
                }
                else {
                    ProtoT::accumMessages(acc, contribution, effective_reduction);
                }
            }
            if constexpr (accumulateShuffledData) {
                ProtoT::restoreData(acc);
            }

            ProtoT::storeOutput(o_ptr, acc, nelems, gr);
        }
    }

    // One-shot reduce-scatter: every wire sends its slice to the owner of the
    // chunk it handles; the owner reduces all peers' slices into its output.
    // Mirrors the scatter+reduce pattern of two-shots allreduce, but two-shots
    // uses its own inlined runAllreduce; this path serves reduce_scatter only.
    inline void runReduceScatter(size_t inputOffset,
                                 size_t iter,
                                 ssize_t workLeft,
                                 ssize_t /*workSize*/,
                                 AccessGranularity gr) {
        if (workLeft <= 0) {
            return;
        }

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto y_id = wireId % nranks; // the rank that owns the chunk this wire handles
        auto x_id = wireId / nranks * nranks; // cableId * nranks

        auto inputOffInType = inputOffset / sizeof(T); // position within a chunk
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo + iter / nSlot;
        size_t slot = (seqNo + iter) % nSlot;
        auto nelems = workElems - inputOffInType;

        const int unroll = 1;
        constexpr bool accumulateShuffledData = sizeof(T) <= sizeof(seqNo);

        message_t v[unroll];

        // Slice of the local input destined for rank y_id (chunk y_id).
        auto* ptr = ingress + (size_t)y_id * workElems + inputOffInType;
        // Reduce-scatter output keeps only this rank's chunk (offset 0).
        auto* o_ptr = egress + inputOffInType;

        ProtoT::loadInput(v[0], ptr, nelems, gr);
        ProtoT::applyPreOpMessages(v[0], reduction_data);

        // scatter: send our slice for rank y_id to its owner's scatter buffer.
        // (Two-shots also waits here for the gathered result; reduce-scatter does not.)
        if (y_id != rank) {
            ProtoT::shuffleData(v[0]);
            ProtoT::insertFlags(v[0], flag);
            ProtoT::sendMessages(scatterSinks[y_id][x_id][slot], v[0]);
        }

        // reduce: owner of the local chunk reduces all peers' slices.
        // (Two-shots then broadcasts the result back; reduce-scatter stops here
        //  and writes the reduced chunk straight to the local output.)
        if (y_id == rank) {
            if constexpr (accumulateShuffledData) {
                ProtoT::shuffleData(v[0]);
            }
            // Ascending rank order, as in runAllreduce.
            message_t messages[unroll];
            message_t acc{};
            for (int p = 0; p < nranks; ++p) {
                if (p != rank) {
                    bool retry;
                    do {
                        retry = false;
                        retry |= ProtoT::recvMessages(
                            messages[0], localScatterSinks[p][x_id][slot], flag);
                    } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(),
                                                retry));

                    if constexpr (!accumulateShuffledData) {
                        ProtoT::restoreData(messages[0]);
                    }
                }
                message_t& contribution = (p == rank) ? v[0] : messages[0];
                if (p == 0) {
                    acc = contribution;
                }
                else {
                    ProtoT::accumMessages(acc, contribution, effective_reduction);
                }
            }
            if constexpr (accumulateShuffledData) {
                ProtoT::restoreData(acc);
            }

            ProtoT::storeOutput(o_ptr, acc, nelems, gr);
        }
    }

    // One-shot allgather: every rank owns one chunk; the owner broadcasts it to
    // all peers while each peer receives the chunks it does not own. No reduction.
    // Mirrors the broadcast pattern of two-shots allreduce, but two-shots uses its
    // own inlined runAllreduce; this path serves allgatherv only.
    inline void runAllgather(size_t inputOffset,
                             size_t iter,
                             ssize_t workLeft,
                             AccessGranularity gr) {
        if (workLeft <= 0) {
            return;
        }

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto y_id = wireId % nranks; // the rank that owns the chunk this wire handles
        auto x_id = wireId / nranks * nranks; // cableId * nranks

        auto inputOffInType = inputOffset / sizeof(T); // position within a chunk
        uint32_t seqNo = getSeqNo();
        auto flag = seqNo + iter / nSlot;
        size_t slot = (seqNo + iter) % nSlot;
        auto nelems = workElems - inputOffInType;

        const int unroll = 1;

        message_t v[unroll];

        // Destination of chunk y_id within the gathered output buffer.
        T* o_ptr;
        if (has_offsets) {
            o_ptr = reinterpret_cast<T*>((char*)egress + offsets[y_id]) + inputOffInType;
        }
        else {
            o_ptr = egress + (size_t)y_id * workElems + inputOffInType;
        }

        // owner of chunk y_id: load its local slice and broadcast to all peers.
        if (y_id == rank) {
            auto* ptr = ingress + inputOffInType;
            ProtoT::loadInput(v[0], ptr, nelems, gr);
            ProtoT::shuffleData(v[0]);
            ProtoT::insertFlags(v[0], flag);
            for (int i = 0; i < nranks - 1; ++i) {
                ProtoT::sendMessages(gatherSinks[i][x_id][slot], v[0]);
            }
        }

        // non-owner: receive chunk y_id from its owner's broadcast.
        if (y_id != rank) {
            bool retry;
            do {
                retry = false;
                retry |= ProtoT::recvMessages(v[0], localGatherSink[wireId][slot], flag);
            } while (
                sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));
        }

        ProtoT::restoreData(v[0]);
        ProtoT::storeOutput(o_ptr, v[0], nelems, gr);
    }

protected:
    inline int getSeqNo() {
        if (LoadSeqNoInRuntime) {
            return *seqNoPtr;
        }
        else {
            return seqNo_;
        }
    }

    T* ingress;
    T* egress;

    int nranks;
    ssize_t workElems;
    int rank;
    uint32_t* seqNoPtr;
    uint32_t seqNo_;
    bool p2p;
    ccl::reduction reduction;

    // Effective reduction is the simple reduction op performed in-flight by ProtoT::accumMessages.
    // Custom reductions fall back to sum and are completed separately via reduction_data /
    // applyPreOpMessages, which applies the per-rank scalar multiply on the loaded input.
    ccl::reduction effective_reduction;

    // Stores data required to perform complex reductions such as average or pre-mul-sum.
    ccl_reduction_data reduction_data;

    oneshotPtr scatterSinks[ARC_MAX_NUM];
    oneshotPtr localScatterSinks[ARC_MAX_NUM];

    // Used by the allgather (broadcast) phase.
    oneshotPtr gatherSinks[ARC_MAX_NUM];
    oneshotPtr localGatherSink;
    size_t offsets[ARC_MAX_NUM];
    bool has_offsets;
};
