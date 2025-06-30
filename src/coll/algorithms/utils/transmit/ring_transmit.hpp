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

//
// For those do not have sub-group level independent forward progress
// or PCIE connection without switch (remote polling).
//
//
// When split barrier is not supported, signal become null,
// wait will be both signal and wait.
static inline void sbarrier_signal_compat(bool p2p) {
#if defined(CCL_SYCL_ENABLE_ARCB) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    if (!p2p)
        sbarrier_signal();
#endif
}

static inline void sbarrier_wait_compat(bool p2p) {
#if defined(CCL_SYCL_ENABLE_ARCB) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    if (!p2p)
        sbarrier_wait();
#endif
}

template <typename T, int NRanks, template <typename, int> class Proto, int SubGroupSize = 16>
class RingTransmit : public Proto<T, SubGroupSize> {
protected:
    static constexpr int parallel_sg = 1;
    using ProtoT = Proto<T, SubGroupSize>;

    using typename ProtoT::message_t;
    using ProtoT::wireCapacityInType;

    using ProtoT::wireTransSize;
    using ProtoT::wireTransElems;

    using ProtoT::loadInput;
    using ProtoT::shuffleData;
    using ProtoT::insertFlags;
    using ProtoT::sendMessages;
    using ProtoT::recvMessages;
    using ProtoT::accumMessages;
    using ProtoT::restoreData;
    using ProtoT::storeOutput;
    using ProtoT::wireCapacity;

public:
    constexpr static size_t nSlot = 4;
#if defined(CCL_SYCL_ENABLE_ARCB)
    constexpr static size_t maxLaunch = 64 * 20;
#else
    constexpr static size_t maxLaunch = 64 * 64;
#endif
    constexpr static size_t ringSize = maxLaunch * wireTransSize * nSlot;
    static_assert(ringSize <= 4 * 1024 * 1024ull * SubGroupSize / 16);

    typedef T (*ringPtr)[nSlot][maxLaunch][wireTransElems];

public:
    RingTransmit(T* input,
                 T* scatterBuf,
                 T* gatherBuf,
                 T* const peerBuf0[],
                 T* const peerBuf1[],
                 ssize_t workSize,
                 int rank,
                 uint32_t seqNo, // Serve as flag for checking
                 bool p2p)
            : workElems(workSize / sizeof(T)),
              rank(rank),
              seqNo(seqNo),
              p2p(p2p) {
        auto next = (rank + 1) % NRanks;
        ingress = input;
        egress = input;

        scatterSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf0[next]);
        gatherSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf1[next]);

        localScatterSink = reinterpret_cast<ringPtr>((uintptr_t)scatterBuf);
        localGatherSink = reinterpret_cast<ringPtr>((uintptr_t)gatherBuf);
    }

    RingTransmit(T* input,
                 T* output,
                 T* scatterBuf,
                 T* gatherBuf,
                 T* const peerBuf0[],
                 T* const peerBuf1[],
                 ssize_t workSize,
                 int rank,
                 uint32_t seqNo, // Serve as flag for checking
                 bool p2p)
            : workElems(workSize / sizeof(T)),
              rank(rank),
              seqNo(seqNo),
              p2p(p2p) {
        auto next = (rank + 1) % NRanks;
        ingress = input;
        egress = output;

        scatterSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf0[next]);
        gatherSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf1[next]);

        localScatterSink = reinterpret_cast<ringPtr>((uintptr_t)scatterBuf);
        localGatherSink = reinterpret_cast<ringPtr>((uintptr_t)gatherBuf);
    }

    template <int __dummy__>
    inline void run(size_t inputOffset, size_t tStep, ssize_t workLeft) {
        runAllreduce(inputOffset, tStep, workLeft);
    }

    inline void send(int wireId,
                     int peer,
                     size_t offset,
                     uint32_t flag,
                     uint32_t slot,
                     ssize_t nelems) {
        message_t v;
        auto* ptr = ingress + peer * workElems + offset;
        loadInput(v, ptr, nelems);

        shuffleData(v);
        insertFlags(v, flag);

        sendMessages(scatterSink[peer][slot][wireId], v);
        sbarrier_signal_compat(p2p);
    }

    inline void loadRecvReduceSend(int wireId,
                                   int peer,
                                   size_t offset,
                                   uint32_t flag,
                                   uint32_t slot,
                                   ssize_t nelems) {
        message_t v;
        message_t messages;

        auto* ptr = ingress + peer * workElems + offset;
        loadInput(v, ptr, nelems);

        bool retry;
        sbarrier_wait_compat(p2p);
        do {
            retry = false;
            retry |= recvMessages(messages, localScatterSink[peer][slot][wireId], flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

        shuffleData(v);
        accumMessages(v, messages);
        insertFlags(v, flag);

        sendMessages(scatterSink[peer][slot][wireId], v);
        sbarrier_signal_compat(p2p);
    }

    inline void loadRecvReduceSendWrtback(int wireId,
                                          int peer,
                                          size_t offset,
                                          uint32_t flag,
                                          uint32_t slot,
                                          ssize_t nelems) {
        message_t v;
        message_t messages;

        auto* ptr = ingress + peer * workElems + offset;
        loadInput(v, ptr, nelems);

        bool retry;
        sbarrier_wait_compat(p2p);
        do {
            retry = false;
            retry |= recvMessages(messages, localScatterSink[peer][slot][wireId], flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

        shuffleData(v);
        accumMessages(v, messages);

        insertFlags(v, flag);
        sendMessages(gatherSink[peer][slot][wireId], v);
        sbarrier_signal_compat(p2p);

        restoreData(v);

        ptr = egress + peer * workElems + offset;
        storeOutput(ptr, v, nelems);
    }

    inline void recvSendWrtback(int wireId,
                                int peer,
                                size_t offset,
                                uint32_t flag,
                                uint32_t slot,
                                ssize_t nelems) {
        message_t v;

        bool retry;
        sbarrier_wait_compat(p2p);
        do {
            retry = false;
            retry |= recvMessages(v, localGatherSink[peer][slot][wireId], flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

        insertFlags(v, flag);
        sendMessages(gatherSink[peer][slot][wireId], v);
        sbarrier_signal_compat(p2p);

        restoreData(v);

        auto* ptr = egress + peer * workElems + offset;
        storeOutput(ptr, v, nelems);
    }

    inline void recvWrtback(int wireId,
                            int peer,
                            size_t offset,
                            uint32_t flag,
                            uint32_t slot,
                            ssize_t nelems) {
        message_t v;

        sbarrier_wait_compat(p2p);
        bool retry;
        do {
            retry = false;
            retry |= recvMessages(v, localGatherSink[peer][slot][wireId], flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

        restoreData(v);

        auto* ptr = egress + peer * workElems + offset;
        storeOutput(ptr, v, nelems);
    }

    inline void runAllreduce(size_t inputOffset, size_t tStep, ssize_t workLeft) {
        if (workLeft <= 0)
            return;

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto offset = inputOffset / sizeof(T);
        auto flag = seqNo + tStep / nSlot;
        auto slot = (seqNo + tStep) % nSlot;
        auto nelems = workLeft / sizeof(T);

        uint32_t p_idx = 0;
        int peer = (rank + p_idx) % NRanks;

        // Step 0
        send(wireId, peer, offset, flag, slot, nelems);

        // Step 1 to N-1
#pragma unroll
        for (int i = 1; i < NRanks - 1; ++i) {
            p_idx = (p_idx - 1) % NRanks;
            peer = (rank + p_idx) % NRanks;
            loadRecvReduceSend(wireId, peer, offset, flag, slot, nelems);
        }

        // Step N
        p_idx = (p_idx - 1) % NRanks;
        peer = (rank + p_idx) % NRanks;
        loadRecvReduceSendWrtback(wireId, peer, offset, flag, slot, nelems);

        // write back
#pragma unroll
        for (uint32_t i = 1; i < NRanks - 1; ++i) {
            p_idx = (p_idx - 1) % NRanks; // 0
            peer = (rank + p_idx) % NRanks;
            recvSendWrtback(wireId, peer, offset, flag, slot, nelems);
        }

        p_idx = (p_idx - 1) % NRanks;
        peer = (rank + p_idx) % NRanks;
        recvWrtback(wireId, peer, offset, flag, slot, nelems);
    }

    inline void runAllgather(size_t inputOffset, size_t tStep, ssize_t workLeft) {
        if (workLeft <= 0)
            return;

        auto wireId =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0) / SubGroupSize;

        auto inputOffInType = inputOffset / sizeof(T);
        auto flag = seqNo + tStep / nSlot;
        auto slot = (seqNo + tStep) % nSlot;
        auto nelems = workLeft / sizeof(T);

        message_t v;

        uint32_t p_idx = 0;
        int peer = (rank + p_idx) % NRanks;

        auto* ptr = ingress + inputOffInType;
        auto* o_ptr = egress + peer * workElems + inputOffInType;
        loadInput(v, ptr, nelems);

        if (ptr != o_ptr)
            storeOutput(o_ptr, v, nelems);

        shuffleData(v);
        insertFlags(v, flag);
        sendMessages(gatherSink[peer][slot][wireId], v);

        sbarrier_signal_compat(p2p);

#pragma unroll
        for (uint32_t i = 1; i < NRanks - 1; ++i) {
            p_idx = (p_idx - 1) % NRanks; // 0
            peer = (rank + p_idx) % NRanks;
            recvSendWrtback(wireId, peer, inputOffInType, flag, slot, nelems);
        }

        p_idx = (p_idx - 1) % NRanks;
        peer = (rank + p_idx) % NRanks;

        recvWrtback(wireId, peer, inputOffInType, flag, slot, nelems);
    }

protected:
    T* ingress;
    T* egress;

    ssize_t workElems;
    int rank;
    uint32_t seqNo;
    bool p2p;

    ringPtr scatterSink;
    ringPtr gatherSink;

    ringPtr localScatterSink;
    ringPtr localGatherSink;
};
