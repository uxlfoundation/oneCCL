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
#include "coll/algorithms/utils/tvisa/include/gen_visa_templates.hpp"
#include "coll/reduction/reduction.hpp"

template <typename T, int SubGroupSize, typename MessageT, typename Derived>
struct ProtoBase {
    // Use derived class's message_t
    using message_t = MessageT;
    template <int unroll>
    static inline void applyPreOpMessages(message_t (&v)[unroll],
                                          const ccl_reduction_data &reduction_data) {
        if (reduction_data.op_type == ccl_reduction_type_internal::ccl_pre_mul_sum) {
            using math_t = sycl::vec<T, sizeof(message_t) / sizeof(T)>;
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                auto a = sycl::bit_cast<math_t>(v[i]);
                v[i] = sycl::bit_cast<message_t>(apply_pre_operation(reduction_data, a));
            }
        }
    }
    static inline void applyPreOpMessages(message_t &v, const ccl_reduction_data &reduction_data) {
        if (reduction_data.op_type == ccl_reduction_type_internal::ccl_pre_mul_sum) {
            using math_t = sycl::vec<T, sizeof(message_t) / sizeof(T)>;
            auto a = sycl::bit_cast<math_t>(v);
            v = sycl::bit_cast<message_t>(apply_pre_operation(reduction_data, a));
        }
    }

    template <int unroll>
    static inline void loadInput(message_t (&v)[unroll],
                                 T *src,
                                 ssize_t nElt,
                                 AccessGranularity gr) {
        Derived::loadInput<unroll>(v, src, nElt, gr);
    }
    static inline void loadInput(message_t &v, T *src, ssize_t nElt, AccessGranularity gr) {
        Derived::loadInput(v, src, nElt, gr);
    }
    template <int unroll>
    static inline void loadInput(message_t (&v)[unroll], T *src, AccessGranularity gr) {
        Derived::loadInput<unroll>(v, src, gr);
    }
    static inline void loadInput(message_t &v, T *src, AccessGranularity gr) {
        Derived::loadInput(v, src, gr);
    }

    template <int unroll>
    static inline void insertFlags(message_t (&messages)[unroll], uint32_t flag) {
        Derived::insertFlags<unroll>(messages, flag);
    }
    static inline void insertFlags(message_t &messages, uint32_t flag) {
        Derived::insertFlags(messages, flag);
    }

    template <int unroll>
    static inline void shuffleData(message_t (&messages)[unroll]) {
        Derived::shuffleData<unroll>(messages);
    }
    static inline void shuffleData(message_t &messages) {
        Derived::shuffleData(messages);
    }

    template <int unroll>
    static inline void restoreData(message_t (&messages)[unroll]) {
        Derived::restoreData<unroll>(messages);
    }
    static inline void restoreData(message_t &messages) {
        Derived::restoreData(messages);
    }

    template <int unroll>
    static inline void storeOutput(T *dst, message_t (&v)[unroll], AccessGranularity gr) {
        Derived::storeOutput<unroll>(dst, v, gr);
    }
    static inline void storeOutput(T *dst, message_t &v, AccessGranularity gr) {
        Derived::storeOutput(dst, v, gr);
    }
    template <int unroll>
    static inline void storeOutput(T *dst,
                                   message_t (&v)[unroll],
                                   ssize_t nElt,
                                   AccessGranularity gr) {
        Derived::storeOutput<unroll>(dst, v, nElt, gr);
    }
    static inline void storeOutput(T *dst, message_t &v, ssize_t nElt, AccessGranularity gr) {
        Derived::storeOutput(dst, v, nElt, gr);
    }

    template <int unroll>
    static inline void sendMessages(T *ptr, message_t (&messages)[unroll]) {
        Derived::sendMessages<unroll>(ptr, messages);
    }
    static inline void sendMessages(T *ptr, message_t &messages) {
        Derived::sendMessages(ptr, messages);
    }

    template <int unroll>
    static inline bool recvMessages(message_t (&messages)[unroll], T *ptr, uint32_t flag) {
        return Derived::recvMessages<unroll>(messages, ptr, flag);
    }
    static inline bool recvMessages(message_t &messages, T *ptr, uint32_t flag) {
        return Derived::recvMessages(messages, ptr, flag);
    }

    template <int unroll>
    static inline void accumMessages(message_t (&v)[unroll], message_t (&m)[unroll]) {
        Derived::accumMessages<unroll>(v, m);
    }
    static inline void accumMessages(message_t &v, message_t &m) {
        Derived::accumMessages(v, m);
    }

    template <int unroll>
    static inline void accumMessages(message_t (&v)[unroll],
                                     message_t (&m)[unroll],
                                     ccl::reduction reduction) {
        Derived::accumMessages<unroll>(v, m, reduction);
    }
    static inline void accumMessages(message_t &v, message_t &m, ccl::reduction reduction) {
        Derived::accumMessages(v, m, reduction);
    }

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

    static inline void send(T *inbuf,
                            T *remotebuf,
                            int wireId,
                            int peer,
                            ssize_t workElems,
                            size_t offset,
                            uint32_t flag,
                            uint32_t slot,
                            ssize_t nelems,
                            const ccl_reduction_data &reduction_data,
                            bool p2p,
                            AccessGranularity gr) {
        message_t v;
        auto *ptr = inbuf + peer * workElems + offset;
        loadInput(v, ptr, nelems, gr);
        /* reduction_data is only populated (via make_reduction_operation)
          when ccl_reduction_type_storage::is_custom(reduction) is true,
          i.e. for user-defined reductions like pre_mul_sum. In that case
          applyPreOpMessages applies the per-rank scalar multiply. */
        applyPreOpMessages(v, reduction_data);

        shuffleData(v);
        insertFlags(v, flag);

        sendMessages(remotebuf, v);
        sbarrier_signal_compat(p2p);
    }

    static inline void loadRecvReduceSend(T *inbuf,
                                          T *recvbuf,
                                          T *remotebuf,
                                          int wireId,
                                          int peer,
                                          ssize_t workElems,
                                          size_t offset,
                                          uint32_t flag,
                                          uint32_t slot,
                                          ssize_t nelems,
                                          ccl::reduction reduction,
                                          const ccl_reduction_data &reduction_data,
                                          bool p2p,
                                          AccessGranularity gr = AccessGranularity::Vector) {
        message_t v;
        message_t messages;

        auto *ptr = inbuf + peer * workElems + offset;
        loadInput(v, ptr, nelems, gr);
        applyPreOpMessages(v, reduction_data);

        bool retry;
        sbarrier_wait_compat(p2p);
        do {
            retry = false;
            retry |= recvMessages(messages, recvbuf, flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

        if constexpr (sizeof(T) > sizeof(flag)) {
            // restore data, accumulate, shuffle
            restoreData(messages);
            accumMessages(v, messages, reduction);
            shuffleData(v);
        }
        else {
            // datasize is smaller than flag,
            // no overlap between datatype and flag
            // can accumulate shuffled data:
            // less ops to perform, faster
            shuffleData(v);
            accumMessages(v, messages, reduction);
        }

        insertFlags(v, flag);

        sendMessages(remotebuf, v);
        sbarrier_signal_compat(p2p);
    }

    static inline void loadRecvReduceSendWrtback(T *inbuf,
                                                 T *recvbuf,
                                                 T *remotebuf,
                                                 T *outbuf,
                                                 int wireId,
                                                 int peer,
                                                 ssize_t workElems,
                                                 size_t offset,
                                                 uint32_t flag,
                                                 uint32_t slot,
                                                 ssize_t nelems,
                                                 ccl::reduction reduction,
                                                 const ccl_reduction_data &reduction_data,
                                                 bool p2p,
                                                 AccessGranularity gr = AccessGranularity::Vector) {
        message_t v;
        message_t messages;

        auto *ptr = inbuf + peer * workElems + offset;
        loadInput(v, ptr, nelems, gr);
        applyPreOpMessages(v, reduction_data);

        bool retry;
        sbarrier_wait_compat(p2p);
        do {
            retry = false;
            retry |= recvMessages(messages, recvbuf, flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

        if constexpr (sizeof(T) > sizeof(flag)) {
            // restore data, accumulate, shuffle
            restoreData(messages);
            accumMessages(v, messages, reduction);
            shuffleData(v);
        }
        else {
            // datasize is smaller than flag,
            // no overlap between datatype and flag
            // can accumulate shuffled data:
            // less ops to perform, faster
            shuffleData(v);
            accumMessages(v, messages, reduction);
        }

        insertFlags(v, flag);
        sendMessages(remotebuf, v);
        sbarrier_signal_compat(p2p);

        restoreData(v);

        ptr = outbuf + peer * workElems + offset;
        storeOutput(ptr, v, nelems, gr);
    }

    static inline void recvSendWrtback(T *recvbuf,
                                       T *remotebuf,
                                       T *outbuf,
                                       size_t *offsets,
                                       int wireId,
                                       int peer,
                                       ssize_t workElems,
                                       size_t offset,
                                       uint32_t flag,
                                       uint32_t slot,
                                       ssize_t nelems,
                                       bool p2p,
                                       AccessGranularity gr) {
        message_t v;

        bool retry;
        sbarrier_wait_compat(p2p);
        do {
            retry = false;
            retry |= recvMessages(v, recvbuf, flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

        insertFlags(v, flag);
        sendMessages(remotebuf, v);
        sbarrier_signal_compat(p2p);

        restoreData(v);

        auto *ptr = outbuf + offset;
        if (offsets)
            ptr = (T *)((char *)ptr + offsets[peer]);
        else
            ptr = ptr + peer * workElems;
        storeOutput(ptr, v, nelems, gr);
    }

    static inline void recvWrtback(T *recvbuf,
                                   T *outbuf,
                                   size_t *offsets,
                                   int wireId,
                                   int peer,
                                   ssize_t workElems,
                                   size_t offset,
                                   uint32_t flag,
                                   uint32_t slot,
                                   ssize_t nelems,
                                   bool p2p,
                                   AccessGranularity gr) {
        message_t v;

        sbarrier_wait_compat(p2p);
        bool retry;
        do {
            retry = false;
            retry |= recvMessages(v, recvbuf, flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

        restoreData(v);

        auto *ptr = outbuf + offset;
        if (offsets)
            ptr = (T *)((char *)ptr + offsets[peer]);
        else
            ptr = ptr + peer * workElems;
        storeOutput(ptr, v, nelems, gr);
    }

    static inline void loadRecvReduceWrtback(T *inbuf,
                                             T *recvbuf,
                                             T *outbuf,
                                             int wireId,
                                             int peer,
                                             ssize_t workElems,
                                             size_t offset,
                                             uint32_t flag,
                                             uint32_t slot,
                                             ssize_t nelems,
                                             ccl::reduction reduction,
                                             const ccl_reduction_data &reduction_data,
                                             bool p2p,
                                             AccessGranularity gr = AccessGranularity::Vector) {
        message_t v;
        message_t messages;

        auto *ptr = inbuf + peer * workElems + offset;
        loadInput(v, ptr, nelems, gr);
        applyPreOpMessages(v, reduction_data);

        bool retry;
        sbarrier_wait_compat(p2p);
        do {
            retry = false;
            retry |= recvMessages(messages, recvbuf, flag);
        } while (sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

        restoreData(messages);
        accumMessages(v, messages, reduction);

        ptr = outbuf + offset;
        storeOutput(ptr, v, nelems, gr);
    }
};
