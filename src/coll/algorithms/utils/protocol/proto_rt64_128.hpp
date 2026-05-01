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

#include <sys/types.h>
#include <cstdint>
#include "coll/algorithms/utils/sycl_kernels.hpp"

template <typename T, int SubGroupSize>
struct Rt64_128_PCIE : public ProtoBase<T,
                                        SubGroupSize,
                                        sycl::vec<uint32_t, 4>,
                                        Rt64_128_PCIE<T, SubGroupSize>> {
    using message_t = sycl::vec<uint32_t, 4>;
#if defined(__SYCL_DEVICE_ONLY__)
    using inner_t = uint32_t __attribute__((ext_vector_type(4)));
#endif

    // transaction of 128-byte is not atomic across HBM channel
    constexpr static int payloadChannels = SubGroupSize * 15 / 16;

    constexpr static size_t wireCapacity = payloadChannels * sizeof(message_t);
    constexpr static size_t wireTransSize = SubGroupSize * sizeof(message_t);

    constexpr static size_t wireCapacityInType = wireCapacity / sizeof(T);
    constexpr static size_t wireTransElems = wireTransSize / sizeof(T);

#if defined(CCL_SYCL_ENABLE_PVC) || defined(CCL_SYCL_ENABLE_ARCB)
    constexpr static auto CommReadCacheCtrl = CacheCtrl::L1UC_L3C;
    constexpr static auto CommWriteCacheCtrl = CacheCtrl::L1UC_L3WB;
#else
    constexpr static auto CommReadCacheCtrl = CacheCtrl::L1UC_L3UC;
    constexpr static auto CommWriteCacheCtrl = CacheCtrl::L1UC_L3UC;
#endif

    static inline void loadByByte(message_t &v, T *src, ssize_t bytes) {
        sycl::vec<uint8_t, 16> raw((uint8_t)0);
        uint8_t *raw_src = reinterpret_cast<uint8_t *>(src);
        for (ssize_t j = 0; j < bytes; ++j) {
            raw[j] = raw_src[j];
        }
        v = sycl::bit_cast<message_t>(raw);
    }

    static inline void loadByElement(message_t &v, T *src, ssize_t nElems) {
        sycl::vec<T, 16 / sizeof(T)> raw((T)0);
        for (ssize_t j = 0; j < nElems; ++j) {
            raw[j] = src[j];
        }
        v = sycl::bit_cast<message_t>(raw);
    }

    static inline void loadInputInner(message_t &v, T *src, ssize_t nElt, AccessGranularity gr) {
        switch (gr) {
            case AccessGranularity::Byte: {
                loadByByte(v, src, nElt * sizeof(T));
                break;
            }
            case AccessGranularity::Item: {
                loadByElement(v, src, nElt);
                break;
            }
            case AccessGranularity::Vector: {
                if (nElt == (ssize_t)(sizeof(message_t) / sizeof(T))) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    lscLoad<SubGroupSize>(v, src);
#endif
                }
                else {
                    loadByElement(v, src, nElt);
                }
                break;
            }
        }
    }

    //
    // Process of pack messages
    // 1. multiple load inputs (16 round maximum)
    // 2. Insert flags at the 16th lane
    // 3. Shuffle flag into the middle of second register
    //
    template <int unroll>
    static inline void loadInput(message_t (&v)[unroll],
                                 T *src,
                                 ssize_t nElt,
                                 AccessGranularity gr) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        ssize_t local_off = lid * sizeof(message_t) / sizeof(T);

        if (lid < payloadChannels) {
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                auto off = i * wireCapacityInType + local_off;
                ssize_t nEltToLoad = nElt - off;
                if (nEltToLoad >= (ssize_t)(sizeof(message_t) / sizeof(T))) {
                    nEltToLoad = (ssize_t)(sizeof(message_t) / sizeof(T));
                }
                loadInputInner(v[i], src + off, nEltToLoad, gr);
            }
        }
    }

    static inline void loadInput(message_t &v, T *src, ssize_t nElt, AccessGranularity gr) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(message_t) / sizeof(T);

        if (lid < payloadChannels) {
            ssize_t nEltToLoad = nElt - off;
            if (nEltToLoad >= (ssize_t)(sizeof(message_t) / sizeof(T))) {
                nEltToLoad = (ssize_t)(sizeof(message_t) / sizeof(T));
            }
            loadInputInner(v, src + off, nEltToLoad, gr);
        }
    }

    template <int unroll>
    static inline void insertFlags(message_t (&messages)[unroll], uint32_t flag) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#pragma unroll
        for (int i = 0; i < unroll; ++i) {
#if defined(CCL_SYCL_ENABLE_PVC) || defined(CCL_SYCL_ENABLE_ARCB)
            asm volatile("mov (M1, 1) %0(3, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 11)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 15)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                         : "rw"(flag));
#else
            asm volatile("mov (M1, 1) %0(6, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(6, 7)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 7)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                         : "rw"(flag));
#endif
        }

        if constexpr (SubGroupSize == 32) {
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                asm volatile("mov (M1, 1) %0(3, 19)<1> %1(0, 0)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 23)<1> %1(0, 0)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 27)<1> %1(0, 0)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 31)<1> %1(0, 0)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             : "rw"(flag));
            }
        }
#endif
    }

    static inline void insertFlags(message_t &messages, uint32_t flag) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#if defined(CCL_SYCL_ENABLE_PVC) || defined(CCL_SYCL_ENABLE_ARCB)
        asm volatile("mov (M1, 1) %0(3, 3)<1> %1(0, 0)<0;1,0>\n"
                     "mov (M1, 1) %0(3, 7)<1> %1(0, 0)<0;1,0>\n"
                     "mov (M1, 1) %0(3, 11)<1> %1(0, 0)<0;1,0>\n"
                     "mov (M1, 1) %0(3, 15)<1> %1(0, 0)<0;1,0>\n"
                     : "+rw"(reinterpret_cast<inner_t &>(messages))
                     : "rw"(flag));
#else
        asm volatile("mov (M1, 1) %0(6, 3)<1> %1(0, 0)<0;1,0>\n"
                     "mov (M1, 1) %0(6, 7)<1> %1(0, 0)<0;1,0>\n"
                     "mov (M1, 1) %0(7, 3)<1> %1(0, 0)<0;1,0>\n"
                     "mov (M1, 1) %0(7, 7)<1> %1(0, 0)<0;1,0>\n"
                     : "+rw"(reinterpret_cast<inner_t &>(messages))
                     : "rw"(flag));
#endif

        if constexpr (SubGroupSize == 32) {
            asm volatile("mov (M1, 1) %0(3, 19)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 23)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 27)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 31)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         : "rw"(flag));
        }
#endif
    }

    template <int unroll>
    static inline void shuffleData(message_t (&messages)[unroll]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#pragma unroll
        for (int i = 0; i < unroll; ++i) {
            if constexpr (SubGroupSize == 16) {
#if defined(CCL_SYCL_ENABLE_PVC) || defined(CCL_SYCL_ENABLE_ARCB)
                asm volatile("\n"
                             "mov (M1, 1) %0(0, 15)<1> %0(3, 3)<0;1,0>\n"
                             "mov (M1, 1) %0(1, 15)<1> %0(3, 7)<0;1,0>\n"
                             "mov (M1, 1) %0(2, 15)<1> %0(3, 11)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             :);
#else
                asm volatile("\n"
                             "mov (M1, 1) %0(1, 7)<1> %0(6, 3)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 7)<1> %0(6, 7)<0;1,0>\n"
                             "mov (M1, 1) %0(5, 7)<1> %0(7, 3)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             :);
#endif
            }
            if constexpr (SubGroupSize == 32) {
                asm volatile("\n"
                             "mov (M1, 1) %0(0, 30)<1> %0(3, 3)<0;1,0>\n"
                             "mov (M1, 1) %0(1, 30)<1> %0(3, 7)<0;1,0>\n"
                             "mov (M1, 1) %0(2, 30)<1> %0(3, 11)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 30)<1> %0(3, 15)<0;1,0>\n"
                             "mov (M1, 1) %0(0, 31)<1> %0(3, 19)<0;1,0>\n"
                             "mov (M1, 1) %0(1, 31)<1> %0(3, 23)<0;1,0>\n"
                             "mov (M1, 1) %0(2, 31)<1> %0(3, 27)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             :);
            }
        }
#endif
    }

    static inline void shuffleData(message_t &messages) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16) {
#if defined(CCL_SYCL_ENABLE_PVC) || defined(CCL_SYCL_ENABLE_ARCB)
            asm volatile("\n"
                         "mov (M1, 1) %0(0, 15)<1> %0(3, 3)<0;1,0>\n"
                         "mov (M1, 1) %0(1, 15)<1> %0(3, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(2, 15)<1> %0(3, 11)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         :);
#else
            asm volatile("\n"
                         "mov (M1, 1) %0(1, 7)<1> %0(6, 3)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %0(6, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(5, 7)<1> %0(7, 3)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         :);
#endif
        }

        if constexpr (SubGroupSize == 32) {
            asm volatile("\n"
                         "mov (M1, 1) %0(0, 30)<1> %0(3, 3)<0;1,0>\n"
                         "mov (M1, 1) %0(1, 30)<1> %0(3, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(2, 30)<1> %0(3, 11)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 30)<1> %0(3, 15)<0;1,0>\n"
                         "mov (M1, 1) %0(0, 31)<1> %0(3, 19)<0;1,0>\n"
                         "mov (M1, 1) %0(1, 31)<1> %0(3, 23)<0;1,0>\n"
                         "mov (M1, 1) %0(2, 31)<1> %0(3, 27)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         :);
        }
#endif
    }

    template <int unroll>
    static inline void restoreData(message_t (&messages)[unroll]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#pragma unroll
        for (int i = 0; i < unroll; ++i) {
            if constexpr (SubGroupSize == 16) {
#if defined(CCL_SYCL_ENABLE_PVC) || defined(CCL_SYCL_ENABLE_ARCB)
                asm volatile("\n"
                             "mov (M1, 1) %0(3, 3)<1> %0(0, 15)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 7)<1> %0(1, 15)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 11)<1> %0(2, 15)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             :);
#else
                asm volatile("\n"
                             "mov (M1, 1) %0(6, 3)<1> %0(1, 7)<0;1,0>\n"
                             "mov (M1, 1) %0(6, 7)<1> %0(3, 7)<0;1,0>\n"
                             "mov (M1, 1) %0(7, 3)<1> %0(5, 7)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             :);
#endif
            }
            else {
                asm volatile("\n"
                             "mov (M1, 1) %0(3, 3)<1> %0(0, 30)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 7)<1> %0(1, 30)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 11)<1> %0(2, 30)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 15)<1> %0(3, 30)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 19)<1> %0(0, 31)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 23)<1> %0(1, 31)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 27)<1> %0(2, 31)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             :);
            }
        }
#endif
    }

    static inline void restoreData(message_t &messages) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16) {
#if defined(CCL_SYCL_ENABLE_PVC) || defined(CCL_SYCL_ENABLE_ARCB)
            asm volatile("\n"
                         "mov (M1, 1) %0(3, 3)<1> %0(0, 15)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %0(1, 15)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 11)<1> %0(2, 15)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         :);
#else
            asm volatile("\n"
                         "mov (M1, 1) %0(6, 3)<1> %0(1, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(6, 7)<1> %0(3, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 3)<1> %0(5, 7)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         :);
#endif
        }
        else {
            asm volatile("\n"
                         "mov (M1, 1) %0(3, 3)<1> %0(0, 30)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %0(1, 30)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 11)<1> %0(2, 30)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 15)<1> %0(3, 30)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 19)<1> %0(0, 31)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 23)<1> %0(1, 31)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 27)<1> %0(2, 31)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         :);
        }
#endif
    }

    static inline void storeByByte(T *dst, message_t &v, ssize_t bytes) {
        auto dst_bytes = reinterpret_cast<uint8_t *>(dst);
        auto v_raw = sycl::bit_cast<sycl::vec<uint8_t, 16>>(v);
        for (ssize_t j = 0; j < bytes; ++j) {
            dst_bytes[j] = v_raw[j];
        }
    }

    static inline void storeByElement(T *dst, message_t &v, ssize_t nElems) {
        auto v_raw = sycl::bit_cast<sycl::vec<T, 16 / sizeof(T)>>(v);
        for (ssize_t j = 0; j < nElems; ++j) {
            dst[j] = v_raw[j];
        }
    }

    static inline void storeOutputInner(T *dst, message_t &v, ssize_t nElt, AccessGranularity gr) {
        switch (gr) {
            case AccessGranularity::Byte: {
                storeByByte(dst, v, nElt * sizeof(T));
                break;
            }
            case AccessGranularity::Item: {
                storeByElement(dst, v, nElt);
                break;
            }
            case AccessGranularity::Vector: {
                if (nElt == (ssize_t)(sizeof(message_t) / sizeof(T))) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    lscStore<SubGroupSize>(dst, v);
#endif
                }
                else {
                    storeByElement(dst, v, nElt);
                }
                break;
            }
        }
    }

    template <int unroll>
    static inline void storeOutput(T *dst,
                                   message_t (&v)[unroll],
                                   ssize_t nElt,
                                   AccessGranularity gr) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        ssize_t local_off = lid * sizeof(message_t) / sizeof(T);
        if (lid < payloadChannels) {
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                ssize_t off = i * wireCapacityInType + local_off;
                ssize_t nEltToStore = nElt - off;
                if (nEltToStore >= (ssize_t)(sizeof(message_t) / sizeof(T))) {
                    nEltToStore = (ssize_t)(sizeof(message_t) / sizeof(T));
                }
                storeOutputInner(dst + off, v[i], nEltToStore, gr);
            }
        }
    }

    static inline void storeOutput(T *dst, message_t &v, ssize_t nElt, AccessGranularity gr) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        ssize_t off = lid * sizeof(message_t) / sizeof(T);
        if (lid < payloadChannels) {
            ssize_t nEltToStore = nElt - off;
            if (nEltToStore >= (ssize_t)(sizeof(message_t) / sizeof(T))) {
                nEltToStore = (ssize_t)(sizeof(message_t) / sizeof(T));
            }
            storeOutputInner(dst + off, v, nEltToStore, gr);
        }
    }

    template <int unroll>
    static inline void sendMessages(T *ptr, message_t (&messages)[unroll]) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

#pragma unroll
        for (int u = 0; u < unroll; ++u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            lscStore<SubGroupSize, CommWriteCacheCtrl>(ptr + u * wireTransElems + local_off,
                                                       messages[u]);
#else
            (void)lid;
            (void)local_off;
#endif
        }
    }

    static inline void sendMessages(T *ptr, message_t &messages) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscStore<SubGroupSize, CommWriteCacheCtrl>(ptr + local_off, messages);
#else
        (void)lid;
        (void)local_off;
#endif
    }

    template <int unroll>
    static inline bool recvMessages(message_t (&messages)[unroll], T *ptr, uint32_t flag) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

        bool retry = false;

#pragma unroll
        for (int u = 0; u < unroll; ++u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            lscLoad<SubGroupSize, CommReadCacheCtrl>(messages[u],
                                                     ptr + u * wireTransElems + local_off);
#else
            (void)lid;
            (void)local_off;
#endif
            if constexpr (SubGroupSize == 16)
                retry =
                    (lid == 3 && messages[u][3] != flag) || (lid == 7 && messages[u][3] != flag) ||
                    (lid == 11 && messages[u][3] != flag) || (lid == 15 && messages[u][3] != flag);

            if constexpr (SubGroupSize == 32)
                retry =
                    (lid == 3 && messages[u][3] != flag) || (lid == 7 && messages[u][3] != flag) ||
                    (lid == 11 && messages[u][3] != flag) ||
                    (lid == 15 && messages[u][3] != flag) ||
                    (lid == 19 && messages[u][3] != flag) ||
                    (lid == 23 && messages[u][3] != flag) ||
                    (lid == 27 && messages[u][3] != flag) || (lid == 31 && messages[u][3] != flag);
        }
        return retry;
    }

    static inline bool recvMessages(message_t &messages, T *ptr, uint32_t flag) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

        bool retry = false;

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscLoad<SubGroupSize, CommReadCacheCtrl>(messages, ptr + local_off);
#else
        (void)lid;
        (void)local_off;
#endif
        if constexpr (SubGroupSize == 16)
            retry = (lid == 3 && messages[3] != flag) || (lid == 7 && messages[3] != flag) ||
                    (lid == 11 && messages[3] != flag) || (lid == 15 && messages[3] != flag);

        if constexpr (SubGroupSize == 32)
            retry = (lid == 3 && messages[3] != flag) || (lid == 7 && messages[3] != flag) ||
                    (lid == 11 && messages[3] != flag) || (lid == 15 && messages[3] != flag) ||
                    (lid == 19 && messages[3] != flag) || (lid == 23 && messages[3] != flag) ||
                    (lid == 27 && messages[3] != flag) || (lid == 31 && messages[3] != flag);
        return retry;
    }

    template <int unroll>

    static inline void accumMessages(message_t (&v)[unroll],
                                     message_t (&m)[unroll],
                                     ccl::reduction reduction = ccl::reduction::sum) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        using math_t = sycl::vec<T, sizeof(message_t) / sizeof(T)>;
        if (reduction == ccl::reduction::sum) {
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                auto a = sycl::bit_cast<math_t>(v[u]);
                auto b = sycl::bit_cast<math_t>(m[u]);
                v[u] = sycl::bit_cast<message_t>(a + b);
            }
        }
        else if (reduction == ccl::reduction::prod) {
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                auto a = sycl::bit_cast<math_t>(v[u]);
                auto b = sycl::bit_cast<math_t>(m[u]);
                v[u] = sycl::bit_cast<message_t>(a * b);
            }
        }
        else if (reduction == ccl::reduction::min) {
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                auto a = sycl::bit_cast<math_t>(v[u]);
                auto b = sycl::bit_cast<math_t>(m[u]);
                v[u] = sycl::bit_cast<message_t>(sycl_min_op{}(a, b));
            }
        }
        else if (reduction == ccl::reduction::max) {
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                auto a = sycl::bit_cast<math_t>(v[u]);
                auto b = sycl::bit_cast<math_t>(m[u]);
                v[u] = sycl::bit_cast<message_t>(sycl_max_op{}(a, b));
            }
        }
        else {
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                auto a = sycl::bit_cast<math_t>(v[u]);
                auto b = sycl::bit_cast<math_t>(m[u]);
                v[u] = sycl::bit_cast<message_t>(a + b);
            }
        }
#endif
    }

    static inline void accumMessages(message_t &v,
                                     message_t &m,
                                     ccl::reduction reduction = ccl::reduction::sum) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        using math_t = sycl::vec<T, sizeof(message_t) / sizeof(T)>;
        auto a = sycl::bit_cast<math_t>(v);
        auto b = sycl::bit_cast<math_t>(m);
        math_t r;
        if (reduction == ccl::reduction::sum) {
            r = a + b;
        }
        else if (reduction == ccl::reduction::prod) {
            r = a * b;
        }
        else if (reduction == ccl::reduction::min) {
            r = sycl_min_op{}(a, b);
        }
        else if (reduction == ccl::reduction::max) {
            r = sycl_max_op{}(a, b);
        }
        else {
            r = a + b;
        }
        v = sycl::bit_cast<message_t>(r);
#endif
    }
};

template <typename T, int SubGroupSize>
struct Rt64_128 : public ProtoBase<T,
                                   SubGroupSize,
                                   sycl::vec<uint32_t, 128 / SubGroupSize / 4>,
                                   Rt64_128<T, SubGroupSize>> {
    constexpr static int nReg128B = 128 / SubGroupSize / 4;
    constexpr static int firstElem = 0;
    constexpr static int lastElem = nReg128B - 1;

    using message_t = sycl::vec<uint32_t, nReg128B>;
#if defined(__SYCL_DEVICE_ONLY__)
    using inner_t = uint32_t __attribute__((ext_vector_type(nReg128B)));
#endif

    // transaction of 128-byte is not atomic across HBM channel
    constexpr static int nChan8B = 8 / sizeof(message_t);
    constexpr static int lastDataChannel = SubGroupSize - nChan8B;
    constexpr static int firstFlagChannel = SubGroupSize / 2 - 1;
    constexpr static int lastFlagChannel = SubGroupSize - 1;
    constexpr static size_t wireCapacity = (SubGroupSize - nChan8B) * sizeof(message_t);
    constexpr static size_t wireTransSize = SubGroupSize * sizeof(message_t);
    constexpr static size_t wireCapacityInType = wireCapacity / sizeof(T);
    constexpr static size_t wireTransElems = wireTransSize / sizeof(T);

    constexpr static auto CommReadCacheCtrl = CacheCtrl::L1UC_L3C;
    constexpr static auto CommWriteCacheCtrl = CacheCtrl::L1UC_L3WB;

    //
    // Process of pack messages
    // 1. multiple load inputs (16 round maximum)
    // 2. Insert flags at the 16th lane
    // 3. Shuffle flag into the middle of second register
    //
    template <int unroll>
    static inline void loadInput(message_t (&v)[unroll], T *src, ssize_t nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

        if (lid < lastDataChannel) { // TODO: diverge
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                auto off = i * wireCapacityInType + local_off;
                if (off < nElt) { // TODO: condition branch !
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    lscLoad<SubGroupSize /*, CacheCtrl::L1UC_L3UC*/>(v[i], src + off);
#else
                    (void)off;
#endif
                }
            }
        }
    }

    static inline void loadInput(message_t &v, T *src, ssize_t nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(message_t) / sizeof(T);

        if (lid < lastDataChannel) { // TODO: diverge
            if (off < nElt) { // TODO: condition branch !
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                lscLoad<SubGroupSize>(v, src + off);
#else
                (void)off;
#endif
            }
        }
    }

    template <int unroll>
    static inline void loadInput(message_t (&v)[unroll], T *src) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

        if (lid < lastDataChannel) { // XXX: diverge
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                auto off = i * wireCapacityInType + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                lscLoad<SubGroupSize>(v[i], src + off);
#else
                (void)off;
#endif
            }
        }
    }

    static inline void loadInput(message_t &v, T *src) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(message_t) / sizeof(T);

        if (lid < lastDataChannel) { // XXX: diverge
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            lscLoad<SubGroupSize>(v, src + off);
#else
            (void)off;
#endif
        }
    }

    template <int unroll>
    static inline void insertFlags(message_t (&messages)[unroll], uint32_t flag) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16) {
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                asm volatile("mov (M1, 1) %0(1, 7)<1> %1(0, 0)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             : "rw"(flag));
            }

#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                asm volatile("mov (M1, 1) %0(1, 15)<1> %1(0, 0)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             : "rw"(flag));
            }
        }
        else {
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                asm volatile("mov (M1, 1) %0(0, 15)<1> %1(0, 0)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             : "rw"(flag));
            }

#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                asm volatile("mov (M1, 1) %0(0, 31)<1> %1(0, 0)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             : "rw"(flag));
            }
        }
#else
        // Add flags at the middle and tail
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        if (lid == firstFlagChannel || lid == lastFlagChannel) {
#pragma unroll
            for (int i = 0; i < unroll; ++i)
                messages[i][lastElem] = flag;
        }
#endif
    }

    static inline void insertFlags(message_t &messages, uint32_t flag) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16) {
            asm volatile("mov (M1, 1) %0(1, 7)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         : "rw"(flag));

            asm volatile("mov (M1, 1) %0(1, 15)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         : "rw"(flag));
        }
        else {
            asm volatile("mov (M1, 1) %0(0, 15)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         : "rw"(flag));

            asm volatile("mov (M1, 1) %0(0, 31)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         : "rw"(flag));
        }
#else
        // Add flags at the middle and tail
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        if (lid == firstFlagChannel || lid == lastFlagChannel) {
            messages[lastElem] = flag;
        }
#endif
    }

    template <int unroll>
    static inline void shuffleData(message_t (&messages)[unroll]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#pragma unroll
        for (int i = 0; i < unroll; ++i) {
            if constexpr (SubGroupSize == 16) {
                asm volatile("\n"
                             "mov (M1, 1) %0(0, 15)<1> %0(1, 7)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             :);
            }
            else {
                asm volatile("\n"
                             "mov (M1, 1) %0(0, 30)<1> %0(0, 15)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             :);
            }
        }
#endif
    }

    static inline void shuffleData(message_t &messages) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16) {
            asm volatile("\n"
                         "mov (M1, 1) %0(0, 15)<1> %0(1, 7)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         :);
        }
        else {
            asm volatile("\n"
                         "mov (M1, 1) %0(0, 30)<1> %0(0, 15)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         :);
        }
#endif
    }

    template <int unroll>
    static inline void restoreData(message_t (&messages)[unroll]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#pragma unroll
        for (int i = 0; i < unroll; ++i) {
            if constexpr (SubGroupSize == 16) {
                asm volatile("\n"
                             "mov (M1, 1) %0(1, 7)<1> %0(0, 15)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             :);
            }
            else {
                asm volatile("\n"
                             "mov (M1, 1) %0(0, 15)<1> %0(0, 30)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<inner_t &>(messages[i]))
                             :);
            }
        }
#endif
    }

    static inline void restoreData(message_t &messages) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16) {
            asm volatile("\n"
                         "mov (M1, 1) %0(1, 7)<1> %0(0, 15)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         :);
        }
        else {
            asm volatile("\n"
                         "mov (M1, 1) %0(0, 15)<1> %0(0, 30)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<inner_t &>(messages))
                         :);
        }
#endif
    }

    template <int unroll>
    static inline void storeOutput(T *dst, message_t (&v)[unroll]) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);
        if (lid < lastDataChannel) { // XXX: Diverge
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                auto off = i * wireCapacityInType + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                lscStore<SubGroupSize /*, CacheCtrl::L1UC_L3UC*/>(dst + off, v[i]);
#else
                (void)off;
                (void)local_off;
#endif
            }
        }
    }

    static inline void storeOutput(T *dst, message_t &v) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(message_t) / sizeof(T);
        if (lid < lastDataChannel) { // XXX: Diverge
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            lscStore<SubGroupSize>(dst + off, v);
#else
            (void)off;
#endif
        }
    }

    template <int unroll>
    static inline void storeOutput(T *dst, message_t (&v)[unroll], ssize_t nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);
        if (lid < lastDataChannel) { // XXX: Fixed diverge
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                auto off = i * wireCapacityInType + local_off;
                if (off < nElt) { // XXX: runtime condition
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    lscStore<SubGroupSize /*, CacheCtrl::L1UC_L3UC*/>(dst + off, v[i]);
#endif
                }
            }
        }
    }

    static inline void storeOutput(T *dst, message_t &v, ssize_t nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(message_t) / sizeof(T);
        if (lid < lastDataChannel) { // XXX: Fixed diverge
            if (off < nElt) { // XXX: runtime condition
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                lscStore<SubGroupSize>(dst + off, v);
#endif
            }
        }
    }

    // We always push 128-byte packages
    template <int unroll>
    static inline void sendMessages(T *ptr, message_t (&messages)[unroll]) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

#pragma unroll
        for (int u = 0; u < unroll; ++u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            lscStore<SubGroupSize, CommWriteCacheCtrl>(ptr + u * wireTransElems + local_off,
                                                       messages[u]);
#else
            (void)lid;
            (void)local_off;
#endif
        }
    }

    static inline void sendMessages(T *ptr, message_t &messages) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscStore<SubGroupSize, CommWriteCacheCtrl>(ptr + local_off, messages);
#else
        (void)lid;
        (void)local_off;
#endif
    }

    template <int unroll>
    static inline bool recvMessages(message_t (&messages)[unroll], T *ptr, uint32_t flag) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

        bool retry = false;

#pragma unroll
        for (int u = 0; u < unroll; ++u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            lscLoad<SubGroupSize, CommReadCacheCtrl>(messages[u],
                                                     ptr + u * wireTransElems + local_off);
#else
            (void)lid;
            (void)local_off;
#endif
            retry |= (lid == firstFlagChannel && messages[u][lastElem] != flag) ||
                     (lid == lastFlagChannel && messages[u][lastElem] != flag);
        }
        return retry;
    }

    static inline bool recvMessages(message_t &messages, T *ptr, uint32_t flag) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

        bool retry = false;

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscLoad<SubGroupSize, CommReadCacheCtrl>(messages, ptr + local_off);
#else
        (void)lid;
        (void)local_off;
#endif
        retry |= (lid == firstFlagChannel && messages[lastElem] != flag) ||
                 (lid == lastFlagChannel && messages[lastElem] != flag);
        return retry;
    }

    template <int unroll>
    static inline void accumMessages(message_t (&v)[unroll], message_t (&m)[unroll]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        using math_t = sycl::vec<T, sizeof(message_t) / sizeof(T)>;
#pragma unroll
        for (int u = 0; u < unroll; ++u)
            v[u] = sycl::bit_cast<message_t>(sycl::bit_cast<math_t>(m[u]) +
                                             sycl::bit_cast<math_t>(v[u]));
#endif
    }

    static inline void accumMessages(message_t &v, message_t &m) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        using math_t = sycl::vec<T, sizeof(message_t) / sizeof(T)>;
        v = sycl::bit_cast<message_t>(sycl::bit_cast<math_t>(m) + sycl::bit_cast<math_t>(v));
#endif
    }
};
