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

template <typename T, int SubGroupSize>
struct Rt64_128_PCIE {
    using message_t = sycl::vec<uint32_t, 4>;

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

    //
    // Process of pack messages
    // 1. multiple load inputs (16 round maximum)
    // 2. Insert flags at the 16th lane
    // 3. Shuffle flag into the middle of second register
    //
    template <int unroll>
    static inline void loadInput(message_t (&v)[unroll], T *src, int nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);

        if (lid < payloadChannels) {
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                auto off = i * wireCapacityInType + local_off;
                if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    lscLoad<SubGroupSize>(v[i], src + off);
#else
                    (void)off;
#endif
                }
            }
        }
    }

    static inline void loadInput(message_t &v, T *src, int nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(message_t) / sizeof(T);

        if (lid < payloadChannels) {
            if (off < nElt) {
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

        if (lid < payloadChannels) {
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

        if (lid < payloadChannels) {
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
#pragma unroll
        for (int i = 0; i < unroll; ++i) {
#if defined(CCL_SYCL_ENABLE_PVC) || defined(CCL_SYCL_ENABLE_ARCB)
            asm volatile("mov (M1, 1) %0(3, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 11)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 15)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
                         : "rw"(flag));
#else
            asm volatile("mov (M1, 1) %0(6, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(6, 7)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 7)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
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
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
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
                     : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
                     : "rw"(flag));
#else
        asm volatile("mov (M1, 1) %0(6, 3)<1> %1(0, 0)<0;1,0>\n"
                     "mov (M1, 1) %0(6, 7)<1> %1(0, 0)<0;1,0>\n"
                     "mov (M1, 1) %0(7, 3)<1> %1(0, 0)<0;1,0>\n"
                     "mov (M1, 1) %0(7, 7)<1> %1(0, 0)<0;1,0>\n"
                     : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
                     : "rw"(flag));
#endif

        if constexpr (SubGroupSize == 32) {
            asm volatile("mov (M1, 1) %0(3, 19)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 23)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 27)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 31)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
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
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
                             :);
#else
                asm volatile("\n"
                             "mov (M1, 1) %0(1, 7)<1> %0(6, 3)<0;1,0>\n"
                             "mov (M1, 1) %0(3, 7)<1> %0(6, 7)<0;1,0>\n"
                             "mov (M1, 1) %0(5, 7)<1> %0(7, 3)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
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
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
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
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
                         :);
#else
            asm volatile("\n"
                         "mov (M1, 1) %0(1, 7)<1> %0(6, 3)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %0(6, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(5, 7)<1> %0(7, 3)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
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
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
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
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
                             :);
#else
                asm volatile("\n"
                             "mov (M1, 1) %0(6, 3)<1> %0(1, 7)<0;1,0>\n"
                             "mov (M1, 1) %0(6, 7)<1> %0(3, 7)<0;1,0>\n"
                             "mov (M1, 1) %0(7, 3)<1> %0(5, 7)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
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
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
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
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
                         :);
#else
            asm volatile("\n"
                         "mov (M1, 1) %0(6, 3)<1> %0(1, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(6, 7)<1> %0(3, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 3)<1> %0(5, 7)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
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
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
                         :);
        }
#endif
    }

    template <int unroll>
    static inline void storeOutput(T *dst, message_t (&v)[unroll]) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);
        if (lid < payloadChannels) { // XXX: Diverge
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                auto off = i * wireCapacityInType + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                lscStore<SubGroupSize>(dst + off, v[i]);
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
        if (lid < payloadChannels) { // XXX: Diverge
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            lscStore<SubGroupSize>(dst + off, v);
#else
            (void)off;
#endif
        }
    }

    template <int unroll>
    static inline void storeOutput(T *dst, message_t (&v)[unroll], int nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(message_t) / sizeof(T);
        if (lid < payloadChannels) { // XXX: Fixed diverge
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                auto off = i * wireCapacityInType + local_off;
                if (off < nElt) { // XXX: runtime condition
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    lscStore<SubGroupSize>(dst + off, v[i]);
#endif
                }
            }
        }
    }

    static inline void storeOutput(T *dst, message_t &v, int nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(message_t) / sizeof(T);
        if (lid < payloadChannels) { // XXX: Fixed diverge
            if (off < nElt) { // XXX: runtime condition
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                lscStore<SubGroupSize>(dst + off, v);
#endif
            }
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

template <typename T, int SubGroupSize>
struct Rt64_128 {
    constexpr static int nReg128B = 128 / SubGroupSize / 4;
    constexpr static int firstElem = 0;
    constexpr static int lastElem = nReg128B - 1;

    using message_t = sycl::vec<uint32_t, nReg128B>;
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
    static inline void loadInput(message_t (&v)[unroll], T *src, int nElt) {
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

    static inline void loadInput(message_t &v, T *src, int nElt) {
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
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
                             : "rw"(flag));
            }

#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                asm volatile("mov (M1, 1) %0(1, 15)<1> %1(0, 0)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
                             : "rw"(flag));
            }
        }
        else {
#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                asm volatile("mov (M1, 1) %0(0, 15)<1> %1(0, 0)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
                             : "rw"(flag));
            }

#pragma unroll
            for (int i = 0; i < unroll; ++i) {
                asm volatile("mov (M1, 1) %0(0, 31)<1> %1(0, 0)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
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
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
                         : "rw"(flag));

            asm volatile("mov (M1, 1) %0(1, 15)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
                         : "rw"(flag));
        }
        else {
            asm volatile("mov (M1, 1) %0(0, 15)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
                         : "rw"(flag));

            asm volatile("mov (M1, 1) %0(0, 31)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
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
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
                             :);
            }
            else {
                asm volatile("\n"
                             "mov (M1, 1) %0(0, 30)<1> %0(0, 15)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
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
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
                         :);
        }
        else {
            asm volatile("\n"
                         "mov (M1, 1) %0(0, 30)<1> %0(0, 15)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
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
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
                             :);
            }
            else {
                asm volatile("\n"
                             "mov (M1, 1) %0(0, 15)<1> %0(0, 30)<0;1,0>\n"
                             : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
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
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
                         :);
        }
        else {
            asm volatile("\n"
                         "mov (M1, 1) %0(0, 15)<1> %0(0, 30)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages))
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
    static inline void storeOutput(T *dst, message_t (&v)[unroll], int nElt) {
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

    template <int unroll>
    static inline void storeOutput(T *dst, message_t &v, int nElt) {
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
