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

#include "coll/algorithms/utils/protocol/proto_base.hpp"
#include "coll/algorithms/utils/sycl_traits.hpp"
#include "coll/algorithms/utils/tvisa/include/lsc.hpp"

template <typename T, int SubGroupSize>
struct Rt64_PCIE
        : public ProtoBase<T, SubGroupSize, sycl::vec<uint32_t, 4>, Rt64_PCIE<T, SubGroupSize>> {
    using message_t = sycl::vec<uint32_t, 4>;
#if defined(__SYCL_DEVICE_ONLY__)
    using inner_t = uint32_t __attribute__((ext_vector_type(4)));
#endif
    constexpr static size_t wireCapacity = SubGroupSize * sizeof(message_t) / 2;
    constexpr static size_t wireTransSize = SubGroupSize * sizeof(message_t);

    constexpr static int wireCapacityInType = wireCapacity / sizeof(T);
    constexpr static int wireTransElems = wireTransSize / sizeof(T);

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

    static inline void loadByElement(message_t &v, T *src, ssize_t nElt) {
        sycl::vec<T, 16 / sizeof(T)> raw((T)0);
        for (ssize_t j = 0; j < nElt; ++j) {
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
                if (nElt == (ssize_t)(sizeof(message_t) / 2 / sizeof(T))) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    if constexpr (SubGroupSize == 16)
                        asm volatile("\n" // Add this partial load to tvisa
                                     "lsc_load.ugm.df.df (M1, 16) %0:d32x2 flat[%1]:a64\n"
                                     : "=rw"(reinterpret_cast<inner_t &>(v))
                                     : "rw"(src));
                    if constexpr (SubGroupSize == 32)
                        asm volatile("\n" // Add this partial load to tvisa
                                     "lsc_load.ugm.df.df (M1, 32) %0:d32x2 flat[%1]:a64\n"
                                     : "=rw"(reinterpret_cast<inner_t &>(v))
                                     : "rw"(src));
#endif
                }
                else {
                    loadByElement(v, src, nElt);
                }
                break;
            }
        }
    }

    // load first row of registers
    template <int unroll>
    static inline void loadInput(message_t (&v)[unroll],
                                 T *src,
                                 ssize_t nElt,
                                 AccessGranularity gr) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        ssize_t local_off = lid * sizeof(message_t) / 2 / sizeof(T);

#pragma unroll
        for (int i = 0; i < unroll; ++i) {
            ssize_t off = i * wireCapacityInType + local_off;
            ssize_t nEltToLoad = nElt - off;
            if (nEltToLoad >= (ssize_t)(sizeof(message_t) / 2 / sizeof(T))) {
                nEltToLoad = (ssize_t)(sizeof(message_t) / 2 / sizeof(T));
            }
            loadInputInner(v[i], src + off, nEltToLoad, gr);
        }
    }

    static inline void loadInput(message_t &v, T *src, ssize_t nElt, AccessGranularity gr) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        ssize_t off = lid * sizeof(message_t) / 2 / sizeof(T);
        ssize_t nEltToLoad = nElt - off;

        if (nEltToLoad >= (ssize_t)(sizeof(message_t) / 2 / sizeof(T))) {
            nEltToLoad = (ssize_t)(sizeof(message_t) / 2 / sizeof(T));
        }
        loadInputInner(v, src + off, nEltToLoad, gr);
    }

    //Insert flags to second row
    template <int unroll>
    static inline void insertFlags(message_t (&messages)[unroll], uint32_t flag) {
#pragma unroll
        for (int i = 0; i < unroll; ++i)
            messages[i][2] = messages[i][3] = flag;
    }

    static inline void insertFlags(message_t &messages, uint32_t flag) {
        messages[2] = messages[3] = flag;
    }

    template <int unroll>
    static inline void shuffleData(message_t (&messages)[unroll]) {}
    static inline void shuffleData(message_t &messages) {}

    template <int unroll>
    static inline void restoreData(message_t (&messages)[unroll]) {}
    static inline void restoreData(message_t &messages) {}

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
                if (nElt == (ssize_t)(sizeof(message_t) / 2 / sizeof(T))) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    if constexpr (SubGroupSize == 16)
                        asm volatile(
                            "\n"
                            "lsc_store.ugm.df.df (M1, 16) flat[%0]:a64 %1:d32x2\n" ::"rw"(dst),
                            "rw"(reinterpret_cast<inner_t &>(v)));
                    if constexpr (SubGroupSize == 32)
                        asm volatile(
                            "\n"
                            "lsc_store.ugm.df.df (M1, 32) flat[%0]:a64 %1:d32x2\n" ::"rw"(dst),
                            "rw"(reinterpret_cast<inner_t &>(v)));
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
        ssize_t local_off = lid * sizeof(message_t) / 2 / sizeof(T);
#pragma unroll
        for (int i = 0; i < unroll; ++i) {
            ssize_t off = i * wireCapacityInType + local_off;
            ssize_t nEltToStore = nElt - off;
            if (nEltToStore >= (ssize_t)(sizeof(message_t) / 2 / sizeof(T))) {
                nEltToStore = (ssize_t)(sizeof(message_t) / 2 / sizeof(T));
            }
            storeOutputInner(dst + off, v[i], nEltToStore, gr);
        }
    }

    static inline void storeOutput(T *dst, message_t &v, ssize_t nElt, AccessGranularity gr) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        ssize_t off = lid * sizeof(message_t) / 2 / sizeof(T);
        ssize_t nEltToStore = nElt - off;
        if (nEltToStore >= (ssize_t)(sizeof(message_t) / 2 / sizeof(T))) {
            nEltToStore = (ssize_t)(sizeof(message_t) / 2 / sizeof(T));
        }
        storeOutputInner(dst + off, v, nEltToStore, gr);
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
            (void)local_off;
#endif
            retry |= (messages[u][2] != flag) || (messages[u][3] != flag);
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
        (void)local_off;
#endif
        retry |= (messages[2] != flag) || (messages[3] != flag);

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
        if (reduction == ccl::reduction::sum) {
            v = sycl::bit_cast<message_t>(a + b);
        }
        else if (reduction == ccl::reduction::prod) {
            v = sycl::bit_cast<message_t>(a * b);
        }
        else if (reduction == ccl::reduction::min) {
            v = sycl::bit_cast<message_t>(sycl_min_op{}(a, b));
        }
        else if (reduction == ccl::reduction::max) {
            v = sycl::bit_cast<message_t>(sycl_max_op{}(a, b));
        }
        else {
            v = sycl::bit_cast<message_t>(a + b);
        }
#endif
    }
};

// should use PCIE variant all the time
template <typename T, int SubGroupSize>
struct Rt64 : public ProtoBase<T, SubGroupSize, sycl::vec<uint32_t, 2>, Rt64<T, SubGroupSize>> {
    using message_t = sycl::vec<uint32_t, 2>;
    static constexpr int dataElem = 0;
    static constexpr int flagElem = 1;

    constexpr static size_t wireCapacity = SubGroupSize * sizeof(message_t) / 2;
    constexpr static size_t wireTransSize = SubGroupSize * sizeof(message_t);

    constexpr static int wireCapacityInType = wireCapacity / sizeof(T);
    constexpr static int wireTransElems = wireTransSize / sizeof(T);

    constexpr static auto CommReadCacheCtrl = CacheCtrl::L1UC_L3C;
    constexpr static auto CommWriteCacheCtrl = CacheCtrl::L1UC_L3WB;

    // load first row of registers
    template <int unroll>
    static inline void loadInput(message_t (&v)[unroll], T *src, ssize_t nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        ssize_t local_off = lid * sizeof(uint32_t) / sizeof(T);

#pragma unroll
        for (int i = 0; i < unroll; ++i) {
            ssize_t off = i * wireCapacityInType + local_off;
            if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                if constexpr (SubGroupSize == 16)
                    asm volatile("\n" // Add this partial load to tvisa
                                 "lsc_load.ugm.df.df (M1, 16) %0:d32 flat[%1]:a64\n"
                                 : "=rw"(v[i][dataElem])
                                 : "rw"(src + off));
                else
                    asm volatile("\n" // Add this partial load to tvisa
                                 "lsc_load.ugm.df.df (M1, 32) %0:d32 flat[%1]:a64\n"
                                 : "=rw"(v[i][dataElem])
                                 : "rw"(src + off));
#else
                v[i][dataElem] = src[off];
#endif
            }
        }
    }

    static inline void loadInput(message_t &v, T *src, ssize_t nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(uint32_t) / sizeof(T);

        if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            if constexpr (SubGroupSize == 16)
                asm volatile("\n" // Add this partial load to tvisa
                             "lsc_load.ugm.df.df (M1, 16) %0:d32 flat[%1]:a64\n"
                             : "=rw"(v[dataElem])
                             : "rw"(src + off));
            else
                asm volatile("\n" // Add this partial load to tvisa
                             "lsc_load.ugm.df.df (M1, 32) %0:d32 flat[%1]:a64\n"
                             : "=rw"(v[dataElem])
                             : "rw"(src + off));
#else
            v[dataElem] = src[off];
#endif
        }
    }

    template <int unroll>
    static inline void loadInput(message_t (&v)[unroll], T *src) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(uint32_t) / sizeof(T);

#pragma unroll
        for (int i = 0; i < unroll; ++i) {
            auto off = i * wireCapacityInType + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            if constexpr (SubGroupSize == 16)
                asm volatile("\n" // Add this partial load to tvisa
                             "lsc_load.ugm.df.df (M1, 16) %0:d32 flat[%1]:a64\n"
                             : "=rw"(v[i][dataElem])
                             : "rw"(src + off));
            else
                asm volatile("\n" // Add this partial load to tvisa
                             "lsc_load.ugm.df.df (M1, 32) %0:d32 flat[%1]:a64\n"
                             : "=rw"(v[i][dataElem])
                             : "rw"(src + off));
#else
            v[i][dataElem] = src[off];
#endif
        }
    }

    static inline void loadInput(message_t &v, T *src) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(uint32_t) / sizeof(T);

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16)
            asm volatile("\n" // Add this partial load to tvisa
                         "lsc_load.ugm.df.df (M1, 16) %0:d32 flat[%1]:a64\n"
                         : "=rw"(v[dataElem])
                         : "rw"(src + off));
        else
            asm volatile("\n" // Add this partial load to tvisa
                         "lsc_load.ugm.df.df (M1, 32) %0:d32 flat[%1]:a64\n"
                         : "=rw"(v[dataElem])
                         : "rw"(src + off));
#else
        v[dataElem] = src[off];
#endif
    }

    //Insert flags to second row
    template <int unroll>
    static inline void insertFlags(message_t (&messages)[unroll], uint32_t flag) {
#pragma unroll
        for (int i = 0; i < unroll; ++i)
            messages[i][flagElem] = flag;
    }

    static inline void insertFlags(message_t &messages, uint32_t flag) {
        messages[flagElem] = flag;
    }

    template <int unroll>
    static inline void shuffleData(message_t (&messages)[unroll]) {}
    static inline void shuffleData(message_t &messages) {}

    template <int unroll>
    static inline void restoreData(message_t (&messages)[unroll]) {}
    static inline void restoreData(message_t &messages) {}

    template <int unroll>
    static inline void storeOutput(T *dst, message_t (&v)[unroll]) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(uint32_t) / sizeof(T);
#pragma unroll
        for (int i = 0; i < unroll; ++i) {
            auto off = i * wireCapacityInType + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            if constexpr (SubGroupSize == 16)
                asm volatile("\n"
                             "lsc_store.ugm.df.df (M1, 16) flat[%0]:a64 %1:d32\n" ::"rw"(dst + off),
                             "rw"(v[i][dataElem]));
            else
                asm volatile("\n"
                             "lsc_store.ugm.df.df (M1, 32) flat[%0]:a64 %1:d32\n" ::"rw"(dst + off),
                             "rw"(v[i][dataElem]));
#else
            dst[off] = v[i][0];
#endif
        }
    }

    static inline void storeOutput(T *dst, message_t &v) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(uint32_t) / sizeof(T);
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16)
            asm volatile("\n"
                         "lsc_store.ugm.df.df (M1, 16) flat[%0]:a64 %1:d32\n" ::"rw"(dst + off),
                         "rw"(v[dataElem]));
        else
            asm volatile("\n"
                         "lsc_store.ugm.df.df (M1, 32) flat[%0]:a64 %1:d32\n" ::"rw"(dst + off),
                         "rw"(v[dataElem]));
#else
        dst[off] = v[0];
#endif
    }

    template <int unroll>
    static inline void storeOutput(T *dst, message_t (&v)[unroll], ssize_t nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int local_off = lid * sizeof(uint32_t) / sizeof(T);
#pragma unroll
        for (int i = 0; i < unroll; ++i) {
            auto off = i * wireCapacityInType + local_off;
            if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                if constexpr (SubGroupSize == 16)
                    asm volatile(
                        "\n"
                        "lsc_store.ugm.df.df (M1, 16) flat[%0]:a64 %1:d32\n" ::"rw"(dst + off),
                        "rw"(v[i][dataElem]));
                else
                    asm volatile(
                        "\n"
                        "lsc_store.ugm.df.df (M1, 32) flat[%0]:a64 %1:d32\n" ::"rw"(dst + off),
                        "rw"(v[i][dataElem]));
#else
                dst[off] = v[i][dataElem];
#endif
            }
        }
    }

    static inline void storeOutput(T *dst, message_t &v, ssize_t nElt) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        auto lid = sg.get_local_id()[0];
        int off = lid * sizeof(uint32_t) / sizeof(T);
        if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            if constexpr (SubGroupSize == 16)
                asm volatile("\n"
                             "lsc_store.ugm.df.df (M1, 16) flat[%0]:a64 %1:d32\n" ::"rw"(dst + off),
                             "rw"(v[dataElem]));
            else
                asm volatile("\n"
                             "lsc_store.ugm.df.df (M1, 32) flat[%0]:a64 %1:d32\n" ::"rw"(dst + off),
                             "rw"(v[dataElem]));
#else
            dst[off] = v[dataElem];
#endif
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
            retry |= (messages[u][flagElem] != flag);
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
        retry |= (messages[flagElem] != flag);

        return retry;
    }

    template <int unroll>
    static inline void accumMessages(message_t (&v)[unroll], message_t (&m)[unroll]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        using math_t = sycl::vec<T, sizeof(uint32_t) / sizeof(T)>;
#pragma unroll
        for (int u = 0; u < unroll; ++u)
            v[u][0] = sycl::bit_cast<uint32_t>(sycl::bit_cast<math_t>(m[u][0]) +
                                               sycl::bit_cast<math_t>(v[u][0]));
#endif
    }

    static inline void accumMessages(message_t &v, message_t &m) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        using math_t = sycl::vec<T, sizeof(uint32_t) / sizeof(T)>;
        v[0] =
            sycl::bit_cast<uint32_t>(sycl::bit_cast<math_t>(m[0]) + sycl::bit_cast<math_t>(v[0]));
#endif
    }
};
