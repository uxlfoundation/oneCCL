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

#define xstr(s) str(s)
#define str(x)  #x

// enumeration for all Sends
template <int DataWidth, int DestRegNumber, DataShuffle Transpose, CacheCtrl Cache>
struct RawSendLoad {
    template <typename T, typename AddressPayload>
    static inline void run(T& target, const AddressPayload& adrs);
};

template <int DataWidth, int DestRegNumber, DataShuffle Transpose, CacheCtrl Cache>
struct RawSendStore {
    template <typename T, typename AddressPayload>
    static inline void run(const AddressPayload& adrs, const T& target);
};

// store a matrix with height = 32 in elem, and width = 32 in bytes
template <int DataWidth, int DestRegNumber, DataShuffle Transpose, CacheCtrl Cache>
struct RawSendStore32_32 {
    template <typename T, typename AddressPayload>
    static inline void run(const AddressPayload& adrs, const T& target);
};

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)

#define EnumerateLoads(DataWidth, DestRegNumber, DataShuffle, CacheCtrl, DescStr) \
    template <> \
    struct RawSendLoad<DataWidth, DestRegNumber, DataShuffle, CacheCtrl> { \
        template <typename T, typename AddressPayload> \
        static inline void run(T& target, const AddressPayload& address) { \
            asm volatile("\n" \
                         "raw_sends.15.1.0." str(DestRegNumber) " (M1, 1) 0x0:ud " str( \
                             DescStr) ":ud %1.0 V0.0 %0.0\n" \
                         : "=rw"(target) \
                         : "rw"(address.getPayload())); \
        } \
    };

#include "list_raw_loads.list"

#define EnumerateStores(DataWidth, SrcRegNumber, DataShuffle, CacheCtrl, DescStr) \
    template <> \
    struct RawSendStore<DataWidth, SrcRegNumber, DataShuffle, CacheCtrl> { \
        template <typename T, typename AddressPayload> \
        static inline void run(const AddressPayload& address, const T& target) { \
            asm volatile("\n" \
                         "raw_sends.15.1." str(SrcRegNumber) ".0 (M1, 1) 0x0:ud " str( \
                             DescStr) ":ud %0.0 %1.0 V0.0\n" ::"rw"(address.getPayload()), \
                         "rw"(target)); \
        } \
    };

#include "list_raw_stores.list"

#define EnumerateLargeStores(DataWidth, DataShuffle, CacheCtrl, DescStr) \
    template <> \
    struct RawSendStore32_32<DataWidth, 16, DataShuffle, CacheCtrl> { \
        template <typename T, typename AddressPayload> \
        static inline void run(const AddressPayload& address, const T& target) { \
            asm volatile( \
                "\n" \
                "raw_sends.15.1.8.0 (M1, 1)  0x0:ud " str( \
                    DescStr) ":ud %0.0 %1.0 V0.0\n" \
                             "add (M1, 1) %0(0, 6)<1> %0(0, 6)<0;1,0> %2\n" \
                             "raw_sends.15.1.8.0 (M1, 1) 0x0:ud " str( \
                                 DescStr) ":ud %0.0 %1.512 V0.0\n" ::"rw"(address.getPayload()), \
                "rw"(target), \
                "i"(16)); \
        } \
    };

EnumerateLargeStores(1, DataShuffle::none, CacheCtrl::DEFAULT, 0x2800207);
EnumerateLargeStores(1, DataShuffle::none, CacheCtrl::L1UC_L3UC, 0x2820207);
EnumerateLargeStores(1, DataShuffle::none, CacheCtrl::L1UC_L3WB, 0x2840207);
EnumerateLargeStores(1, DataShuffle::none, CacheCtrl::L1WT_L3UC, 0x2860207);
EnumerateLargeStores(1, DataShuffle::none, CacheCtrl::L1WT_L3WB, 0x2880207);
EnumerateLargeStores(1, DataShuffle::none, CacheCtrl::L1S_L3UC, 0x28a0207);
EnumerateLargeStores(1, DataShuffle::none, CacheCtrl::L1S_L3WB, 0x28c0207);
EnumerateLargeStores(1, DataShuffle::none, CacheCtrl::L1WB_L3WB, 0x28e0207);
#endif

template <int DataWidth, int VectorSize, int SubGroupSize, CacheCtrl = CacheCtrl::DEFAULT>
struct LscLoad {
    template <typename T>
    static inline void run(T& var, const void* addr);
};

template <int DataWidth, int VectorSize, int SubGroupSize, CacheCtrl = CacheCtrl::DEFAULT>
struct LscStore {
    template <typename T>
    static inline void run(void* addr, const T& var);
};

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)

#define EnumerateLSCLoad(DataWidth, VectorSize, SubGroupSize, CacheCtrl, CacheStr, DtypeStr) \
    template <> \
    struct LscLoad<DataWidth, VectorSize, SubGroupSize, CacheCtrl> { \
        template <typename T> \
        static inline void run(T& var, const void* addr) { \
            asm volatile("\n" \
                         "lsc_load.ugm." str(CacheStr) " (M1, " str(SubGroupSize) ") %0:" str( \
                             DtypeStr) " flat[%1]:a64\n" \
                         : "=rw"(var) \
                         : "rw"(addr)); \
        } \
    };

#include "list_ugm_loads.list"

#define EnumerateLSCStore(DataWidth, VectorSize, SubGroupSize, CacheCtrl, CacheStr, DtypeStr) \
    template <> \
    struct LscStore<DataWidth, VectorSize, SubGroupSize, CacheCtrl> { \
        template <typename T> \
        static inline void run(void* addr, const T& var) { \
            asm volatile("\n" \
                         "lsc_store.ugm." str(CacheStr) " (M1, " str( \
                             SubGroupSize) ") flat[%0]:a64 %1:" str(DtypeStr) "\n" ::"rw"(addr), \
                         "rw"(var)); \
        } \
    };

#include "list_ugm_stores.list"

#endif

template <int DataWidth, int DestRegNumber, DataShuffle Transpose, CacheCtrl Cache>
struct RawPrefetch {
    template <typename AddressPayload>
    static inline void run(const AddressPayload& addr);
};

template <int DataWidth, int VectorSize, int SubGroupSize, CacheCtrl = CacheCtrl::DEFAULT>
struct LscPrefetch {
    static inline void run(const void* addr);
};

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)

#define EnumeratePrefetch(DataWidth, DestRegNumber, DataShuffle, CacheCtrl, DescStr) \
    template <> \
    struct RawPrefetch<DataWidth, DestRegNumber, DataShuffle, CacheCtrl> { \
        template <typename AddressPayload> \
        static inline void run(const AddressPayload& address) { \
            asm volatile("\n" \
                         "raw_sends.15.1.0." str(DestRegNumber) " (M1, 1) 0x0:ud " str( \
                             DescStr) ":ud %0.0 V0.0 V0.0\n" \
                         : \
                         : "rw"(address.getPayload())); \
        } \
    };

#include "list_raw_prefetches.list"

#define EnumerateLSCPrefetch(DataWidth, VectorSize, SubGroupSize, CacheCtrl, CacheStr, DtypeStr) \
    template <> \
    struct LscPrefetch<DataWidth, VectorSize, SubGroupSize, CacheCtrl> { \
        static inline void run(const void* addr) { \
            asm volatile("\n" \
                         "lsc_load.ugm." str(CacheStr) " (M1, " str(SubGroupSize) ") V0:" str( \
                             DtypeStr) " flat[%0]:a64\n" \
                         : \
                         : "rw"(addr)); \
        } \
    };

#include "list_ugm_prefetches.list"

#endif
