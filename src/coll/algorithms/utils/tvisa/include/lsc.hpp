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

#include <type_traits>
#include <sycl/sycl.hpp>

enum DataShuffle { none, transpose, vnni };

enum CacheCtrl {
    DEFAULT = 0,
    L1UC_L3UC,
    L1UC_L3C,
    L1UC_L3WB = L1UC_L3C,
    L1C_L3UC,
    L1WT_L3UC = L1C_L3UC,
    L1C_L3C,
    L1WT_L3WB = L1C_L3C,
    L1S_L3UC,
    L1S_L3C,
    L1S_L3WB = L1S_L3C,
    L1IAR_L3C,
    L1WB_L3WB = L1IAR_L3C
};

enum BarrierType { ProducerConsumer = 0, ProducerOnly, ConsumerOnly };

#include "regmap.hpp"
#include "lsc_untyped_list.hpp"

// API
template <int subGroupSize, CacheCtrl CTL = CacheCtrl::DEFAULT, typename T>
static inline void lscLoad(T& var, void* addr) {
    LscLoad<sizeof(T), 1, subGroupSize, CTL>::run(var, addr);
}

template <int subGroupSize, CacheCtrl CTL = CacheCtrl::DEFAULT, typename T>
static inline void lscStore(void* addr, const T& var) {
    LscStore<sizeof(T), 1, subGroupSize, CTL>::run(addr, var);
}

// Will array type be contiguous in registers???
template <int subGroupSize, CacheCtrl CTL = CacheCtrl::DEFAULT, typename T, int N>
static inline void lscLoad(T (&var)[N], void* addr) {
    LscLoad<sizeof(T), N, subGroupSize, CTL>::run(var, addr);
}

template <int subGroupSize, CacheCtrl CTL = CacheCtrl::DEFAULT, typename T, int N>
static inline void lscStore(void* addr, const T (&var)[N]) {
    LscStore<sizeof(T), N, subGroupSize, CTL>::run(addr, var);
}

template <int subGroupSize, CacheCtrl CTL = CacheCtrl::DEFAULT, typename T, int N>
static inline void lscLoad(sycl::vec<T, N>& var, void* addr) {
    LscLoad<sizeof(T), N, subGroupSize, CTL>::run(
        reinterpret_cast<typename sycl::vec<T, N>::vector_t&>(var), addr);
}

template <int subGroupSize, CacheCtrl CTL = CacheCtrl::DEFAULT, typename T, int N>
static inline void lscStore(void* addr, const sycl::vec<T, N>& var) {
    LscStore<sizeof(T), N, subGroupSize, CTL>::run(
        addr, reinterpret_cast<const typename sycl::vec<T, N>::vector_t&>(var));
}

// Usage: lscPrefetch<message_t>(ptr + local_off);
template <typename T, int N, int SubGroupSize, CacheCtrl CTL = CacheCtrl::DEFAULT>
static inline void lscPrefetch(void* addr) {
    LscPrefetch<sizeof(T), N, SubGroupSize, CTL>::run(addr);
}

// Usage: lscPrefetch(reinterpret_cast<message_t *>(ptr));
template <int SubGroupSize, CacheCtrl CTL = CacheCtrl::DEFAULT, typename T, int N>
static inline void lscPrefetch(sycl::vec<T, N>* addr) {
    LscPrefetch<sizeof(T), N, SubGroupSize, CTL>::run(addr);
}

//
// lscLoad<CacheCtrl::L1UC_L3UC>(M, address);
//
// ----------------TODO: remove these interface----------------------
template <CacheCtrl CTL = CacheCtrl::DEFAULT,
          typename T,
          int BlockHeight,
          int BlockWidth,
          DataShuffle Transpose,
          int SubGroupSize = 16>
static inline void lscLoad(__Matrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize>& M,
                           const AddressPayload<BlockHeight, BlockWidth>& address) {
    constexpr auto PhyNumRegs =
        __Matrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize>::PhyNumRegs;
    constexpr auto DataWidth =
        __Matrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize>::LSCWidth;
    RawSendLoad<DataWidth, PhyNumRegs, Transpose, CTL>::run(M.getStorage(), address);
}

template <CacheCtrl CTL = CacheCtrl::DEFAULT,
          typename T,
          int BlockHeight,
          int BlockWidth,
          DataShuffle Transpose,
          int SubGroupSize = 16>
static inline void lscStore(
    AddressPayload<BlockHeight, BlockWidth>& address,
    const __Matrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize>& M) {
    constexpr auto PhyNumRegs =
        __Matrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize>::PhyNumRegs;
    constexpr auto DataWidth = Log2<sizeof(T)>();
    static_assert(Transpose == DataShuffle::none, "Store support only none shuffled matrix");
    RawSendStore<DataWidth, PhyNumRegs, Transpose, CTL>::run(address, M.getStorage());
}

// ----------------TODO: END remove ------------------------------

template <CacheCtrl CTL = CacheCtrl::DEFAULT,
          typename T,
          int BlockHeight,
          int BlockWidth,
          DataShuffle Transpose,
          int SubGroupSize,
          int ArrayLength>
static inline void lscLoad(
    __ArrayMatrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize, ArrayLength>& M,
    const AddressPayload<BlockHeight, BlockWidth, ArrayLength>& address) {
    constexpr auto PhyNumRegs =
        __ArrayMatrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize, ArrayLength>::PhyNumRegs;
    constexpr auto DataWidth =
        __ArrayMatrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize, ArrayLength>::LSCWidth;
    RawSendLoad<DataWidth, PhyNumRegs, Transpose, CTL>::run(M.getStorage(), address);
}

template <CacheCtrl CTL = CacheCtrl::DEFAULT,
          typename T,
          int BlockHeight,
          int BlockWidth,
          DataShuffle Transpose,
          int SubGroupSize,
          int ArrayLength>
static inline void lscPrefetch(
    const AddressPayload<BlockHeight, BlockWidth, ArrayLength>& address) {
    constexpr auto DataWidth =
        __ArrayMatrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize, ArrayLength>::LSCWidth;
    RawPrefetch<DataWidth, 0, Transpose, CTL>::run(address);
}

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
template <int BlockHeight, int BlockWidth, int ArrayLength>
template <typename T, int SubGroupSize, CacheCtrl CTL>
inline void AddressPayload<BlockHeight, BlockWidth, ArrayLength>::prefetch() const {
    constexpr auto LSCWidth =
        RegisterLayout<T, BlockHeight, BlockWidth, DataShuffle::none, SubGroupSize, ArrayLength>::
            LSCWidth;
    RawPrefetch<LSCWidth, 0, DataShuffle::none, CTL>::run(*this);
}
#endif

template <CacheCtrl CTL = CacheCtrl::DEFAULT,
          typename T,
          int BlockHeight,
          int BlockWidth,
          DataShuffle Transpose,
          int SubGroupSize,
          int ArrayLength>
static inline typename std::enable_if<BlockHeight <= 16, void>::type lscStore(
    AddressPayload<BlockHeight, BlockWidth, ArrayLength>& address,
    const __ArrayMatrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize, ArrayLength>& M) {
    constexpr auto PhyNumRegs =
        __ArrayMatrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize>::PhyNumRegs;
    constexpr auto DataWidth = Log2<sizeof(T)>();
    static_assert(Transpose == DataShuffle::none, "Store support only none shuffled matrix");
    static_assert(PhyNumRegs <= 8, "Store size must <= 512 bytes");
    static_assert(ArrayLength == 1, "Store allows only single Array");
    RawSendStore<DataWidth, PhyNumRegs, Transpose, CTL>::run(address, M.getStorage());
}

//  store more than 512 bytes, will decomposed to multiple 512-byte store, and only support update y_off
template <CacheCtrl CTL = CacheCtrl::DEFAULT,
          typename T,
          int BlockHeight,
          int BlockWidth,
          DataShuffle Transpose,
          int SubGroupSize,
          int ArrayLength>
static inline
    typename std::enable_if<BlockHeight == 32 && BlockWidth == 16 && sizeof(T) == 2, void>::type
    lscStore(
        AddressPayload<BlockHeight, BlockWidth, ArrayLength>& address,
        const __ArrayMatrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize, ArrayLength>& M) {
    static_assert(Transpose == DataShuffle::none, "Store support only none shuffled matrix");
    static_assert(BlockWidth == 16, "Only support store decomposed to multiple 16x32 bytes");
    static_assert(ArrayLength == 1, "Store allows only single Array");

    constexpr auto PhyNumRegs =
        __ArrayMatrix<T, BlockHeight, BlockWidth, Transpose, SubGroupSize, ArrayLength>::PhyNumRegs;

    constexpr auto DataWidth = Log2<sizeof(T)>();
#if defined(__SYCL_DEVICE_ONLY__)
    AddressPayload<16, 16> address_for_write(address);
    RawSendStore32_32<DataWidth, PhyNumRegs, Transpose, CTL>::run(address_for_write,
                                                                  M.getStorage());
#endif
}

// TODO: START remove
template <typename T, int Height, int Width, DataShuffle Transpose, int SubGroupSize, int ArraySize>
template <CacheCtrl CTL>
inline __Matrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>&
__Matrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>::load(
    const AddressPayload<Height, Width, ArraySize>& address) {
    constexpr auto PhyNumRegs = __Matrix<T, Height, Width, Transpose, SubGroupSize>::PhyNumRegs;
    constexpr auto DataWidth = __Matrix::LSCWidth;
    RawSendLoad<DataWidth, PhyNumRegs, Transpose, CTL>::run(this->getStorage(), address);
    return *this;
}

template <typename T, int Height, int Width, DataShuffle Transpose, int SubGroupSize, int ArraySize>
template <CacheCtrl CTL>
inline __Matrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>&
__Matrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>::store(
    const AddressPayload<Height, Width, ArraySize>& address) {
    constexpr auto PhyNumRegs = __Matrix<T, Height, Width, Transpose, SubGroupSize>::PhyNumRegs;
    constexpr auto DataWidth = Log2<sizeof(T)>();
    static_assert(Transpose == DataShuffle::none, "Store support only none shuffled matrix");
    RawSendStore<DataWidth, PhyNumRegs, Transpose, CTL>::run(address, this->getStorage());
    return *this;
}
// TODO: End remove

template <typename T, int Height, int Width, DataShuffle Transpose, int SubGroupSize, int ArraySize>
template <CacheCtrl CTL>
inline __ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>&
__ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>::load(
    const AddressPayload<Height, Width, ArraySize>& address) {
    constexpr auto PhyNumRegs = __ArrayMatrix::PhyNumRegs;
    constexpr auto DataWidth = __ArrayMatrix::LSCWidth;
    RawSendLoad<DataWidth, PhyNumRegs, Transpose, CTL>::run(this->getStorage(), address);
    return *this;
}

template <typename T, int Height, int Width, DataShuffle Transpose, int SubGroupSize, int ArraySize>
template <CacheCtrl CTL>
inline __ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>&
__ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>::store(
    const AddressPayload<Height, Width, ArraySize>& address) {
    constexpr auto PhyNumRegs = __ArrayMatrix::PhyNumRegs;
    constexpr auto DataWidth = Log2<sizeof(T)>();
    static_assert(Transpose == DataShuffle::none, "Store support only none shuffled matrix");
    static_assert(ArraySize == 1, "Store allows only single Array");
    RawSendStore<DataWidth, PhyNumRegs, Transpose, CTL>::run(address, this->getStorage());
    return *this;
}

template <typename T, int Height, int Width, DataShuffle Transpose, int SubGroupSize, int ArraySize>
template <CacheCtrl CTL>
inline __RawMatrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>&
__RawMatrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>::load(
    const AddressPayload<Height, Width, ArraySize>& address) {
    constexpr auto PhyNumRegs = __RawMatrix::PhyNumRegs;
    constexpr auto DataWidth = __RawMatrix::LSCWidth;
    RawSendLoad<DataWidth, PhyNumRegs, Transpose, CTL>::run(this->getStorage(), address);
    return *this;
}

template <typename T, int Height, int Width, DataShuffle Transpose, int SubGroupSize, int ArraySize>
template <CacheCtrl CTL>
inline __RawMatrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>&
__RawMatrix<T, Height, Width, Transpose, SubGroupSize, ArraySize>::store(
    const AddressPayload<Height, Width, ArraySize>& address) {
    constexpr auto PhyNumRegs = __RawMatrix::PhyNumRegs;
    constexpr auto DataWidth = Log2<sizeof(T)>();
    static_assert(Transpose == DataShuffle::none, "Store support only none shuffled matrix");
    static_assert(ArraySize == 1, "Store allows only single Array");
    RawSendStore<DataWidth, PhyNumRegs, Transpose, CTL>::run(address, this->getStorage());
    return *this;
}
