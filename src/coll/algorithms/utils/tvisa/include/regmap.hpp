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

#include "CL/cl_platform.h"
#include <sycl/sycl.hpp>
#include <type_traits>
#include "set_zero.hpp"

// TODO: move somewhere else
template <int N>
constexpr int Log2Ceiling() {
    return Log2Ceiling<N / 2>() + 1;
}

template <>
constexpr int Log2Ceiling<2>() {
    return 1;
}

template <>
constexpr int Log2Ceiling<1>() {
    return 1;
}

template <int N>
constexpr int Log2() {
    return Log2<N / 2>() + 1;
}

template <>
constexpr int Log2<4>() {
    return 2;
}
template <>
constexpr int Log2<2>() {
    return 1;
}
template <>
constexpr int Log2<1>() {
    return 0;
}

template <int N, int L>
constexpr int LowBound() {
    if constexpr (N < L)
        return L;
    else
        return N;
}

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)

//
// XXX: AddressPayload is reinterpretable among configurations
//  1. Surface Pitch is in byte
//  2. Surface Width is in byte
//  3. Surface Height
//  4. StartX/StartY and Block Width/Height are in elements.
//
template <int BlockHeight, int BlockWidth, int ArrayLength = 1>
struct AddressPayload {
    inline AddressPayload() = default;
    inline AddressPayload(const void* const SurfaceBase,
                          uint32_t SurfaceHeight,
                          uint32_t SurfaceWidth,
                          uint32_t SurfacePitch,
                          int Src0AddrX,
                          int Src0AddrY) {
        constexpr uint32_t BWHAL =
            (BlockWidth - 1) | (BlockHeight - 1) << 8 | ((ArrayLength - 1) & 0xf) << 16;

        asm volatile("{\n"
                     ".decl alias64 v_type=G type=uq num_elts=8 align=GRF alias=<%0, 0>\n"
                     "mov (M1, 1) alias64(0, 0)<1> %1(0, 0)<0;1,0>\n"
                     "add (M1, 1) %0(0, 2)<1> %2(0, 0)<0;1,0> -1:ud\n"
                     "add (M1, 1) %0(0, 3)<1> %3(0, 0)<0;1,0> -1:ud\n"
                     "add (M1, 1) %0(0, 4)<1> %4(0, 0)<0;1,0> -1:ud\n"
                     "mov (M1, 1) %0(0, 5)<1> %5(0, 0)<0;1,0> \n"
                     "mov (M1, 1) %0(0, 6)<1> %6(0, 0)<0;1,0> \n"
                     "mov (M1, 1) %0(0, 7)<1> %7\n"
                     "}\n"
                     : "=rw"(payloadReg_)
                     : "rw"(SurfaceBase),
                       "rw"(SurfaceWidth),
                       "rw"(SurfaceHeight),
                       "rw"(SurfacePitch),
                       "rw"(Src0AddrX),
                       "rw"(Src0AddrY),
                       "i"(BWHAL)
                     /*
        : "+rw"(payloadReg_) : "rw.u"(SurfaceBase), "rw.u"(SurfaceWidth),
        "rw.u"(SurfaceHeight), "rw.u"(SurfacePitch), "rw.u"(Src0AddrX),
        "rw.u"(Src0AddrY), "i"(BWHAL)
        */
        );
    }

    template <int OldHeight, int OldWidth, int OldArrayLength>
    inline AddressPayload(const AddressPayload<OldHeight, OldWidth, OldArrayLength>& payload)
            : payloadReg_(payload.getPayload()) {
        if constexpr (OldWidth != BlockWidth || OldHeight != BlockHeight ||
                      OldArrayLength != ArrayLength) {
            constexpr uint32_t BWHAL =
                (BlockWidth - 1) | (BlockHeight - 1) << 8 | ((ArrayLength - 1) & 0xf) << 16;
            asm volatile("mov (M1,1) %0(0,7)<1> %1\n" : "+rw"(payloadReg_) : "i"(BWHAL));
        }
    }

    template <int OldHeight, int OldWidth, int OldArrayLength>
    inline AddressPayload& operator=(
        const AddressPayload<OldHeight, OldWidth, OldArrayLength>& payload) {
        payloadReg_ = payload.getPayload();
        if constexpr (OldWidth != BlockWidth || OldHeight != BlockHeight ||
                      OldArrayLength != ArrayLength) {
            constexpr uint32_t BWHAL =
                (BlockWidth - 1) | (BlockHeight - 1) << 8 | ((ArrayLength - 1) & 0xf) << 16;
            asm volatile("mov (M1,1) %0(0,7)<1> %1\n" : "+rw"(payloadReg_) : "i"(BWHAL));
        }
        return *this;
    }

    inline uint32_t& getPayload() {
        return payloadReg_;
    }

    inline const uint32_t& getPayload() const {
        return payloadReg_;
    }

    inline AddressPayload& UpdateSrc0AddrX(int offset) {
        asm volatile("mov (M1, 1) %0(0, 5)<1> %1(0, 0)<0;1,0> \n"
                     : "+rw"(payloadReg_)
                     : "rw"(offset));
        return *this;
    }

    inline AddressPayload& UpdateSrc0AddrY(int offset) {
        asm volatile("mov (M1, 1) %0(0, 6)<1> %1(0, 0)<0;1,0> \n"
                     : "+rw"(payloadReg_)
                     : "rw"(offset));
        return *this;
    }

    inline AddressPayload& addSrc0AddrX(int offset) {
        asm volatile("add (M1, 1) %0(0, 5)<1> %0(0, 5)<0;1,0> %1(0,0)<0;1,0>\n"
                     : "+rw"(payloadReg_)
                     : "rw"(offset));
        return *this;
    }

    inline AddressPayload& addSrc0AddrY(int offset) {
        asm volatile("add (M1, 1) %0(0, 6)<1> %0(0, 6)<0;1,0> %1(0,0)<0;1,0>\n"
                     : "+rw"(payloadReg_)
                     : "rw"(offset));
        return *this;
    }

    // XXX: potential problem in final code! IGC crash on DESSA pass
    inline AddressPayload& updateSurfaceBase(void* addr) {
        asm volatile("{\n"
                     ".decl alias64 v_type=G type=uq num_elts=8 align=GRF alias=<%0, 0>\n"
                     "mov (M1,1) alias64(0, 0)<1> %1(0, 0)<0;1,0>\n"
                     // "mov (M1,16) %0(0,0)<1> %0(0,0)<1;1,0>\n"
                     "}\n"
                     : "+rw"(payloadReg_)
                     : "rw"(addr));
        return *this;
    }

    inline AddressPayload cloneAndUpdateSurfaceBase(void* addr) {
        AddressPayload newPayload(*this);
        return newPayload.updateSurfaceBase(addr);
    }

    inline AddressPayload& updateSrc0AddrX(int x_off) {
        asm volatile("mov (M1, 1) %0(0, 5)<1> %1(0, 0)<0;1,0>\n"
                     : "+rw"(payloadReg_)
                     : "rw"(x_off));
        return *this;
    }

    inline AddressPayload& updateSrc0AddrY(int y_off) {
        asm volatile("mov (M1, 1) %0(0, 6)<1> %1(0, 0)<0;1,0>\n"
                     : "+rw"(payloadReg_)
                     : "rw"(y_off));
        return *this;
    }

    template <typename T, int SubGroupSize = 16, CacheCtrl CTL = CacheCtrl::DEFAULT>
    inline void prefetch() const;

private:
    uint32_t payloadReg_;
};

struct BarrierPayload {
    BarrierPayload(uint8_t ID, BarrierType type, uint8_t NumOfProducer, uint8_t NumOfConsumer) {
        uint32_t payload = ID | ((uint32_t)type) << 14 | ((uint32_t)NumOfProducer) << 16 |
                           ((uint32_t)NumOfConsumer) << 24;

        asm volatile("\n"
                     "mov (M1, 1) %0(0,2)<1> %1(0,0)<0;1,0>\n"
                     : "=rw"(payloadReg_)
                     : "rw"(payload));
    }

    inline uint32_t& getPayload() {
        return payloadReg_;
    }

    inline const uint32_t& getPayload() const {
        return payloadReg_;
    }

private:
    uint32_t payloadReg_;
};

#else
template <int BlockHeight, int BlockWidth, int ArrayLength = 1>
struct AddressPayload;

struct BarrierPayload;
#endif

template <typename T,
          int Height,
          int Width,
          DataShuffle Transpose,
          int SubGroupSize = 16,
          int ArraySize = 1>
struct RegisterLayout;

template <typename T, int Height, int Width, int SubGroupSize, int ArraySize>
struct RegisterLayout<T, Height, Width, DataShuffle::none, SubGroupSize, ArraySize> {
protected:
    static constexpr int NElemsPerLane = sizeof(int) / sizeof(T);
    static constexpr int PaddedWidth = 1 << Log2Ceiling<sizeof(T) * Width>();
    static constexpr int RegSize = SubGroupSize * sizeof(int);
    static constexpr int PhyRegSize = 64;
    static constexpr int AllocSize = PaddedWidth * Height * ArraySize;

public:
    static constexpr int NumRegs = (AllocSize + RegSize - 1) / RegSize;
    static constexpr int PhyNumRegs = (AllocSize + PhyRegSize - 1) / PhyRegSize;
    static constexpr int N = NumRegs * NElemsPerLane;
    static constexpr int LSCWidth = Log2<sizeof(T)>();
};

//
// Transpose is always on the granularity of equal or more than 4-byte
//
template <typename T, int Height, int Width, int SubGroupSize, int ArraySize>
struct RegisterLayout<T, Height, Width, DataShuffle::transpose, SubGroupSize, ArraySize> {
protected:
    static constexpr int NElemsPerLane = sizeof(int) / sizeof(T);
    static constexpr int PaddedHeight = LowBound<1 << Log2Ceiling<sizeof(T) * Height>(), 4>();
    static constexpr int RegSize = SubGroupSize * sizeof(int);
    static constexpr int AllocSize = Width * PaddedHeight * ArraySize;
    static constexpr int PhyRegSize = 64;
    static_assert(Height * sizeof(T) <= PhyRegSize, "No register space for transpose");

public:
    static constexpr int NumRegs = (AllocSize + RegSize - 1) / RegSize;
    static constexpr int PhyNumRegs = (AllocSize + PhyRegSize - 1) / PhyRegSize;
    static constexpr int N = NumRegs * NElemsPerLane;
    static constexpr int LSCWidth = LowBound<Log2<sizeof(T)>(), 2>();
};

template <typename T, int Height, int Width, int SubGroupSize, int ArraySize>
struct RegisterLayout<T, Height, Width, DataShuffle::vnni, SubGroupSize, ArraySize> {
protected:
    static constexpr int NElemsPerLane = sizeof(int) / sizeof(T);
    static constexpr int PaddedHeight =
        (Height + NElemsPerLane - 1) / NElemsPerLane * NElemsPerLane;
    // We eliminate the case implicit operation is carried.
    static_assert(Height == PaddedHeight, "Error: HeightxElemsize must align to 4 bytes");
    static constexpr int PaddedWidth = 1 << Log2Ceiling<sizeof(T) * Width>();
    static constexpr int RegSize = SubGroupSize * sizeof(int);
    static constexpr int AllocSize = PaddedWidth * PaddedHeight * ArraySize;
    static constexpr int PhyRegSize = 64;
    static_assert(Width * sizeof(int) <= PhyRegSize, "No register space for VNNI transform");

public:
    static constexpr int NumRegs = (AllocSize + RegSize - 1) / RegSize;
    static constexpr int PhyNumRegs = (AllocSize + PhyRegSize - 1) / PhyRegSize;
    static constexpr int N = NumRegs * NElemsPerLane;
    static constexpr int LSCWidth = Log2<sizeof(T)>();
    static_assert(LSCWidth < 2, "VNNI only makes sense on less than 16-bit");
};

//
// memory region represents as regsiter image
//
template <typename T,
          int Height,
          int Width,
          DataShuffle Transpose = DataShuffle::none,
          int SubGroupSize = 16,
          int ArraySize = 1>
struct __Matrix {
    using layout = RegisterLayout<T, Height, Width, Transpose, SubGroupSize, ArraySize>;
    static constexpr int PhyNumRegs = layout::PhyNumRegs;
    static constexpr int NumRegs = layout::NumRegs;
    static constexpr int N = layout::N;
    static constexpr int LSCWidth = layout::LSCWidth;

    static_assert(sizeof(sycl::vec<T, N>) / sizeof(uint32_t) == NumRegs);
    using rawType = sycl::vec<uint32_t, sizeof(sycl::vec<T, N>) / sizeof(uint32_t)>;

    inline typename sycl::vec<T, N>::vector_t& getStorage() {
        return reinterpret_cast<typename sycl::vec<T, N>::vector_t&>(registerImage_);
    }
    inline const typename sycl::vec<T, N>::vector_t& getStorage() const {
        return reinterpret_cast<const typename sycl::vec<T, N>::vector_t&>(registerImage_);
    }
    inline typename sycl::vec<T, N>& getImage() {
        return registerImage_;
    }
    inline const typename sycl::vec<T, N>& getImage() const {
        return registerImage_;
    }

    /* XXX: Generate unecessary copy ??? */
    // inline typename rawType::vector_t& getRawStorage() {
    //   return reinterpret_cast<typename rawType::vector_t&>(registerImage_);
    // }
    // inline const typename rawType::vector_t& getRawStorage() const {
    //   return reinterpret_cast<const typename rawType::vector_t&>(registerImage_);
    // }

    __Matrix() = default;
    __Matrix(const sycl::vec<T, N>& rh) : registerImage_(rh) {}
    __Matrix(sycl::vec<T, N>&& rh) : registerImage_(std::move(rh)) {}
    __Matrix(const __Matrix& rh) : registerImage_(rh.registerImage_) {}
    __Matrix(__Matrix&& rh) : registerImage_(std::move(rh.registerImage_)) {}

    template <typename SrcT>
    __Matrix(const __Matrix<SrcT, Height, Width, Transpose, SubGroupSize, ArraySize>& rh) {
#pragma unroll
        for (int i = 0; i < N; ++i)
            registerImage_[i] = static_cast<T>((rh.getImage())[i]);
    }

    inline __Matrix& operator=(__Matrix&& rh) {
        registerImage_ = std::move(rh.registerImage_);
        return *this;
    }
    inline __Matrix& operator=(const __Matrix& rh) {
        registerImage_ = rh.registerImage_;
        return *this;
    }
    inline __Matrix operator+(const __Matrix& m) {
        return { registerImage_ + m.registerImage_ };
    }
    inline __Matrix operator-(const __Matrix& m) {
        return { registerImage_ - m.registerImage_ };
    }
    inline __Matrix operator*(const __Matrix& m) {
        return { registerImage_ * m.registerImage_ };
    }
    inline __Matrix operator/(const __Matrix& m) {
        return { registerImage_ * m.registerImage_ };
    }

    template <CacheCtrl CTL = CacheCtrl::DEFAULT>
    inline __Matrix& load(const AddressPayload<Height, Width, ArraySize>& address);

    template <CacheCtrl CTL = CacheCtrl::DEFAULT>
    inline __Matrix& store(const AddressPayload<Height, Width, ArraySize>& address);

    // Review as re-shuffled tensor in logic view, need careful review here.
    template <DataShuffle reshuffle>
    inline typename std::enable_if<
        Transpose == DataShuffle::vnni && reshuffle == DataShuffle::none,
        __Matrix<T, Height / 2, Width * 2, reshuffle, SubGroupSize, ArraySize>>::type
    cast() {
        return { registerImage_ };
    }

    template <DataShuffle reshuffle>
    inline typename std::enable_if<
        Transpose == DataShuffle::transpose && reshuffle == DataShuffle::none,
        __Matrix<T, Width, Height, reshuffle, SubGroupSize, ArraySize>>::type
    cast() {
        return { registerImage_ };
    }

    template <DataShuffle reshuffle>
    inline typename std::enable_if<
        Transpose == DataShuffle::none && reshuffle == DataShuffle::vnni,
        __Matrix<T, Height * 2, Width / 2, reshuffle, SubGroupSize, ArraySize>>::type
    cast() {
        return { registerImage_ };
    }

    template <DataShuffle reshuffle>
    inline typename std::enable_if<
        Transpose == DataShuffle::none && reshuffle == DataShuffle::transpose,
        __Matrix<T, Width, Height, DataShuffle::transpose, SubGroupSize, ArraySize>>::type
    cast() {
        return { registerImage_ };
    }

    //
    // XXX: performance problem for SIMD16/sycl::half when compiler generate
    // two instructions for single register operation.
    //
    inline void zero() {
        registerImage_ = 0;
    }

private:
    sycl::vec<T, N> registerImage_;
};

//
// Test for normal array type instead of vector type
//

template <typename T>
struct StorageScalarType {
    typedef T type;
};

template <>
struct StorageScalarType<sycl::half> {
    typedef _Float16 type;
};

template <DataShuffle Tranpose>
struct rawMatrixScalarType;
template <>
struct rawMatrixScalarType<DataShuffle::none> {
    typedef short type;
};
template <>
struct rawMatrixScalarType<DataShuffle::vnni> {
    typedef int type;
};

template <typename T,
          int Height,
          int Width,
          DataShuffle Transpose = DataShuffle::none,
          int SubGroupSize = 16,
          int ArraySize = 1>
struct __ArrayMatrix {
    using layout = RegisterLayout<T, Height, Width, Transpose, SubGroupSize, ArraySize>;
    static constexpr int PhyNumRegs = layout::PhyNumRegs;
    static constexpr int NumRegs = layout::NumRegs;
    static constexpr int N = layout::N;
    static constexpr int LSCWidth = layout::LSCWidth;

    static_assert(NumRegs == sizeof(T) * N / sizeof(uint32_t));

    using scalarType = typename StorageScalarType<T>::type;
    using storageType = scalarType __attribute__((ext_vector_type(N)));

    inline storageType& getStorage() {
        return registerImage_;
    }
    inline const storageType& getStorage() const {
        return registerImage_;
    }

    __ArrayMatrix() = default;
    __ArrayMatrix(const storageType& rh) : registerImage_(rh) {}
    __ArrayMatrix(storageType&& rh) : registerImage_(std::move(rh)) {}
    __ArrayMatrix(const __ArrayMatrix& rh) : registerImage_(rh.registerImage_) {}

    inline __ArrayMatrix& operator=(const __ArrayMatrix& rh) {
        registerImage_ = rh.registerImage_;
        return *this;
    }
    inline __ArrayMatrix& operator=(__ArrayMatrix&& rh) {
        registerImage_ = std::move(rh.registerImage_);
        return *this;
    }

    inline __ArrayMatrix operator+(const __ArrayMatrix& m) {
        return { registerImage_ + m.registerImage_ };
    }
    inline __ArrayMatrix operator-(const __ArrayMatrix& m) {
        return { registerImage_ - m.registerImage_ };
    }
    inline __ArrayMatrix operator*(const __ArrayMatrix& m) {
        return { registerImage_ * m.registerImage_ };
    }
    inline __ArrayMatrix operator/(const __ArrayMatrix& m) {
        return { registerImage_ * m.registerImage_ };
    }

    template <CacheCtrl CTL = CacheCtrl::DEFAULT>
    inline __ArrayMatrix& load(const AddressPayload<Height, Width, ArraySize>& address);

    template <CacheCtrl CTL = CacheCtrl::DEFAULT>
    inline __ArrayMatrix& store(const AddressPayload<Height, Width, ArraySize>& address);

    template <int ArrayOff, int newArraySize = 1>
    inline auto& subArrayView() {
        static_assert(ArrayOff + newArraySize <= ArraySize);

        using __newMatrixType =
            __ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, newArraySize>;
        using newStorageType = typename __newMatrixType::storageType;

        auto& splitImage =
            reinterpret_cast<newStorageType(&)[sizeof(storageType) / sizeof(newStorageType)]>(
                registerImage_);

        return reinterpret_cast<__newMatrixType&>(splitImage[ArrayOff]);
    }

    // Another form???
    template <int newArraySize = 1>
    inline auto& subArrayView(int ArrayOff) {
        using newStorageType =
            typename __ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, newArraySize>::
                storageType;

        auto& splitImage =
            reinterpret_cast<newStorageType(&)[sizeof(storageType) / sizeof(newStorageType)]>(
                registerImage_);

        return reinterpret_cast<
            __ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, newArraySize>&>(
            splitImage[ArrayOff]);
    }

    // Becareful, view can only start on register boundary
    template <int RowStart, int RowHeight>
    inline auto& subTileView() {
        static_assert(RowStart + RowHeight <= Height, "View out of bound.");
        static_assert(ArraySize == 1, "Can't view subtiles when arraysize is larger than 1.");

        using __newMatrixType =
            __ArrayMatrix<T, RowHeight, Width, Transpose, SubGroupSize, ArraySize>;
        using newStorageType = typename __newMatrixType::storageType;
        constexpr auto ChunkSize = __newMatrixType::N;

        static_assert(N % ChunkSize == 0, "Unevenly divided view to subtile was not supported");

        constexpr auto Offset =
            RowStart == 0
                ? 0
                : RegisterLayout<T, RowStart, Width, Transpose, SubGroupSize, ArraySize>::N;

        static_assert(Offset % ChunkSize == 0, "Unaligned subtile offset was not supported");
        constexpr auto ChunkOff = Offset / ChunkSize;

        auto& splitImage =
            reinterpret_cast<newStorageType(&)[sizeof(storageType) / sizeof(newStorageType)]>(
                registerImage_);

        return reinterpret_cast<
            __ArrayMatrix<T, RowHeight, Width, Transpose, SubGroupSize, ArraySize>&>(
            splitImage[ChunkOff]);
    }

    template <typename NewT>
    inline auto& viewAs() {
        constexpr int NewWidth =
            sizeof(NewT) < sizeof(T) ? Width * sizeof(T) / sizeof(NewT) : Width;
        constexpr int NewHeight =
            sizeof(NewT) < sizeof(T) ? Height : Height * sizeof(T) / sizeof(NewT);

        using NewType =
            __ArrayMatrix<NewT, NewHeight, NewWidth, Transpose, SubGroupSize, ArraySize>;

        static_assert(sizeof(NewType) == sizeof(*this), "View size is different from original!");

        return reinterpret_cast<NewType&>(*this);
    }

    // Review as re-shuffled tensor in logic view, need careful review here.
    template <DataShuffle reshuffle>
    inline typename std::enable_if<
        Transpose == DataShuffle::vnni && reshuffle == DataShuffle::none,
        __ArrayMatrix<T, Height / 2, Width * 2, reshuffle, SubGroupSize, ArraySize>>::type
    cast() {
        return { registerImage_ };
    }

    template <DataShuffle reshuffle>
    inline typename std::enable_if<
        Transpose == DataShuffle::transpose && reshuffle == DataShuffle::none,
        __ArrayMatrix<T, Width, Height, reshuffle, SubGroupSize, ArraySize>>::type
    cast() {
        return { registerImage_ };
    }

    template <DataShuffle reshuffle>
    inline typename std::enable_if<
        Transpose == DataShuffle::none && reshuffle == DataShuffle::vnni,
        __ArrayMatrix<T, Height * 2, Width / 2, reshuffle, SubGroupSize, ArraySize>>::type
    cast() {
        return { registerImage_ };
    }

    template <DataShuffle reshuffle>
    inline typename std::enable_if<
        Transpose == DataShuffle::none && reshuffle == DataShuffle::transpose,
        __ArrayMatrix<T, Width, Height, DataShuffle::transpose, SubGroupSize, ArraySize>>::type
    cast() {
        return { registerImage_ };
    }

    inline void zero() {
        memset(registerImage_, 0, sizeof(registerImage_));
    }

private:
    storageType registerImage_;
};

//
// __RawMatrix for workaround IGC dpas type
//
template <typename T,
          int Height,
          int Width,
          DataShuffle Transpose = DataShuffle::none,
          int SubGroupSize = 16,
          int ArraySize = 1>
struct __RawMatrix {
    using layout = RegisterLayout<T, Height, Width, Transpose, SubGroupSize, ArraySize>;
    static constexpr int PhyNumRegs = layout::PhyNumRegs;
    static constexpr int NumRegs = layout::NumRegs;
    static constexpr int N = layout::N;
    static constexpr int LSCWidth = layout::LSCWidth;

    using scalarType = typename rawMatrixScalarType<Transpose>::type;
    static constexpr int r_N = N * sizeof(T) / sizeof(scalarType);
    static_assert(r_N > 0, "In RawMatrix, Something went wrong");
    using storageType = scalarType __attribute__((ext_vector_type(r_N)));

    inline storageType& getStorage() {
        return registerImage_;
    }
    inline const storageType& getStorage() const {
        return registerImage_;
    }

    __RawMatrix() = default;
    __RawMatrix(storageType& rh) : registerImage_(rh) {}
    __RawMatrix(storageType&& rh) : registerImage_(std::move(rh)) {}
    __RawMatrix(const __RawMatrix& rh) : registerImage_(rh.registerImage_) {}
    __RawMatrix(__RawMatrix&& rh) : registerImage_(std::move(rh.registerImage_)) {}

    inline __RawMatrix& operator=(__RawMatrix&& rh) {
        registerImage_ = std::move(rh.registerImage_);
        return *this;
    }
    inline __RawMatrix& operator=(const __RawMatrix& rh) {
        registerImage_ = rh.registerImage_;
        return *this;
    }

    template <int ArrayOff, int newArraySize = 1>
    inline auto& subArrayView() {
        static_assert(ArrayOff + newArraySize <= ArraySize);

        using __newMatrixType =
            __RawMatrix<T, Height, Width, Transpose, SubGroupSize, newArraySize>;
        using newStorageType = typename __newMatrixType::storageType;

        auto& splitImage =
            reinterpret_cast<newStorageType(&)[sizeof(storageType) / sizeof(newStorageType)]>(
                registerImage_);

        return reinterpret_cast<__newMatrixType&>(splitImage[ArrayOff]);
    }

    template <int RowStart, int RowHeight>
    inline auto& subTileView() {
        static_assert(RowStart + RowHeight <= Height, "View out of bound.");
        static_assert(ArraySize == 1, "Can't view subtiles when arraysize is larger than 1.");

        using __newMatrixType =
            __RawMatrix<T, RowHeight, Width, Transpose, SubGroupSize, ArraySize>;
        using newStorageType = typename __newMatrixType::storageType;
        constexpr auto ChunkSize = __newMatrixType::r_N;
        static_assert(r_N % ChunkSize == 0, "Unevenly divided view to subtile was not supported");
        constexpr auto __Offset =
            RowStart == 0
                ? 0
                : RegisterLayout<T, RowStart, Width, Transpose, SubGroupSize, ArraySize>::N;
        constexpr auto Offset = __Offset * sizeof(T) / sizeof(scalarType);

        static_assert(Offset % ChunkSize == 0, "Unaligned offset to subtile was not supported");
        constexpr auto ChunkOff = Offset / ChunkSize;

        auto& splitImage =
            reinterpret_cast<newStorageType(&)[sizeof(storageType) / sizeof(newStorageType)]>(
                registerImage_);

        return reinterpret_cast<
            __RawMatrix<T, RowHeight, Width, Transpose, SubGroupSize, ArraySize>&>(
            splitImage[ChunkOff]);
    }

    template <CacheCtrl CTL = CacheCtrl::DEFAULT>
    inline __RawMatrix& load(const AddressPayload<Height, Width, ArraySize>& address);

    template <CacheCtrl CTL = CacheCtrl::DEFAULT>
    inline __RawMatrix& store(const AddressPayload<Height, Width, ArraySize>& address);

private:
    storageType registerImage_;
};

template <typename T,
          int Height,
          int Width,
          DataShuffle Transpose = DataShuffle::none,
          int SubGroupSize = 16>
struct Concat {
    static inline __ArrayMatrix<T, Height, Width * 2, Transpose, SubGroupSize, 1> run(
        const __ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, 1>& M_0,
        const __ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, 1>& M_1);
};

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
template <>
struct Concat<sycl::half, 8, 16, DataShuffle::none, 16> {
    static inline __ArrayMatrix<sycl::half, 8, 32, DataShuffle::none, 16, 1> run(
        const __ArrayMatrix<sycl::half, 8, 16, DataShuffle::none, 16, 1>& M_0,
        const __ArrayMatrix<sycl::half, 8, 16, DataShuffle::none, 16, 1>& M_1) {
        __ArrayMatrix<sycl::half, 8, 32, DataShuffle::none, 16, 1> Output;
        asm volatile("{\n"
                     ".decl alias_0f v_type=G type=hf num_elts=128 align=wordx32 alias=<%1, 0>\n"
                     ".decl alias_0w v_type=G type=uw num_elts=128 align=wordx32 alias=<%1, 0>\n"
                     ".decl alias_1f v_type=G type=hf num_elts=128 align=wordx32 alias=<%2, 0>\n"
                     ".decl alias_1w v_type=G type=uw num_elts=128 align=wordx32 alias=<%2, 0>\n"
                     ".decl aw v_type=G type=uw num_elts=128 align=wordx32 alias=<%0, 0>\n"
                     "mov (M1, 16) %0(0,0)<1>  alias_0f(0,0)<1;1,0>\n"
                     "mov (M1, 16) aw(1,0)<1>  alias_0w(0,16)<1;1,0>\n"
                     "mov (M1, 16) %0(2,0)<1>  alias_0f(1,0)<1;1,0>\n"
                     "mov (M1, 16) aw(3,0)<1>  alias_0w(1,16)<1;1,0>\n"
                     "mov (M1, 16) %0(4,0)<1>  alias_0f(2,0)<1;1,0>\n"
                     "mov (M1, 16) aw(5,0)<1>  alias_0w(2,16)<1;1,0>\n"
                     "mov (M1, 16) %0(6,0)<1>  alias_0f(3,0)<1;1,0>\n"
                     "mov (M1, 16) aw(7,0)<1>  alias_0w(3,16)<1;1,0>\n"
                     "mov (M1, 16) %0(0,16)<1> alias_1f(0,0)<1;1,0>\n"
                     "mov (M1, 16) aw(1,16)<1> alias_1w(0,16)<1;1,0>\n"
                     "mov (M1, 16) %0(2,16)<1> alias_1f(1,0)<1;1,0>\n"
                     "mov (M1, 16) aw(3,16)<1> alias_1w(1,16)<1;1,0>\n"
                     "mov (M1, 16) %0(4,16)<1> alias_1f(2,0)<1;1,0>\n"
                     "mov (M1, 16) aw(5,16)<1> alias_1w(2,16)<1;1,0>\n"
                     "mov (M1, 16) %0(6,16)<1> alias_1f(3,0)<1;1,0>\n"
                     "mov (M1, 16) aw(7,16)<1> alias_1w(3,16)<1;1,0>\n"
                     "\n}"
                     : "=rw"(Output.getStorage())
                     : "rw"(M_0.getStorage()), "rw"(M_1.getStorage()));
        return Output;
    }
};
#endif

template <typename T,
          int Height,
          int Width,
          DataShuffle Transpose = DataShuffle::none,
          int SubGroupSize = 16>
static inline __ArrayMatrix<T, Height, Width * 2, Transpose, SubGroupSize, 1> concat(
    const __ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, 1>& M_0,
    const __ArrayMatrix<T, Height, Width, Transpose, SubGroupSize, 1>& M_1) {
    return Concat<T, Height, Width, Transpose, SubGroupSize>::run(M_0, M_1);
}
