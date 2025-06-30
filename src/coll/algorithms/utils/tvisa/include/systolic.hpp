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

#include <sycl/sycl.hpp>

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
// TODO: try in real case
void dp4a(uint32_t& dst, uint32_t& accum, uint32_t a, uint32_t b) {
    asm volatile("dp4a (M1, 16) %0 %1 %2 %3\n" : "=rw"(dst), "+rw"(accum) : "rw"(a), "rw"(b));
}
#endif

extern "C" {
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef short short8 __attribute__((ext_vector_type(8)));
typedef int int8 __attribute__((ext_vector_type(8)));

extern SYCL_EXTERNAL half8 __builtin_IB_sub_group16_fdpas_hf_hf_hf_hf_8_8(half8 acc,
                                                                          short8 a,
                                                                          int8 b);
}

namespace {
struct systolic_config {
    static constexpr int Depth = 8;
    static constexpr int Width = 16;
};

template <typename OT,
          typename AccumT,
          typename IT1,
          typename IT2 = IT1,
          int SubGroupSize = 16,
          typename config = systolic_config>
struct Dpas {
    static constexpr int Depth = config::Depth;
    static constexpr int Width = config::Width;

    static constexpr int OpsPerChan = sizeof(int) / sizeof(IT2);
    static constexpr int Src1ElemsPerChan = sizeof(int) / OpsPerChan;

    static constexpr int K = Depth * OpsPerChan;
    static constexpr int N = Width * Src1ElemsPerChan;

    // TODO: DataShuffle of A can be any
    template <int M>
    static inline void run(
        __ArrayMatrix<OT, M, N, DataShuffle::none, SubGroupSize, 1>& C, /* dst */
        __ArrayMatrix<AccumT, M, N, DataShuffle::none, SubGroupSize, 1>& Accum, /* src0 */
        __ArrayMatrix<IT1, M, K, DataShuffle::none, SubGroupSize, 1>& A, /* src2 */
        __ArrayMatrix<IT2, K, N, DataShuffle::vnni, SubGroupSize, 1>& B /* src1 */
    );

    template <int M>
    static inline void run(
        __ArrayMatrix<OT, M, N, DataShuffle::none, SubGroupSize, 1>& C, /* dst/src0 */
        __ArrayMatrix<IT1, M, K, DataShuffle::none, SubGroupSize, 1>& A, /* src2 */
        __ArrayMatrix<IT2, K, N, DataShuffle::vnni, SubGroupSize, 1>& B /* src1 */
    );

    // Interface using intrinsics
    template <int M>
    static inline void run(
        __ArrayMatrix<OT, M, N, DataShuffle::none, SubGroupSize, 1>& C, /* dst/src0 */
        __RawMatrix<IT1, M, K, DataShuffle::none, SubGroupSize, 1>& A, /* src2 */
        __RawMatrix<IT2, K, N, DataShuffle::vnni, SubGroupSize, 1>& B /* src1 */
    );
};

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)

// cover half, PVC
template <int SubGroupSize>
struct Dpas<sycl::half, sycl::half, sycl::half, sycl::half, SubGroupSize, systolic_config> {
    static constexpr int Depth = systolic_config::Depth;
    static constexpr int Width = systolic_config::Width;

    static constexpr int OpsPerChan = sizeof(int) / sizeof(sycl::half); /* 2 */
    static constexpr int Src1ElemsPerChan = 1;

    static constexpr int K = Depth * OpsPerChan; /* 16 */
    static constexpr int N = Width * Src1ElemsPerChan; /* 16 */

    template <int M>
    static inline void run(__ArrayMatrix<sycl::half, M, N, DataShuffle::none, SubGroupSize, 1>&,
                           __ArrayMatrix<sycl::half, M, N, DataShuffle::none, SubGroupSize, 1>&,
                           __ArrayMatrix<sycl::half, M, K, DataShuffle::none, SubGroupSize, 1>&,
                           __ArrayMatrix<sycl::half, K, N, DataShuffle::vnni, SubGroupSize, 1>&);

    template <int M>
    static inline void run(__ArrayMatrix<sycl::half, M, N, DataShuffle::none, SubGroupSize, 1>&,
                           __ArrayMatrix<sycl::half, M, K, DataShuffle::none, SubGroupSize, 1>&,
                           __ArrayMatrix<sycl::half, K, N, DataShuffle::vnni, SubGroupSize, 1>&);

    template <int M>
    static inline void run(__ArrayMatrix<sycl::half, M, N, DataShuffle::none, SubGroupSize, 1>&,
                           __RawMatrix<sycl::half, M, K, DataShuffle::none, SubGroupSize, 1>&,
                           __RawMatrix<sycl::half, K, N, DataShuffle::vnni, SubGroupSize, 1>&);

#define GenRepeat(M) \
    template <> \
    inline void run<M>( \
        __ArrayMatrix<sycl::half, M, N, DataShuffle::none, SubGroupSize, 1> & C, /* dst */ \
        __ArrayMatrix<sycl::half, M, N, DataShuffle::none, SubGroupSize, 1> & Accum, /* src0 */ \
        __ArrayMatrix<sycl::half, M, K, DataShuffle::none, SubGroupSize, 1> & A, /* src2 */ \
        __ArrayMatrix<sycl::half, K, N, DataShuffle::vnni, SubGroupSize, 1> & B /* src1 */ \
    ) { \
        asm volatile("{\n" \
                     ".decl aliasA v_type=G type=d num_elts=64 align=GRF alias=<%2,0>\n" \
                     ".decl aliasB v_type=G type=d num_elts=128 align=GRF alias=<%3,0>\n" \
                     "dpas.hf.hf.8." str(M) " (M1, 16) %0.0 %1.0 aliasB.0 aliasA(0, 0)\n" \
                                            "}\n" \
                     : "+rw"(C.getStorage()) \
                     : "rw"(Accum.getStorage()), "rw"(A.getStorage()), "rw"(B.getStorage())); \
    } \
\
    template <> \
    inline void run<M>( \
        __ArrayMatrix<sycl::half, M, N, DataShuffle::none, SubGroupSize, 1> & C, /* dst */ \
        __ArrayMatrix<sycl::half, M, K, DataShuffle::none, SubGroupSize, 1> & A, /* src2 */ \
        __ArrayMatrix<sycl::half, K, N, DataShuffle::vnni, SubGroupSize, 1> & B /* src1 */ \
    ) { \
        asm volatile("{\n" \
                     ".decl aliasA v_type=G type=d num_elts=64 align=GRF alias=<%1,0>\n" \
                     ".decl aliasB v_type=G type=d num_elts=128 align=GRF alias=<%2,0>\n" \
                     "dpas.hf.hf.8." str(M) " (M1, 16) %0.0 %0.0 aliasB.0 aliasA(0, 0)\n" \
                                            "}\n" \
                     : "+rw"(C.getStorage()) \
                     : "rw"(A.getStorage()), "rw"(B.getStorage())); \
    }

    GenRepeat(1);
    GenRepeat(2);
    GenRepeat(3);
    GenRepeat(4);
    // GenRepeat(5);
    // GenRepeat(6);
    GenRepeat(7);
    GenRepeat(8);
#undef GenRepeat

    template <>
    inline void run<16>(
        __ArrayMatrix<sycl::half, 16, N, DataShuffle::none, SubGroupSize, 1>& C, /* dst */
        __ArrayMatrix<sycl::half, 16, N, DataShuffle::none, SubGroupSize, 1>& Accum, /* src0 */
        __ArrayMatrix<sycl::half, 16, K, DataShuffle::none, SubGroupSize, 1>& A, /* src2 */
        __ArrayMatrix<sycl::half, K, N, DataShuffle::vnni, SubGroupSize, 1>& B /* src1 */
    ) {
        asm volatile("{\n"
                     ".decl aliasA v_type=G type=d num_elts=128 align=GRF alias=<%2,0>\n"
                     ".decl aliasB v_type=G type=d num_elts=128 align=GRF alias=<%3,0>\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.0 %1.0 aliasB.0 aliasA(0, 0)\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.256 %1.256 aliasB.0 aliasA(4, 0)\n"
                     "}\n"
                     : "+rw"(C.getStorage())
                     : "rw"(Accum.getStorage()), "rw"(A.getStorage()), "rw"(B.getStorage()));
    }

    template <>
    inline void run<16>(
        __ArrayMatrix<sycl::half, 16, N, DataShuffle::none, SubGroupSize, 1>& C, /* dst */
        __ArrayMatrix<sycl::half, 16, K, DataShuffle::none, SubGroupSize, 1>& A, /* src2 */
        __ArrayMatrix<sycl::half, K, N, DataShuffle::vnni, SubGroupSize, 1>& B /* src1 */
    ) {
        asm volatile("{\n"
                     ".decl aliasA v_type=G type=d num_elts=128 align=GRF alias=<%1,0>\n"
                     ".decl aliasB v_type=G type=d num_elts=128 align=GRF alias=<%2,0>\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 aliasB.0 aliasA(0, 0)\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.256 %0.256 aliasB.0 aliasA(4, 0)\n"
                     "}\n"
                     : "+rw"(C.getStorage())
                     : "rw"(A.getStorage()), "rw"(B.getStorage()));
    }

    template <>
    inline void run<16>(
        __ArrayMatrix<sycl::half, 16, N, DataShuffle::none, SubGroupSize, 1>& C, /* dst */
        __RawMatrix<sycl::half, 16, K, DataShuffle::none, SubGroupSize, 1>& A, /* src2 */
        __RawMatrix<sycl::half, K, N, DataShuffle::vnni, SubGroupSize, 1>& B /* src1 */
    ) {
        auto& A_0 = A.template subTileView<0, 8>();
        auto& A_1 = A.template subTileView<8, 8>();

        auto& C_0 = C.template subTileView<0, 8>();
        auto& C_1 = C.template subTileView<8, 8>();

        C_0.getStorage() = __builtin_IB_sub_group16_fdpas_hf_hf_hf_hf_8_8(
            C_0.getStorage(), A_0.getStorage(), B.getStorage());
        C_1.getStorage() = __builtin_IB_sub_group16_fdpas_hf_hf_hf_hf_8_8(
            C_1.getStorage(), A_1.getStorage(), B.getStorage());
    }

    template <>
    inline void run<32>(
        __ArrayMatrix<sycl::half, 32, N, DataShuffle::none, SubGroupSize, 1>& C, /* dst */
        __ArrayMatrix<sycl::half, 32, N, DataShuffle::none, SubGroupSize, 1>& Accum, /* src0 */
        __ArrayMatrix<sycl::half, 32, K, DataShuffle::none, SubGroupSize, 1>& A, /* src2 */
        __ArrayMatrix<sycl::half, K, N, DataShuffle::vnni, SubGroupSize, 1>& B /* src1 */
    ) {
        asm volatile("{\n"
                     ".decl aliasA v_type=G type=d num_elts=256 align=GRF alias=<%2,0>\n"
                     ".decl aliasB v_type=G type=d num_elts=128 align=GRF alias=<%3,0>\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.0 %1.0 aliasB.0 aliasA(0, 0)\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.256 %1.256 aliasB.0 aliasA(4, 0)\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.512 %1.512 aliasB.0 aliasA(8, 0)\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.768 %1.768 aliasB.0 aliasA(12, 0)\n"
                     "}\n"
                     : "+rw"(C.getStorage())
                     : "rw"(Accum.getStorage()), "rw"(A.getStorage()), "rw"(B.getStorage()));
    }

#define repeatDpas32(dstType, srcType, dstTLen, srcTLen) \
    template <> \
    inline void run<32>(__ArrayMatrix<dstType, 32, N, DataShuffle::none> & C, /* dst */ \
                        __ArrayMatrix<srcType, 32, K, DataShuffle::none> & A, /* src2 */ \
                        __ArrayMatrix<srcType, K, N, DataShuffle::vnni> & B /* src1 */ \
    ) { \
        asm volatile( \
            "{\n" \
            ".decl aliasA v_type=G type=d num_elts=" xstr( \
                32 * 16 * srcTLen / \
                4) " align=GRF alias=<%1,0>\n" \
                   ".decl aliasB v_type=G type=d num_elts=" xstr( \
                       16 * 16 * srcTLen / \
                       4) " align=GRF alias=<%2,0>\n" \
                          "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 aliasB.0 aliasA(0, 0)\n" \
                          "dpas.hf.hf.8.8 (M1, 16) %0." xstr(1 * 8 * 16 * dstTLen) " %0." xstr(1 * 8 * 16 * dstTLen) " aliasB.0 aliasA(" xstr( \
                              srcTLen * \
                              2) ", 0)\n" \
                                 "dpas.hf.hf.8.8 (M1, 16) %0." xstr(2 * 8 * 16 * dstTLen) " %0." xstr(2 * 8 * 16 * dstTLen) " aliasB.0 aliasA(" xstr( \
                                     srcTLen * \
                                     4) ", 0)\n" \
                                        "dpas.hf.hf.8.8 (M1, 16) %0." xstr(3 * 8 * 16 * dstTLen) " %0." xstr( \
                                            3 * 8 * 16 * \
                                            dstTLen) " aliasB.0 aliasA(" xstr(srcTLen * \
                                                                              6) ", 0)\n" \
                                                                                 "}\n" \
            : "+rw"(C.getStorage()) \
            : "rw"(A.getStorage()), "rw"(B.getStorage())); \
    }

    // XXX: test it
    // repeatDpas32(float, sycl::half, 4, 2);
    // repeatDpas32(sycl::half, sycl::half, 2, 2);

    template <>
    inline void run<32>(
        __ArrayMatrix<sycl::half, 32, N, DataShuffle::none, SubGroupSize, 1>& C, /* dst */
        __ArrayMatrix<sycl::half, 32, K, DataShuffle::none, SubGroupSize, 1>& A, /* src2 */
        __ArrayMatrix<sycl::half, K, N, DataShuffle::vnni, SubGroupSize, 1>& B /* src1 */
    ) {
        asm volatile("{\n"
                     ".decl aliasA v_type=G type=d num_elts=256 align=GRF alias=<%1,0>\n"
                     ".decl aliasB v_type=G type=d num_elts=128 align=GRF alias=<%2,0>\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 aliasB.0 aliasA(0, 0)\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.256 %0.256 aliasB.0 aliasA(4, 0)\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.512 %0.512 aliasB.0 aliasA(8, 0)\n"
                     "dpas.hf.hf.8.8 (M1, 16) %0.768 %0.768 aliasB.0 aliasA(12, 0)\n"
                     "}\n"
                     : "+rw"(C.getStorage())
                     : "rw"(A.getStorage()), "rw"(B.getStorage()));
    }
};

// using bf16 = sycl::ext::oneapi::bfloat16;
//
// // cover brain-float, PVC
// template <typename OT, typename AccumT>
// struct Dpas<OT, AccumT, bf16, bf16, systolic_config> {
//   static constexpr int Depth = systolic_config::Depth;
//   static constexpr int Width = systolic_config::Width;
//
//   static constexpr int OpsPerChan = sizeof(int) / sizeof(sycl::half); /* 2 */
//   static constexpr int Src1ElemsPerChan = 1;
//
//   static constexpr int K = Depth * OpsPerChan; /* 16 */
//   static constexpr int N = Width * Src1ElemsPerChan; /* 16 */
//
//   template <int M> static inline void run(
//       __ArrayMatrix<OT, M, N, DataShuffle::none>&,
//       __ArrayMatrix<AccumT, M, N, DataShuffle::none>&,
//       __ArrayMatrix<bf16, M, K, DataShuffle::none>&,
//       __ArrayMatrix<bf16, K, N, DataShuffle::vnni>&
//   );
//
// #define GenRepeat(M)  \
//   template <> static inline void run<M>(  \
//       __ArrayMatrix<OT, M, N, DataShuffle::none>& C,   /* dst */  \
//       __ArrayMatrix<AccumT, M, N, DataShuffle::none>& Accum, /* src0 */ \
//       __ArrayMatrix<bf16, M, K, DataShuffle::none>& A,  /* src2 */ \
//       __ArrayMatrix<bf16, K, N, DataShuffle::vnni>& B   /* src1 */ \
//   ) { \
//     asm volatile ("\n"  \
//         "dpas.bf.bf.8." str(M) " (M1, 16) %0.0 %1.0 %3.0 %2(0, 0)\n"  \
//         : "=rw"(C.getStorage()): "rw"(Accum.getStorage())  \
//         "rw"(A.getStorage()), "rw"(B.getStorage())  \
//     );  \
//   }
//
//   GenRepeat(1);
//   GenRepeat(2);
//   GenRepeat(3);
//   GenRepeat(4);
//   // GenRepeat(5);
//   // GenRepeat(6);
//   GenRepeat(7);
//   GenRepeat(8);
// #undef GenRepeat
// };

#endif
} // namespace

template <int M,
          int K,
          int N,
          typename OT,
          typename AccumT,
          typename IT1,
          typename IT2,
          int SubGroupSize = 16,
          typename config = systolic_config>
static inline void dpas(__ArrayMatrix<OT, M, N, DataShuffle::none, SubGroupSize, 1>& D,
                        __ArrayMatrix<AccumT, M, N, DataShuffle::none, SubGroupSize, 1>& C,
                        __ArrayMatrix<IT1, M, K, DataShuffle::none, SubGroupSize, 1>& A,
                        __ArrayMatrix<IT2, K, N, DataShuffle::vnni, SubGroupSize, 1>& B) {
    // TODO: check accepted parameters with static assert;
    Dpas<OT, AccumT, IT1, IT2, SubGroupSize, systolic_config>::template run<M>(D, C, A, B);
}

template <int M,
          int K,
          int N,
          typename OT,
          typename IT1,
          typename IT2,
          int SubGroupSize = 16,
          typename config = systolic_config>
static inline void dpas(__ArrayMatrix<OT, M, N, DataShuffle::none, SubGroupSize, 1>& C,
                        __ArrayMatrix<IT1, M, K, DataShuffle::none, SubGroupSize, 1>& A,
                        __ArrayMatrix<IT2, K, N, DataShuffle::vnni, SubGroupSize, 1>& B) {
    // TODO: check accepted parameters with static assert;
    Dpas<OT, OT, IT1, IT2, SubGroupSize, systolic_config>::template run<M>(C, A, B);
}

template <int M,
          int K,
          int N,
          typename OT,
          typename IT1,
          typename IT2,
          int SubGroupSize = 16,
          typename config = systolic_config>
static inline void dpas(__ArrayMatrix<OT, M, N, DataShuffle::none, SubGroupSize, 1>& C,
                        __RawMatrix<IT1, M, K, DataShuffle::none, SubGroupSize, 1>& A,
                        __RawMatrix<IT2, K, N, DataShuffle::vnni, SubGroupSize, 1>& B) {
    // TODO: check accepted parameters with static assert;
    static_assert(SubGroupSize == 16, "SubGroupSize for dpas must be 16");
    Dpas<OT, OT, IT1, IT2, SubGroupSize, systolic_config>::template run<M>(C, A, B);
}
