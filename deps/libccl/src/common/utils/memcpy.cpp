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

#include "common/log/log.hpp"
#include "common/utils/memcpy.hpp"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>

namespace ccl {

__attribute__((__always_inline__)) inline int is_nts_supported() {
#ifdef CCL_AVX_COMPILER
    static int is_avx_enabled = -1;
    if (is_avx_enabled == -1) {
        uint32_t reg[4];
        /* AVX capabilities for NTS implementation */
        /* CPUID.(EAX=01H):ECX.AVX [bit 28] */
        __asm__ __volatile__("cpuid"
                             : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                             : "a"(1));
        is_avx_enabled = ((reg[2] & (1u << 28)) >> 28);
        LOG_DEBUG("AVX enabled: ", is_avx_enabled);
    }
    return is_avx_enabled;
#else // CCL_AVX_COMPILER
    LOG_DEBUG("AVX disabled due to compiler");
    return 0;
#endif // CCL_AVX_COMPILER
}

void memcpy(void *dst, const void *src, size_t size) {
    std::copy((char *)(src), (char *)(src) + (size), (char *)(dst));
}

void memcpy_nontemporal(void *dst, const void *src, size_t size) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    size_t n = size;

    if (!is_nts_supported()) {
        LOG_DEBUG("NTS-based memcpy is requested but not supported, use regular memcpy");
    }

#ifdef CCL_AVX_COMPILER
    if ((n <= 256) || !is_nts_supported()) {
        memcpy(d, s, n);
        return;
    }

    if (((uintptr_t)d) & 63) {
        const uintptr_t t = 64 - (((uintptr_t)d) & 63);
        memcpy(d, s, t);
        d += t;
        s += t;
        n -= t;
    }

    while (n >= 256) {
        __m256i ymm0 = _mm256_loadu_si256((__m256i const *)(s + (32 * 0)));
        __m256i ymm1 = _mm256_loadu_si256((__m256i const *)(s + (32 * 1)));
        __m256i ymm2 = _mm256_loadu_si256((__m256i const *)(s + (32 * 2)));
        __m256i ymm3 = _mm256_loadu_si256((__m256i const *)(s + (32 * 3)));
        __m256i ymm4 = _mm256_loadu_si256((__m256i const *)(s + (32 * 4)));
        __m256i ymm5 = _mm256_loadu_si256((__m256i const *)(s + (32 * 5)));
        __m256i ymm6 = _mm256_loadu_si256((__m256i const *)(s + (32 * 6)));
        __m256i ymm7 = _mm256_loadu_si256((__m256i const *)(s + (32 * 7)));
        _mm256_stream_si256((__m256i *)(d + (32 * 0)), ymm0);
        _mm256_stream_si256((__m256i *)(d + (32 * 1)), ymm1);
        _mm256_stream_si256((__m256i *)(d + (32 * 2)), ymm2);
        _mm256_stream_si256((__m256i *)(d + (32 * 3)), ymm3);
        _mm256_stream_si256((__m256i *)(d + (32 * 4)), ymm4);
        _mm256_stream_si256((__m256i *)(d + (32 * 5)), ymm5);
        _mm256_stream_si256((__m256i *)(d + (32 * 6)), ymm6);
        _mm256_stream_si256((__m256i *)(d + (32 * 7)), ymm7);
        d += 256;
        s += 256;
        n -= 256;
    }

    if (n & 128) {
        __m256i ymm0 = _mm256_loadu_si256((__m256i const *)(s + (32 * 0)));
        __m256i ymm1 = _mm256_loadu_si256((__m256i const *)(s + (32 * 1)));
        __m256i ymm2 = _mm256_loadu_si256((__m256i const *)(s + (32 * 2)));
        __m256i ymm3 = _mm256_loadu_si256((__m256i const *)(s + (32 * 3)));
        _mm256_stream_si256((__m256i *)(d + (32 * 0)), ymm0);
        _mm256_stream_si256((__m256i *)(d + (32 * 1)), ymm1);
        _mm256_stream_si256((__m256i *)(d + (32 * 2)), ymm2);
        _mm256_stream_si256((__m256i *)(d + (32 * 3)), ymm3);
        d += 128;
        s += 128;
    }

    if (n & 64) {
        __m256i ymm0 = _mm256_loadu_si256((__m256i const *)(s + (32 * 0)));
        __m256i ymm1 = _mm256_loadu_si256((__m256i const *)(s + (32 * 1)));
        _mm256_stream_si256((__m256i *)(d + (32 * 0)), ymm0);
        _mm256_stream_si256((__m256i *)(d + (32 * 1)), ymm1);
        d += 64;
        s += 64;
    }

    if (n & 63) {
        memcpy(d, s, (n & 63));
    }

    _mm_sfence();

#else // CCL_AVX_COMPILER
    memcpy(d, s, n);
#endif // CCL_AVX_COMPILER
}

} // namespace ccl
