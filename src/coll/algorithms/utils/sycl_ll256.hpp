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

#include <vector>
#include <sstream>
#include <iostream>

#include <sycl/sycl.hpp>

using message_t = sycl::vec<uint32_t, 4>;
typedef uint32_t pattern_t;
extern uint16_t pattern_counter;

#define __LscFence() __asm__ __volatile__("lsc_fence.ugm.clean.sysrel")
#define __LscLoadUnCached(var, addr) \
    __asm__ __volatile__("lsc_load.ugm.uc.uc   (M1, 16)  %0:d64  flat[%1]:a64" \
                         : "=rw"(var) \
                         : "rw"(addr) \
                         : "memory")
#define __LscLoadCached(var, addr) \
    __asm__ __volatile__("lsc_load.ugm.ca.ca   (M1, 16)  %0:d64  flat[%1]:a64" \
                         : "=rw"(var) \
                         : "rw"(addr) \
                         : "memory")
#ifdef CCL_SYCL_ENABLE_ARCA
#define __LscLoadUnCachedVec(var, addr) \
    __asm__ __volatile__("lsc_load.ugm.uc.uc   (M1, 16)  %0:d32x4  flat[%1]:a64" \
                         : "=rw"(reinterpret_cast<typename message_t::vector_t &>(var)) \
                         : "rw"(addr) \
                         : "memory")
#else
#define __LscLoadUnCachedVec(var, addr) \
    __asm__ __volatile__("lsc_load.ugm.uc.ca   (M1, 16)  %0:d32x4  flat[%1]:a64" \
                         : "=rw"(reinterpret_cast<typename message_t::vector_t &>(var)) \
                         : "rw"(addr) \
                         : "memory")
#endif
#define __LscLoadCachedVec(var, addr) \
    __asm__ __volatile__("lsc_load.ugm.ca.ca   (M1, 16)  %0:d32x4  flat[%1]:a64" \
                         : "=rw"(reinterpret_cast<typename message_t::vector_t &>(var)) \
                         : "rw"(addr) \
                         : "memory")

#define __LscStoreUnCached(addr, var) \
    __asm__ __volatile__("lsc_store.ugm.uc.uc  (M1, 16)  flat[%0]:a64  %1:d64" \
                         : \
                         : "rw"(addr), "rw"(var) \
                         : "memory")
#define __LscStoreCached(addr, var) \
    __asm__ __volatile__("lsc_store.ugm.ca.ca  (M1, 16)  flat[%0]:a64  %1:d64" \
                         : \
                         : "rw"(addr), "rw"(var) \
                         : "memory")
#ifdef CCL_SYCL_ENABLE_ARCA
#define __LscStoreUnCachedVec(addr, var) \
    __asm__ __volatile__("lsc_store.ugm.uc.uc  (M1, 16)  flat[%0]:a64  %1:d32x4" \
                         : \
                         : "rw"(addr), "rw"(reinterpret_cast<typename message_t::vector_t &>(var)) \
                         : "memory")
#else
#define __LscStoreUnCachedVec(addr, var) \
    __asm__ __volatile__("lsc_store.ugm.uc.wb  (M1, 16)  flat[%0]:a64  %1:d32x4" \
                         : \
                         : "rw"(addr), "rw"(reinterpret_cast<typename message_t::vector_t &>(var)) \
                         : "memory")
#endif
#define __LscStoreCachedVec(addr, var) \
    __asm__ __volatile__("lsc_store.ugm.ca.ca  (M1, 16)  flat[%0]:a64  %1:d32x4" \
                         : \
                         : "rw"(addr), "rw"(reinterpret_cast<typename message_t::vector_t &>(var)) \
                         : "memory")

#define LscLoadCached    __LscLoadCachedVec
#define LscLoadUnCached  __LscLoadUnCachedVec
#define LscStoreCached   __LscStoreCachedVec
#define LscStoreUnCached __LscStoreUnCachedVec

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
// load data, and check if arrived
static inline void sync_data(char *src,
                             message_t &data,
                             sycl::sub_group &sg,
                             int lid,
                             pattern_t pattern) {
    size_t sz = sizeof(message_t);
    //auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();

    do {
        LscLoadUnCached(data, src);
    } while (sycl::any_of_group(
        sg,
        ((lid == 3) && (data[3] != pattern)) || ((lid == 7) && (data[3] != pattern)) ||
            ((lid == 11) && (data[3] != pattern)) || ((lid == 15) && (data[3] != pattern))));
}

static inline void shuffle_data(message_t &data) {
#ifdef CCL_SYCL_ENABLE_ARCA
    __asm__ __volatile__("mov (M1, 1) %0(1, 7)<1> %0(6, 3)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %0(6, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(5, 7)<1> %0(7, 3)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         :);
#else
    __asm__ __volatile__("mov (M1, 1) %0(0, 15)<1> %0(3, 3)<0;1,0>\n"
                         "mov (M1, 1) %0(1, 15)<1> %0(3, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(2, 15)<1> %0(3, 11)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         :);
#endif
}

static inline void insert_pattern(message_t &data, pattern_t pattern) {
#ifdef CCL_SYCL_ENABLE_ARCA
    __asm__ __volatile__("mov (M1, 1) %0(6, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(6, 7)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 7)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : "rw"(pattern));
#else
    __asm__ __volatile__("mov (M1, 1) %0(3, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 11)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 15)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : "rw"(pattern));
#endif
}

static inline void restore_data(message_t &data) {
#ifdef CCL_SYCL_ENABLE_ARCA
    __asm__ __volatile__("mov (M1, 1) %0(6, 3)<1> %0(1, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(6, 7)<1> %0(3, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 3)<1> %0(5, 7)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         :);
#else
    __asm__ __volatile__("mov (M1, 1) %0(3, 3)<1> %0(0, 15)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %0(1, 15)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 11)<1> %0(2, 15)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         :);
#endif
}
#endif

static inline void ll256_send_data(message_t &src_data, char *dst, pattern_t pattern) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    shuffle_data(src_data);

    insert_pattern(src_data, pattern);

    LscStoreUnCached(dst, src_data);
    //*(message_t *)dst = src_data;
#endif
}

static inline void ll256_send(char *src, char *dst, bool load, pattern_t pattern) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    if (load)
        LscLoadCached(data, src);
    //data = *(message_t *)src;

    shuffle_data(data);

    //sycl::ext::oneapi::experimental::printf("before shuffle_data: dst %p src %p data 0x%08X 0x%08X 0x%08X 0x%08X\n", (void *)dst, (void *)src, data[0], data[1], data[2], data[3]);
    insert_pattern(data, pattern);
    //sycl::ext::oneapi::experimental::printf("after shuffle_data: dst %p src %p data 0x%08X 0x%08X 0x%08X 0x%08X\n", (void *)dst, (void *)src, data[0], data[1], data[2], data[3]);

    LscStoreUnCached(dst, data);
    //*(message_t *)dst = data;
#endif
}

// check tmp buffer for data arrived, save the data in message_t data
static inline void ll256_recv_data(message_t &recv_data,
                                   char *tmpbuf,
                                   sycl::sub_group &sg,
                                   int lid,
                                   pattern_t pattern) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    /* check if data arrived in src */
    sync_data(tmpbuf, recv_data, sg, lid, pattern);

    restore_data(recv_data);
#endif
}

static inline void ll256_recv(char *recvbuf,
                              char *tmpbuf,
                              sycl::sub_group &sg,
                              int lid,
                              int req_workitems,
                              pattern_t pattern) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;

    /* check if data arrived in src */
    sync_data(tmpbuf, data, sg, lid, pattern);

    restore_data(data);

    if (lid < req_workitems) {
        LscStoreUnCached(recvbuf, data);
        //*(message_t *)recvbuf = data;
    }
#endif
}

// recv to local_recvbuf and send the data to remote_recvbuf
static inline void ll256_forward(char *tmpbuf,
                                 char *local_recvbuf,
                                 char *remote_recvbuf,
                                 sycl::sub_group &sg,
                                 int lid,
                                 int req_workitems,
                                 pattern_t pattern) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    sync_data(tmpbuf, data, sg, lid, pattern);
    LscStoreUnCached(remote_recvbuf, data);
    //*(message_t *)local_recvbuf = data;

    restore_data(data);

    if (lid < req_workitems) {
        LscStoreUnCached(local_recvbuf, data);
        //*(message_t *)remote_recvbuf = data;
    }
#endif
}
