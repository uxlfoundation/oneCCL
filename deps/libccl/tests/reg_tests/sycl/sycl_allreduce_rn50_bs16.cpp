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

#include <ostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <sys/wait.h>
#include "sycl_base.hpp"
#include "ctevent.h"

/* The test does allreduce similar to Resnet50 application. The key feature is the
   input vector sizes which are particular to Resnet50. The group size parameter
   allows the user to divide the vector into smaller groups which can decrease the
   size of the allreduce kernels. */

using namespace std;
using namespace sycl;

int64_t MemcpyInFusionBuffer(const vector<float *> &entries,
                             const std::vector<size_t> &sizes,
                             void *buffer,
                             queue &q,
                             size_t index,
                             size_t end) {
    int64_t offset = 0;
    for (size_t i = index; i < end; i++) {
        void *buffer_offset = (uint8_t *)buffer + offset;
        auto e = q.memcpy(buffer_offset, entries[i], sizes[i]);
        offset += sizes[i];
    }
    return offset;
}
void MemcpyOutFusionBuffer(vector<float *> &entries,
                           std::vector<size_t> &sizes,
                           void *buffer,
                           queue &q,
                           size_t index,
                           size_t end) {
    int64_t offset = 0;
    for (size_t i = index; i < end; i++) {
        void *buffer_offset = (uint8_t *)buffer + offset;
        auto e = q.memcpy(entries[i], buffer_offset, sizes[i]);
        offset += sizes[i];
    }
}

int main(int argc, char *argv[]) {
    CTEvent::enabled() = true;
    auto pid = CTEvent::forkcmd();
    const std::vector<size_t> sizes{
        4000, 8192000, 8192, 8192, 4194304, 2048, 2048, 9437184, 2048, 2048, 4194304, 8192,
        8192, 4194304, 2048, 2048, 9437184, 2048, 2048, 4194304, 8192, 8192, 8388608, 8192,
        8192, 4194304, 2048, 2048, 9437184, 2048, 2048, 2097152, 4096, 4096, 1048576, 1024,
        1024, 2359296, 1024, 1024, 1048576, 4096, 4096, 1048576, 1024, 1024, 2359296, 1024,
        1024, 1048576, 4096, 4096, 1048576, 1024, 1024, 2359296, 1024, 1024, 1048576, 4096,
        4096, 1048576, 1024, 1024, 2359296, 1024, 1024, 1048576, 4096, 4096, 1048576, 1024,
        1024, 2359296, 1024, 1024, 1048576, 4096, 4096, 2097152, 4096, 4096, 1048576, 1024,
        1024, 2359296, 1024, 1024, 524288,  2048, 2048, 262144,  512,  512,  589824,  512,
        512,  262144,  2048, 2048, 262144,  512,  512,  589824,  512,  512,  262144,  2048,
        2048, 262144,  512,  512,  589824,  512,  512,  262144,  2048, 2048, 524288,  2048,
        2048, 262144,  512,  512,  589824,  512,  512,  131072,  1024, 1024, 65536,   256,
        256,  147456,  256,  256,  65536,   1024, 1024, 65536,   256,  256,  147456,  256,
        256,  65536,   1024, 1024, 65536,   1024, 1024, 65536,   256,  256,  147456,  256,
        256,  16384,   256,  256,  37632
    };
    vector<float *> entries;
    int c, dtype_size = 4, warmup = 20, iters = 10;
    bool enable_cache = false, root_as_device = false, out_of_order = false;
    bool group_api = false;
    string group;
    std::vector<int> groups{ 161 };

    while ((c = getopt(argc, argv, "aorc:g:i:w:")) != -1)
        switch (c) {
            case 'a': group_api = true; break;
            case 'c': enable_cache = true; break;
            case 'r': root_as_device = true; break;
            case 'o': out_of_order = true; break;
            case 'g': {
                stringstream ss(optarg);
                groups.clear();
                while (getline(ss, group, ','))
                    groups.push_back(atoi(group.c_str()));
                break;
            }
            case 'i': iters = atoi(optarg); break;
            case 'w': warmup = atoi(optarg); break;
            case '?': std::cout << "Unknown option\n"; return 1;
            default: abort();
        }

    assert(accumulate(groups.begin(), groups.end(), 0) == int(sizes.size()));

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    CTEvent::setconfig("/dev/null");

    atexit(mpi_finalize);

    auto q = out_of_order
                 ? create_sycl_queues("gpu", { rank }, root_as_device, {})[0]
                 : create_sycl_queues(
                       "gpu", { rank }, root_as_device, { sycl::property::queue::in_order() })[0];
    buf_allocator<float> allocator(q);
    auto usm_alloc_type = usm::alloc::device;

    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    /* create buffers */
    size_t count = 0;
    for (size_t i = 0; i < sizes.size(); i++) {
        entries.push_back(allocator.allocate(sizes[i] / dtype_size, usm_alloc_type));
        count += sizes[i] / dtype_size;
    }
    auto send_buf = allocator.allocate(count, usm_alloc_type);

    if (group_api) {
        ccl::group_start();
    }

    /* invoke allreduce */
    for (int i = 0; i < warmup + iters; i++) {
        if (i >= warmup)
            CTEvent::push("iteration");
        size_t offset = 0;
        for (auto g : groups) {
            string suffix = std::to_string(offset);
            if (i >= warmup)
                CTEvent::push("memcpy_in_" + suffix);
            auto cur_count =
                MemcpyInFusionBuffer(entries, sizes, send_buf, q, offset, offset + g) / dtype_size;
            q.wait();
            if (i >= warmup) {
                CTEvent::pop();
                CTEvent::push("cclallreduce_" + suffix);
            }
            auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
            if (enable_cache) {
                string match_id = "_len_" + std::to_string(cur_count);

                for (size_t idx = 0; idx < entries.size(); idx++) {
                    match_id += "_" + std::to_string(idx);
                }

                attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(match_id));
                attr.set<ccl::operation_attr_id::to_cache>(true);
            }
            else {
                attr.set<ccl::operation_attr_id::to_cache>(false);
            }

            if (group_api) {
                ccl::allreduce(send_buf,
                               send_buf,
                               cur_count,
                               ccl::datatype::float32,
                               ccl::reduction::sum,
                               comm,
                               stream,
                               attr);
            }
            else {
                ccl::allreduce(send_buf,
                               send_buf,
                               cur_count,
                               ccl::datatype::float32,
                               ccl::reduction::sum,
                               comm,
                               stream,
                               attr)
                    .wait();
            }
            if (i >= warmup) {
                CTEvent::pop();
                CTEvent::push("memcpy_out_" + suffix);
            }
            MemcpyInFusionBuffer(entries, sizes, send_buf, q, offset, offset + g);
            q.wait();
            if (i >= warmup)
                CTEvent::pop();
            offset += g;
        }
        if (i >= warmup)
            CTEvent::pop();
    }

    if (group_api) {
        ccl::group_end();
    }

    if (!handle_exception(q))
        return -1;

    if (rank == 0) {
        CTEvent::print_stats();
    }

    if (pid > 0)
        wait(&pid);

    return 0;
}
