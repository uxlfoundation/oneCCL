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

#include <mpi.h>

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

#include "transport.hpp"

transport_data::transport_data() {
    init_by_mpi();

    service_comms.push_back(ccl::create_communicator(size, rank, kvs));
}

transport_data::~transport_data() {
    deinit_by_mpi();
}

transport_data& transport_data::instance() {
    static transport_data inst;
    return inst;
}

void transport_data::init_by_mpi() {
    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ccl::shared_ptr_class<ccl::kvs> kvs_candidate;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs_candidate = ccl::create_main_kvs();
        main_addr = kvs_candidate->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs_candidate = ccl::create_kvs(main_addr);
    }
    kvs = kvs_candidate;
    init_comms();
}

void transport_data::deinit_by_mpi() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);
    if (!is_finalized) {
        MPI_Finalize();
    }
}

void transport_data::init_comms() {
    std::vector<int> local_ranks;
    for (int idx = 0; idx < ranks_per_proc; idx++) {
        local_ranks.push_back(rank * ranks_per_proc + idx);
    }

    ccl::context context = ccl::create_context();
    std::vector<ccl::device> devices;
    std::map<int, ccl::device> r2d_map;

#ifdef CCL_ENABLE_SYCL
    auto sycl_queues = create_sycl_queues("gpu", local_ranks);
    ASSERT(!sycl_queues.empty(), "queues should contain at least one queue");
    ASSERT(static_cast<size_t>(ranks_per_proc) == sycl_queues.size(),
           "ranks and queues sizes should match");

    auto sycl_context = sycl_queues[0].get_context();
    context = ccl::create_context(sycl_context);

    for (int idx = 0; idx < ranks_per_proc; idx++) {
        streams.push_back(ccl::create_stream(sycl_queues[idx]));
        devices.push_back(ccl::create_device(sycl_queues[idx].get_device()));
        allocators.push_back(buf_allocator<char>(streams[0].get_native()));
    }
#else // CCL_ENABLE_SYCL
    for (int idx = 0; idx < ranks_per_proc; idx++) {
        streams.push_back(ccl::create_stream());
        devices.push_back(ccl::create_device());
    }
#endif // CCL_ENABLE_SYCL

    for (int idx = 0; idx < ranks_per_proc; idx++) {
        r2d_map.emplace(local_ranks[idx], devices[idx]);
    }

    comms = ccl::create_communicators(size * ranks_per_proc, r2d_map, context, kvs);

    ASSERT((int)comms.size() == ranks_per_proc,
           "unexpected comms size %zu, expected %d",
           comms.size(),
           ranks_per_proc);
}

void transport_data::reset_comms() {
    ccl::barrier(get_service_comm());
    comms.clear();
    service_comms.clear();
}

int transport_data::get_rank() const noexcept {
    return rank;
}

int transport_data::get_size() const noexcept {
    return size;
}

ccl::shared_ptr_class<ccl::kvs> transport_data::get_kvs() {
    return kvs;
}

ccl::communicator& transport_data::get_comm() {
    return comms[0];
}

ccl::communicator& transport_data::get_service_comm() {
    return service_comms[0];
}

ccl::stream& transport_data::get_stream() {
    return streams[0];
}

#ifdef CCL_ENABLE_SYCL
buf_allocator<char>& transport_data::get_allocator() {
    return allocators[0];
}
#endif // CCL_ENABLE_SYCL
