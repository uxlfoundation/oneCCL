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
#include "sycl_coll.hpp"
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

size_t transport_data::get_comm_size() {
    return transport_data::instance().get_comms()[0].size();
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
}

void transport_data::deinit_by_mpi() {
    MPI_Finalize();
}

ccl::communicator& transport_data::get_service_comm() {
    return service_comms[0];
}

std::vector<ccl::stream>& transport_data::get_streams() {
    return streams;
}

std::vector<ccl::stream>& transport_data::get_bench_streams() {
    return bench_streams;
}

void transport_data::init_comms(user_options_t& options) {
    int ranks_per_proc = options.ranks_per_proc;

    std::vector<int> proc_ranks;
    for (int idx = 0; idx < ranks_per_proc; idx++) {
        proc_ranks.push_back(rank * ranks_per_proc + idx);
    }

    ccl::context context = ccl::create_context();
    std::vector<ccl::device> devices;
    std::map<int, ccl::device> r2d_map;

    if (options.backend == BACKEND_HOST) {
        for (int idx = 0; idx < ranks_per_proc; idx++) {
            streams.push_back(ccl::create_stream());
            bench_streams.push_back(ccl::create_stream());
            devices.push_back(ccl::create_device());
        }
    }
#ifdef CCL_ENABLE_SYCL
    else if (options.backend == BACKEND_SYCL) {
        show_extened_info(options.show_additional_info);

        // empty props means out-of-order sycl queue
        sycl::property_list props{};
        if (options.sycl_queue_type) {
            props = { sycl::property::queue::in_order{},
                      sycl::property::queue::enable_profiling{} };
        }
        auto sycl_queues = create_sycl_queues(
            sycl_dev_names[options.sycl_dev_type], proc_ranks, options.sycl_root_dev, props);
        ASSERT(!sycl_queues.empty(), "queues should contain at least one queue");
        ASSERT(static_cast<size_t>(ranks_per_proc) == sycl_queues.size(),
               "ranks and queues sizes should match");

        auto sycl_context = sycl_queues[0].get_context();
        context = ccl::create_context(sycl_context);

        for (int idx = 0; idx < ranks_per_proc; idx++) {
            streams.push_back(ccl::create_stream(sycl_queues[idx]));
            auto q = sycl::queue(sycl_queues[idx].get_context(), sycl_queues[idx].get_device());
            bench_streams.push_back(ccl::create_stream(q));
            devices.push_back(ccl::create_device(sycl_queues[idx].get_device()));
            // TODO: multidevice unsupported yet
            // ASSERT(sycl_context == sycl_queues[idx].get_context(),
            //    "all sycl queues should be from the same sycl context");
        }
    }
#endif // CCL_ENABLE_SYCL
    else {
        ASSERT(0, "unknown backend %d", (int)options.backend);
    }

    for (int idx = 0; idx < ranks_per_proc; idx++) {
        r2d_map.emplace(proc_ranks[idx], devices[idx]);
    }

    comms = ccl::create_communicators(size * ranks_per_proc, r2d_map, context, kvs);

    ASSERT((int)comms.size() == ranks_per_proc,
           "unexpected comms size %zu, expected %d",
           comms.size(),
           ranks_per_proc);
}

std::vector<ccl::communicator>& transport_data::get_comms() {
    return comms;
}

void transport_data::reset_comms() {
    ccl::barrier(get_service_comm());
    comms.clear();
    service_comms.clear();
}
