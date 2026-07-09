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

#include <map>
#include <vector>

#include "oneapi/ccl.hpp"
#ifdef CCL_ENABLE_SYCL
#include "sycl_base.hpp"
#endif // CCL_ENABLE_SYCL
#include "pt2pt_base.hpp"

class transport_data {
public:
    transport_data(const transport_data& other) = delete;
    transport_data& operator=(const transport_data& other) = delete;
    static transport_data& instance();
    static size_t get_comm_size();

    int get_rank() const noexcept;
    int get_size() const noexcept;

    ccl::shared_ptr_class<ccl::kvs> get_kvs();

    void init_comms(user_options_t& options);
    std::vector<ccl::communicator>& get_comms();
    void reset_comms();

#ifdef CCL_ENABLE_SYCL
    std::vector<ccl::stream>& get_streams();

    void create_sycl_queue(user_options_t& options);
    sycl::queue get_sycl_queue();
#endif // CCL_ENABLE_SYCL

private:
    transport_data();
    ~transport_data();

    int rank;
    int size;

    std::vector<size_t> local_ranks;

    ccl::shared_ptr_class<ccl::kvs> kvs;
    std::vector<ccl::communicator> comms;

#ifdef CCL_ENABLE_SYCL
    std::vector<ccl::stream> streams;
    sycl::queue queue;
#endif // CCL_ENABLE_SYCL

    void init_by_mpi();
    void deinit_by_mpi();
};

transport_data::transport_data() {
    init_by_mpi();
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

#ifdef CCL_ENABLE_SYCL
std::vector<ccl::stream>& transport_data::get_streams() {
    return streams;
}
#endif // CCL_ENABLE_SYCL

void transport_data::init_comms(user_options_t& options) {
#ifdef CCL_ENABLE_SYCL
    if (options.backend == BACKEND_GPU) {
        create_sycl_queue(options);

        auto q = get_sycl_queue();

        // create communicator
        auto dev = ccl::create_device(q.get_device());
        auto ctx = ccl::create_context(q.get_context());
        comms.push_back(ccl::create_communicator(size, rank, dev, ctx, kvs));

        // create stream
        streams.push_back(ccl::create_stream(q));
    }
    else {
#endif // CCL_ENABLE_SYCL
        comms.push_back(ccl::create_communicator(size, rank, kvs));
#ifdef CCL_ENABLE_SYCL
    }
#endif // CCL_ENABLE_SYCL
}

std::vector<ccl::communicator>& transport_data::get_comms() {
    return comms;
}

#ifdef CCL_ENABLE_SYCL
void transport_data::create_sycl_queue(user_options_t& options) {
    sycl::property_list props{};
    if (options.queue) {
        props = { sycl::property::queue::in_order{} };
    }

    if (!::create_sycl_queue("gpu", rank, queue, props)) {
        exit(INVALID_RETURN);
    }
}

sycl::queue transport_data::get_sycl_queue() {
    return queue;
}
#endif // CCL_ENABLE_SYCL

void transport_data::reset_comms() {
    comms.clear();
#ifdef CCL_ENABLE_SYCL
    streams.clear();
#endif // CCL_ENABLE_SYCL
}
