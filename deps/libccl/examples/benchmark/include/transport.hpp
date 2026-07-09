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
#include "types.hpp"

class transport_data {
public:
    static transport_data& instance();
    static size_t get_comm_size();

    int get_rank() const noexcept;
    int get_size() const noexcept;

    ccl::shared_ptr_class<ccl::kvs> get_kvs();
    ccl::communicator& get_service_comm();
    void init_comms(user_options_t& options);
    std::vector<ccl::communicator>& get_comms();
    void reset_comms();

    std::vector<ccl::stream>& get_streams();
    std::vector<ccl::stream>& get_bench_streams();

private:
    transport_data();
    ~transport_data();

    int rank;
    int size;

    std::vector<size_t> local_ranks;

    ccl::shared_ptr_class<ccl::kvs> kvs;
    std::vector<ccl::communicator> service_comms;
    std::vector<ccl::communicator> comms;

    /*
       FIXME: explicitly separate CCL and bench streams
              while runtime doesn't provide MT on the same queue
    */
    std::vector<ccl::stream> streams;
    std::vector<ccl::stream> bench_streams;

    void init_by_mpi();
    void deinit_by_mpi();
};
