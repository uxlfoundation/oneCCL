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

#include <vector>
#include <list>
#include <string>
#include <utility>
#include <thread>
#include <random>

#include "base.hpp"

int main() {
    /**
     * The example only works with CCL_ATL_TRANSPORT=ofi
     */
    setenv("CCL_ATL_TRANSPORT", "ofi", 0);

    setenv("CCL_UNORDERED_COLL", "1", 1);

    const size_t buf_size = 1024;
    const size_t iter_count = 64;

    std::vector<ccl::string_class> match_ids;

    /* event, operation idx */
    std::list<std::pair<ccl::event, size_t>> active_ops;

    std::vector<std::vector<float>> send_bufs;
    std::vector<std::vector<float>> recv_bufs;

    ccl::init();

    int size, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto comm = ccl::create_communicator(size, rank, kvs);

    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
    attr.set<ccl::operation_attr_id::to_cache>(true);

    std::random_device rand_dev;
    std::uniform_int_distribution<size_t> distribution(0, size - 1);

    for (auto idx = 0; idx < size; idx++) {
        match_ids.emplace_back("match_id_" + std::to_string(idx));
        send_bufs.emplace_back(buf_size, idx + 1);
        recv_bufs.emplace_back(buf_size, 0.0f);
    }

    if (rank != 0) {
        // delay non-root ranks to check that delayed comm creation works
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    for (size_t iter = 0; iter < iter_count; ++iter) {
        std::cout << "starting iter " << iter << std::endl;

        size_t start_idx = distribution(rand_dev);
        size_t rank_idx = start_idx;

        for (auto idx = 0; idx < size; idx++) {
            std::cout << "submit allreduce " << rank_idx << " for match_id " << match_ids[rank_idx]
                      << std::endl;

            attr.set<ccl::operation_attr_id::match_id>(match_ids[rank_idx]);

            active_ops.emplace_back(ccl::allreduce(send_bufs[rank_idx].data(),
                                                   recv_bufs[rank_idx].data(),
                                                   buf_size,
                                                   ccl::reduction::sum,
                                                   comm,
                                                   attr),
                                    rank_idx);

            rank_idx = (rank_idx + 1) % size;
        }

        while (!active_ops.empty()) {
            for (auto it = active_ops.begin(); it != active_ops.end();) {
                if (it->first.test()) {
                    float expected = (it->second + 1) * size;
                    printf(
                        "completed allreduce %zu for match_id %s. Actual %3.2f, expected %3.2f\n",
                        it->second,
                        match_ids[it->second].c_str(),
                        recv_bufs[it->second][0],
                        expected);

                    for (size_t idx = 0; idx < buf_size; ++idx) {
                        if (recv_bufs[it->second][idx] != expected) {
                            fprintf(
                                stderr,
                                "unexpected result, rank %zu, result %3.2f, intitial %3.2f, exp %3.f\n",
                                it->second,
                                recv_bufs[it->second][idx],
                                send_bufs[it->second][idx],
                                expected);
                            std::cout << "FAILED" << std::endl;
                            exit(1);
                        }
                    }

                    it = active_ops.erase(it);
                }
                else {
                    ++it;
                }
            }
        }

        std::cout << "iter " << iter << " has been finished" << std::endl;
    }

    ccl::barrier(comm);

    if (rank == 0)
        std::cout << "PASSED" << std::endl << std::flush;

    return 0;
}
