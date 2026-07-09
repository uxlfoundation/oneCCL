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

#include "base.hpp"

class external_kvs : public ccl::kvs_interface {
public:
    external_kvs(ccl::shared_ptr_class<ccl::kvs> kvs) : kvs(kvs) {}

    virtual ccl::vector_class<char> get(const ccl::string_class& key) {
        return kvs->get(key);
    }

    virtual void set(const ccl::string_class& key, const ccl::vector_class<char>& data) {
        return kvs->set(key, data);
    }
    virtual int get_id() {
        return 0;
    }

private:
    ccl::shared_ptr_class<ccl::kvs> kvs;
};

void run_collective(const char* cmd_name,
                    const std::vector<float>& send_buf,
                    std::vector<float>& recv_buf,
                    const ccl::communicator& comm,
                    const ccl::allreduce_attr& attr) {
    std::chrono::system_clock::duration exec_time{ 0 };
    float expected = (comm.size() - 1) * (static_cast<float>(comm.size()) / 2);

    ccl::barrier(comm);

    for (size_t idx = 0; idx < ITERS; ++idx) {
        auto start = std::chrono::system_clock::now();
        ccl::allreduce(
            send_buf.data(), recv_buf.data(), recv_buf.size(), ccl::reduction::sum, comm, attr)
            .wait();
        exec_time += std::chrono::system_clock::now() - start;
    }

    for (size_t idx = 0; idx < recv_buf.size(); idx++) {
        if (recv_buf[idx] != expected) {
            fprintf(stderr, "idx %zu, expected %4.4f, got %4.4f\n", idx, expected, recv_buf[idx]);

            std::cout << "FAILED" << std::endl;
            std::terminate();
        }
    }

    ccl::barrier(comm);

    std::cout << "avg time of " << cmd_name << ": "
              << std::chrono::duration_cast<std::chrono::microseconds>(exec_time).count() / ITERS
              << ", us" << std::endl;
}

int main() {
    ccl::init_attr init_attr = ccl::create_init_attr();
    ccl::init(init_attr);

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

    auto ext_kvs = std::shared_ptr<external_kvs>(new external_kvs(kvs));

    auto comm = ccl::create_communicator(size, rank, ext_kvs);
    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();

    MSG_LOOP(comm, std::vector<float> send_buf(msg_count, static_cast<float>(comm.rank()));
             std::vector<float> recv_buf(msg_count);
             run_collective("regular allreduce", send_buf, recv_buf, comm, attr););

    return 0;
}
