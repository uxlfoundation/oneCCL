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


#include <chrono>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

class TestInstance;

typedef ccl::communicator (*create_comm_function)(int size,
                                                  int rank,
                                                  ccl::device&,
                                                  const ccl::context&,
                                                  ccl::shared_ptr_class<ccl::kvs_interface>,
                                                  const ccl::comm_attr&);
typedef std::shared_ptr<ccl::kvs> (*create_kvs_function)(size_t size, size_t rank);

// forward function declarations -----------------------------

static void run_test_scenario_unguarded(TestInstance& test_instance,
                                        size_t iterations_in_batch,
                                        size_t iterations_in_recording);

// helper functions ------------------------------------------

static std::vector<sycl::event> get_sycl_events(const ccl::vector_class<ccl::event>& deps) {
    std::vector<sycl::event> ret;
    ret.reserve(deps.size());
    for (auto& dep : deps) {
        ret.push_back(dep.get_native());
    }
    return ret;
}

static std::vector<sycl::device> select_devices() {
    // Find and initialize Level-Zero devices and queues
    std::vector<sycl::device> devices;
    std::vector<sycl::queue> queues;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
        auto platform_name = platform.get_info<sycl::info::platform::name>();
        bool is_level_zero = platform_name.find("Level-Zero") != std::string::npos;
        if (is_level_zero) {
            std::cout << "Platform_name is:  " << platform_name << std::endl;
            auto device_list = platform.get_devices();
            for (const auto& device : device_list) {
                if (device.is_gpu()) {
                    devices.push_back(device);
                }
            }
        }
    }

    return devices;
}

static uint64_t get_microseconds() {
    auto time_now = std::chrono::steady_clock::now();
    auto time_since_epoch = time_now.time_since_epoch();
    auto time_since_epoch_millis =
        std::chrono::duration_cast<std::chrono::microseconds>(time_since_epoch).count();
    return static_cast<uint64_t>(time_since_epoch_millis);
}

// test instance code ----------------------------------------

class TestInstance {
protected:
    std::vector<sycl::device> devices;
    sycl::context context;
    sycl::queue q;
    size_t count;
    int* send_buf;
    int* recv_buf;
    ccl::device ccl_device;
    ccl::context ccl_context;
    std::shared_ptr<ccl::kvs> kvs;
    ccl::communicator comm;
    size_t comm_size;
    size_t comm_rank;
    ccl::stream stream;
    std::string name;

    void reinit(size_t send_buf_size, size_t recv_buf_size, size_t new_count) {
        sycl::free(send_buf, q);
        sycl::free(recv_buf, q);

        send_buf = sycl::aligned_alloc_device<int>(4 * 1024, send_buf_size, q);
        recv_buf = sycl::aligned_alloc_device<int>(4 * 1024, recv_buf_size, q);

        count = new_count;
    }

public:
    TestInstance(std::string name,
                 size_t count,
                 size_t send_buf_size,
                 size_t recv_buf_size,
                 size_t comm_size,
                 size_t comm_rank,
                 std::vector<sycl::device> devices,
                 sycl::context context,
                 create_kvs_function create_kvs,
                 create_comm_function create_comm)
            : devices(devices),
              context(context),
              q(context,
                devices[comm_rank % devices.size()],
                sycl::property_list{ sycl::property::queue::in_order{} }),
              count(count),
              send_buf(sycl::aligned_alloc_device<int>(4 * 1024, send_buf_size, q)),
              recv_buf(sycl::aligned_alloc_device<int>(4 * 1024, recv_buf_size, q)),
              ccl_device(ccl::create_device(devices[comm_rank % devices.size()])),
              ccl_context(ccl::create_context(context)),
              kvs(create_kvs(comm_size, comm_rank)),
              comm(create_comm(comm_size,
                               comm_rank,
                               ccl_device,
                               ccl_context,
                               kvs,
                               ccl::default_comm_attr)),
              comm_size(comm_size),
              comm_rank(comm_rank),
              stream(ccl::create_stream(q)),
              name(name) {
        std::cout << "Running [" << name << "] on "
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";
    }

    ~TestInstance() {
        sycl::free(send_buf, q);
        sycl::free(recv_buf, q);
    }

    sycl::queue& get_queue() {
        return q;
    }

    std::string get_name() {
        return name;
    }

    size_t get_count() {
        return count;
    }

    virtual void init_clear_buffers() = 0;
    // manipulate input buffers
    //
    // should be executed on input buffers and recordable
    virtual std::vector<ccl::event> pre_run(const std::vector<ccl::event>& deps) = 0;
    virtual ccl::event run_test(const std::vector<ccl::event>& deps) = 0;
    virtual void check_test() = 0;
    virtual void reinit(size_t new_count) = 0;

    size_t get_comm_size() {
        return this->comm_size;
    }
    size_t get_comm_rank() {
        return this->comm_rank;
    }
};

class TestInstanceAllreduce : public TestInstance {
public:
    TestInstanceAllreduce(size_t count,
                          size_t comm_size,
                          size_t comm_rank,
                          std::vector<sycl::device> devices,
                          sycl::context context,
                          create_kvs_function create_kvs,
                          create_comm_function create_comm = ccl::create_communicator)
            : TestInstance(std::string("allreduce"),
                           count,
                           count,
                           count,
                           comm_size,
                           comm_rank,
                           devices,
                           context,
                           create_kvs,
                           create_comm) {}

    void init_clear_buffers() {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;

        q.submit([&](auto& h) {
            h.parallel_for(count, [=](auto id) {
                recv_buf[id] = -3;
                send_buf[id] = -2;
            });
        });
        std::cout << "clear init requested" << std::endl;
        q.wait_and_throw();
        std::cout << "clear init completed" << std::endl;
    }

    std::vector<ccl::event> pre_run(const std::vector<ccl::event>& deps) {
        const size_t l_rank = comm_rank;
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;

        sycl::event e = q.submit([&](auto& h) {
            h.depends_on(get_sycl_events(deps));
            h.parallel_for(count, [=](auto id) {
                // rank 0:  [1, 2, 3, 4]
                // rank 1:  [2, 3, 4, 5]
                // rank 2:  [3, 4, 5, 6]
                // rank 3:  [4, 5, 6, 7]
                // allreduced:
                //          [10, 14, 18, 22]
                send_buf[id] = l_rank + id + 1;
                recv_buf[id] = -1;
            });
        });

        std::vector<ccl::event> ret;
        ret.push_back(ccl::event::create_from_native(e));

        return ret;
    }

    ccl::event run_test(const std::vector<ccl::event>& deps) {
        auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
        return ccl::allreduce(send_buf,
                              recv_buf,
                              count,
                              ccl::datatype::int32,
                              ccl::reduction::sum,
                              comm,
                              stream,
                              attr,
                              deps);
    }

    void check_test() {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_size = comm_size;

        int check_sum = 0;

        for (size_t i = 1; i <= l_size; ++i) {
            check_sum += i;
        }

        /* open recv_buf and check its correctness on the device side */
        sycl::buffer<int> check_buf(count);
        sycl::buffer<int> rbuf(count);
        sycl::buffer<int> sbuf(count);

        std::cout << "check_test: submitting, count: " << count << std::endl;
        q.submit([&](auto& h) {
            sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
            sycl::accessor rbuf_acc(rbuf, h, sycl::write_only);
            sycl::accessor sbuf_acc(sbuf, h, sycl::write_only);
            h.parallel_for(count, [=](auto id) {
                rbuf_acc[id] = recv_buf[id];
                sbuf_acc[id] = send_buf[id];
                if (recv_buf[id] != static_cast<int>(check_sum + l_size * id)) {
                    check_buf_acc[id] = -1;
                }
                else {
                    check_buf_acc[id] = 0;
                }
            });
        });

        std::cout << "check_test: waiting" << std::endl;
        q.wait();
        std::cout << "check_test: waited" << std::endl;

        /* print out the result of the test on the host side */
        {
            size_t i;
            sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
            sycl::host_accessor rbuf_acc(rbuf, sycl::read_only);
            sycl::host_accessor sbuf_acc(sbuf, sycl::read_only);
            bool passed = true;
            for (i = 0; i < count; i++) {
                if (check_buf_acc[i] != 0) {
                    std::cout << "FAILED\n";
                    std::cout << "i, sbuf[i], rbuf[i], check_buf_acc[i], expected: " << i << ", "
                              << sbuf_acc[i] << ", " << rbuf_acc[i] << ", " << check_buf_acc[i]
                              << ", " << static_cast<int>(check_sum + l_size * i) << std::endl;
                    passed = false;
                    break;
                }
            }
            if (passed) {
                std::cout << "PASSED\n";
            }
        }
    }

    void reinit(size_t new_count) {
        TestInstance::reinit(new_count, new_count, new_count);
    }
};

class TestInstanceReduceScatter : public TestInstance {
public:
    TestInstanceReduceScatter(size_t count,
                              size_t comm_size,
                              size_t comm_rank,
                              std::vector<sycl::device> devices,
                              sycl::context context,
                              create_kvs_function create_kvs,
                              create_comm_function create_comm = ccl::create_communicator)
            : TestInstance(std::string("reduce_scatter"),
                           count,
                           count * comm_size,
                           count,
                           comm_size,
                           comm_rank,
                           devices,
                           context,
                           create_kvs,
                           create_comm) {}

    void init_clear_buffers() {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;

        q.submit([&](auto& h) {
            h.parallel_for(count * comm_size, [=](auto id) {
                send_buf[id] = -2;
            });
        });
        q.submit([&](auto& h) {
            h.parallel_for(count, [=](auto id) {
                recv_buf[id] = -3;
            });
        });
        std::cout << "clear init requested" << std::endl;
        q.wait_and_throw();
        std::cout << "clear init completed" << std::endl;
    }

    std::vector<ccl::event> pre_run(const std::vector<ccl::event>& deps) {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_rank = comm_rank;
        const size_t captured_count = this->count;

        auto e = q.submit([&](auto& h) {
            h.depends_on(get_sycl_events(deps));
            h.parallel_for(count * comm_size, [=](auto id) {
                // rank 0:  [1, 2, 3, 4]
                // rank 1:  [2, 3, 4, 5]
                // rank 2:  [3, 4, 5, 6]
                // rank 3:  [4, 5, 6, 7]
                // allreduced:
                //          [10, 14, 18, 22]
                send_buf[id] = l_rank + id + 1;
                if (id < captured_count)
                    recv_buf[id] = -1;
            });
        });

        std::vector<ccl::event> ret;
        ret.push_back(ccl::event::create_from_native(e));

        return ret;
    }

    ccl::event run_test(const std::vector<ccl::event>& deps) {
        auto attr = ccl::create_operation_attr<ccl::reduce_scatter_attr>();
        return ccl::reduce_scatter(send_buf,
                                   recv_buf,
                                   count,
                                   ccl::datatype::int32,
                                   ccl::reduction::sum,
                                   comm,
                                   stream,
                                   attr,
                                   deps);
    }

    void check_test() {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_size = comm_size;
        const size_t captured_count = this->count;

        int check_sum = 0;

        for (size_t i = 1; i <= l_size; ++i) {
            check_sum += i;
        }

        /* open recv_buf and check its correctness on the device side */
        sycl::buffer<int> check_buf(count);
        sycl::buffer<int> rbuf(count);
        sycl::buffer<int> sbuf(count * comm_size);

        std::cout << "check_test: submitting, count: " << count << std::endl;
        int c_rank = comm_rank;
        q.submit([&](auto& h) {
            sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
            sycl::accessor rbuf_acc(rbuf, h, sycl::write_only);
            sycl::accessor sbuf_acc(sbuf, h, sycl::write_only);
            h.parallel_for(count * comm_size, [=](auto id) {
                sbuf_acc[id] = send_buf[id];
                if (id < captured_count) {
                    rbuf_acc[id] = recv_buf[id];
                    size_t recv_buf_id_start_id = id + c_rank * captured_count;
                    int expected = static_cast<int>(check_sum + l_size * recv_buf_id_start_id);
                    if (recv_buf[id] != expected) {
                        // check_buf_acc[id] = -1;
                        check_buf_acc[id] = -1;
                    }
                    else {
                        check_buf_acc[id] = 0;
                    }
                }
            });
        });

        std::cout << "check_test: waiting" << std::endl;
        q.wait();
        std::cout << "check_test: waited" << std::endl;

        /* print out the result of the test on the host side */
        {
            size_t i;
            sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
            sycl::host_accessor rbuf_acc(rbuf, sycl::read_only);
            sycl::host_accessor sbuf_acc(sbuf, sycl::read_only);
            bool passed = true;
            for (i = 0; i < count; i++) {
                if (check_buf_acc[i] != 0) {
                    size_t recv_buf_id_start_id = i + comm_rank * captured_count;
                    std::cout << "FAILED\n";
                    std::cout << "i, sbuf[i], rbuf[i], check_buf_acc[i], expected: " << i << ", "
                              << sbuf_acc[i] << ", " << rbuf_acc[i] << ", " << check_buf_acc[i]
                              << ", " << static_cast<int>(check_sum + l_size * recv_buf_id_start_id)
                              << std::endl;
                    passed = false;
                    break;
                }
            }
            if (passed) {
                std::cout << "PASSED\n";
            }
        }
    }
    void reinit(size_t new_count) {
        TestInstance::reinit(new_count * comm_size, new_count, new_count);
    }
};

class TestInstanceAllgather : public TestInstance {
public:
    TestInstanceAllgather(size_t count,
                          size_t comm_size,
                          size_t comm_rank,
                          std::vector<sycl::device> devices,
                          sycl::context context,
                          create_kvs_function create_kvs,
                          create_comm_function create_comm = ccl::create_communicator)
            : TestInstance(std::string("allgather"),
                           count,
                           count,
                           count * comm_size,
                           comm_size,
                           comm_rank,
                           devices,
                           context,
                           create_kvs,
                           create_comm) {}

    void init_clear_buffers() {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t captured_count = this->count;

        q.submit([&](auto& h) {
            h.parallel_for(count * comm_size, [=](auto id) {
                recv_buf[id] = -3;
                if (id < captured_count) {
                    send_buf[id] = -2;
                }
            });
        });
        std::cout << "clear init requested" << std::endl;
        q.wait_and_throw();
        std::cout << "clear init completed" << std::endl;
    }

    std::vector<ccl::event> pre_run(const std::vector<ccl::event>& deps) {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_rank = comm_rank;
        const size_t captured_count = this->count;

        auto e = q.submit([&](auto& h) {
            h.depends_on(get_sycl_events(deps));
            h.parallel_for(count * comm_size, [=](auto id) {
                // rank 0:  [1, 2, 3, 4]
                // rank 1:  [2, 3, 4, 5]
                // rank 2:  [3, 4, 5, 6]
                // rank 3:  [4, 5, 6, 7]
                // allreduced:
                //          [10, 14, 18, 22]
                if (id < captured_count) {
                    send_buf[id] = l_rank + id + 1;
                }
                recv_buf[id] = -1;
            });
        });

        std::vector<ccl::event> ret;
        ret.push_back(ccl::event::create_from_native(e));

        return ret;
    }
    ccl::event run_test(const std::vector<ccl::event>& deps) {
        auto attr = ccl::create_operation_attr<ccl::allgather_attr>();
        return ccl::allgather(
            send_buf, recv_buf, count, ccl::datatype::int32, comm, stream, attr, deps);
    }

    void check_test() {
        const size_t captured_count = this->count;

        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_size = comm_size;

        /* open recv_buf and check its correctness on the device side */
        sycl::buffer<int> check_buf(count);
        sycl::buffer<int> rbuf(count);
        sycl::buffer<int> sbuf(count);

        int c_rank = comm_rank;
        std::cout << "check_test: submitting, count: " << count << std::endl;
        q.submit([&](auto& h) {
            sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
            sycl::accessor rbuf_acc(rbuf, h, sycl::write_only);
            sycl::accessor sbuf_acc(sbuf, h, sycl::write_only);
            h.parallel_for(count, [=](auto id) {
                rbuf_acc[id] = recv_buf[id];
                sbuf_acc[id] = send_buf[id];
                int src_rank = id / captured_count;
                int src_idx = id % captured_count;
                int expected = static_cast<int>(src_rank + src_idx + 1);
                if (recv_buf[id] != expected) {
                    check_buf_acc[id] = -1;
                }
                else {
                    check_buf_acc[id] = 0;
                }
            });
        });

        std::cout << "check_test: waiting" << std::endl;
        q.wait();
        std::cout << "check_test: waited" << std::endl;

        /* print out the result of the test on the host side */
        {
            size_t i;
            sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
            sycl::host_accessor rbuf_acc(rbuf, sycl::read_only);
            sycl::host_accessor sbuf_acc(sbuf, sycl::read_only);
            bool passed = true;
            for (i = 0; i < count; i++) {
                if (check_buf_acc[i] == -1) {
                    int src_rank = i / count;
                    int src_idx = i % count;
                    int expected = static_cast<int>(src_rank + src_idx + 1);
                    std::cout << "FAILED\n";
                    std::cout << "i, sbuf[i], rbuf[i], check_buf_acc[i], expected: " << i << ", "
                              << sbuf_acc[i] << ", " << rbuf_acc[i] << ", " << check_buf_acc[i]
                              << ", " << expected << std::endl;
                    passed = false;
                    break;
                }
            }
            if (passed) {
                std::cout << "PASSED\n";
            }
        }
    }
    void reinit(size_t new_count) {
        TestInstance::reinit(new_count, new_count * comm_size, new_count);
    }
};

class TestInstanceAlltoall : public TestInstance {
public:
    TestInstanceAlltoall(size_t count,
                         size_t comm_size,
                         size_t comm_rank,
                          std::vector<sycl::device> devices,
                          sycl::context context,
                         create_kvs_function create_kvs,
                         create_comm_function create_comm = ccl::create_communicator)
            : TestInstance(std::string("alltoall"),
                           count,
                           count * comm_size,
                           count * comm_size,
                           comm_size,
                           comm_rank,
                           devices,
                           context,
                           create_kvs,
                           create_comm) {}

    void init_clear_buffers() {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t captured_count = this->count;

        q.submit([&](auto& h) {
            h.parallel_for(count * comm_size, [=](auto id) {
                recv_buf[id] = -3;
                send_buf[id] = -2;
            });
        });
        std::cout << "clear init requested" << std::endl;
        q.wait_and_throw();
        std::cout << "clear init completed" << std::endl;
    }

    std::vector<ccl::event> pre_run(const std::vector<ccl::event>& deps) {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_size = comm_size;
        const size_t l_rank = comm_rank;
        const size_t captured_count = this->count;

        auto e = q.submit([&](auto& h) {
            h.depends_on(get_sycl_events(deps));
            h.parallel_for(count * comm_size, [=](auto id) {
                // rank 0:  [1,  2,  3,  4]
                // rank 1:  [5,  6,  7,  8]
                //
                // alltoallv'ed::
                // rank 0:  [1,  2,  5,  6]
                // rank 1:  [3,  4,  7,  8]
                send_buf[id] = static_cast<int>(4 * l_rank + id + 1);
                recv_buf[id] = -4;
            });
        });

        std::vector<ccl::event> ret;
        ret.push_back(ccl::event::create_from_native(e));

        return ret;
    }

    ccl::event run_test(const std::vector<ccl::event>& deps) {
        auto attr = ccl::create_operation_attr<ccl::alltoall_attr>();
        return ccl::alltoall(
            send_buf, recv_buf, count, ccl::datatype::int32, comm, stream, attr, deps);
    }

    void check_test() {
        const size_t captured_count = this->count;

        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_size = comm_size;

        /* open recv_buf and check its correctness on the device side */
        sycl::buffer<int> check_buf(count * comm_size);
        sycl::buffer<int> rbuf(count * comm_size);
        sycl::buffer<int> sbuf(count * comm_size);

        int c_rank = comm_rank;
        std::cout << "check_test: submitting, count: " << count << std::endl;
        q.submit([&](auto& h) {
            sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
            sycl::accessor rbuf_acc(rbuf, h, sycl::write_only);
            sycl::accessor sbuf_acc(sbuf, h, sycl::write_only);
            h.parallel_for(count * comm_size, [=](auto id) {
                rbuf_acc[id] = recv_buf[id];
                sbuf_acc[id] = send_buf[id];
                size_t src_rank = id / captured_count;
                size_t src_idx = id % captured_count + c_rank * captured_count;
                int expected = static_cast<int>(src_rank * 4 + src_idx + 1);
                if (recv_buf[id] != expected) {
                    check_buf_acc[id] = -1;
                }
                else {
                    check_buf_acc[id] = 0;
                }
            });
        });

        std::cout << "check_test: waiting" << std::endl;
        q.wait();
        std::cout << "check_test: waited" << std::endl;

        /* print out the result of the test on the host side */
        {
            size_t i;
            sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
            sycl::host_accessor rbuf_acc(rbuf, sycl::read_only);
            sycl::host_accessor sbuf_acc(sbuf, sycl::read_only);
            bool passed = true;
            for (i = 0; i < count * comm_size; i++) {
                if (check_buf_acc[i] == -1) {
                    size_t src_rank = i / captured_count;
                    size_t src_idx = i % captured_count + c_rank * captured_count;
                    int expected = static_cast<int>(src_rank * 4 + src_idx + 1);
                    std::cout << "FAILED\n";
                    std::cout
                        << "i, sbuf[i], rbuf[i], check_buf_acc[i], expected (src_rank, src_idx): "
                        << i << ", " << sbuf_acc[i] << ", " << rbuf_acc[i] << ", "
                        << check_buf_acc[i] << ", " << expected << " (" << src_rank << ", "
                        << src_idx << ")" << std::endl;
                    passed = false;
                    break;
                }
            }
            if (passed) {
                std::cout << "PASSED\n";
            }
        }
    }
    void reinit(size_t new_count) {
        TestInstance::reinit(new_count * comm_size, new_count * comm_size, new_count);
    }
};

class TestInstanceBroadcast : public TestInstance {
public:
    TestInstanceBroadcast(size_t count,
                          size_t comm_size,
                          size_t comm_rank,
                          std::vector<sycl::device> devices,
                          sycl::context context,
                          create_kvs_function create_kvs,
                          create_comm_function create_comm = ccl::create_communicator)
            : TestInstance(std::string("broadcast"),
                           count,
                           count,
                           count,
                           comm_size,
                           comm_rank,
                           devices,
                           context,
                           create_kvs,
                           create_comm) {}

    void init_clear_buffers() {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t captured_count = this->count;

        q.submit([&](auto& h) {
            h.parallel_for(count, [=](auto id) {
                recv_buf[id] = -3;
                send_buf[id] = -2;
            });
        });
        std::cout << "clear init requested" << std::endl;
        q.wait_and_throw();
        std::cout << "clear init completed" << std::endl;
    }

    std::vector<ccl::event> pre_run(const std::vector<ccl::event>& deps) {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_rank = comm_rank;

        auto e = q.submit([&](auto& h) {
            h.depends_on(get_sycl_events(deps));
            h.parallel_for(count, [=](auto id) {
                // rank 0:  [1, 2, 3, 4]
                // rank 1:  [2, 3, 4, 5]
                // rank 2:  [3, 4, 5, 6]
                // rank 3:  [4, 5, 6, 7]
                // broadcasted - rank 1:
                //          [2, 3, 4, 5]
                send_buf[id] = l_rank + id + 1;
                recv_buf[id] = -1;
            });
        });

        std::vector<ccl::event> ret;
        ret.push_back(ccl::event::create_from_native(e));

        return ret;
    }
    ccl::event run_test(const std::vector<ccl::event>& deps) {
        auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();
        return ccl::broadcast(
            send_buf, recv_buf, count, ccl::datatype::int32, 1, comm, stream, attr, deps);
    }

    void check_test() {
        const size_t captured_count = this->count;

        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_size = comm_size;

        /* open recv_buf and check its correctness on the device side */
        sycl::buffer<int> check_buf(count);
        sycl::buffer<int> rbuf(count);
        sycl::buffer<int> sbuf(count);

        int c_rank = comm_rank;
        std::cout << "check_test: submitting, count: " << count << std::endl;
        q.submit([&](auto& h) {
            sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
            sycl::accessor rbuf_acc(rbuf, h, sycl::write_only);
            sycl::accessor sbuf_acc(sbuf, h, sycl::write_only);
            h.parallel_for(count, [=](auto id) {
                rbuf_acc[id] = recv_buf[id];
                sbuf_acc[id] = send_buf[id];
                int expected = static_cast<int>(id) + 2;
                if (recv_buf[id] != expected) {
                    check_buf_acc[id] = -1;
                }
                else {
                    check_buf_acc[id] = 0;
                }
            });
        });

        std::cout << "check_test: waiting" << std::endl;
        q.wait();
        std::cout << "check_test: waited" << std::endl;

        /* print out the result of the test on the host side */
        {
            size_t i;
            sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
            sycl::host_accessor rbuf_acc(rbuf, sycl::read_only);
            sycl::host_accessor sbuf_acc(sbuf, sycl::read_only);
            bool passed = true;
            for (i = 0; i < count; i++) {
                if (check_buf_acc[i] == -1) {
                    int expected = static_cast<int>(i) + 2;
                    std::cout << "FAILED\n";
                    std::cout << "i, sbuf[i], rbuf[i], check_buf_acc[i], expected: " << i << ", "
                              << sbuf_acc[i] << ", " << rbuf_acc[i] << ", " << check_buf_acc[i]
                              << ", " << expected << std::endl;
                    passed = false;
                    break;
                }
            }
            if (passed) {
                std::cout << "PASSED\n";
            }
        }
    }
    void reinit(size_t new_count) {
        TestInstance::reinit(new_count, new_count, new_count);
    }
};

class TestInstancePt2Pt : public TestInstance {
public:
    TestInstancePt2Pt(size_t count,
                      size_t comm_size,
                      size_t comm_rank,
                      std::vector<sycl::device> devices,
                      sycl::context context,
                      create_kvs_function create_kvs,
                      create_comm_function create_comm = ccl::create_communicator)
            : TestInstance(std::string("pt2pt"),
                           count,
                           count,
                           count,
                           comm_size,
                           comm_rank,
                           devices,
                           context,
                           create_kvs,
                           create_comm) {}

    void init_clear_buffers() {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t captured_count = this->count;

        q.submit([&](auto& h) {
            h.parallel_for(count, [=](auto id) {
                recv_buf[id] = -3;
                send_buf[id] = -2;
            });
        });
        std::cout << "clear init requested" << std::endl;
        q.wait_and_throw();
        std::cout << "clear init completed" << std::endl;
    }

    std::vector<ccl::event> pre_run(const std::vector<ccl::event>& deps) {
        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_rank = comm_rank;

        auto e = q.submit([&](auto& h) {
            h.depends_on(get_sycl_events(deps));
            h.parallel_for(count, [=](auto id) {
                // rank 0:  [1, 2, 3, 4]
                // rank 1:  [2, 3, 4, 5]
                // rank 2:  [3, 4, 5, 6]
                // rank 3:  [4, 5, 6, 7]
                // broadcasted - rank 1:
                //          [2, 3, 4, 5]
                send_buf[id] = l_rank + id + 1;
                recv_buf[id] = -1;
            });
        });

        std::vector<ccl::event> ret;
        ret.push_back(ccl::event::create_from_native(e));

        return ret;
    }
    ccl::event run_test(const std::vector<ccl::event>& deps) {
        const size_t l_rank = comm_rank;
        auto attr = ccl::create_operation_attr<ccl::pt2pt_attr>();
        if (l_rank == 0) {
            return ccl::send(send_buf, count, ccl::datatype::int32, 1, comm, stream, attr, deps);
        }
        else if (l_rank == 1) {
            return ccl::recv(recv_buf, count, ccl::datatype::int32, 0, comm, stream, attr, deps);
        }

        // else: the rank does not participate, return empty event
        return ccl::event();
    }

    void check_test() {
        const size_t captured_count = this->count;

        int* send_buf = this->send_buf;
        int* recv_buf = this->recv_buf;
        const size_t l_rank = comm_rank;
        const size_t l_size = comm_size;

        /* open recv_buf and check its correctness on the device side */
        sycl::buffer<int> check_buf(count);
        sycl::buffer<int> rbuf(count);
        sycl::buffer<int> sbuf(count);

        int c_rank = comm_rank;
        std::cout << "check_test: submitting, count: " << count << std::endl;
        q.submit([&](auto& h) {
            sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
            sycl::accessor rbuf_acc(rbuf, h, sycl::write_only);
            sycl::accessor sbuf_acc(sbuf, h, sycl::write_only);
            h.parallel_for(count, [=](auto id) {
                rbuf_acc[id] = recv_buf[id];
                sbuf_acc[id] = send_buf[id];
                int expected = static_cast<int>(id) + 1;
                if (l_rank == 1 && recv_buf[id] != expected) {
                    check_buf_acc[id] = -1;
                }
                else {
                    check_buf_acc[id] = 0;
                }
            });
        });

        std::cout << "check_test: waiting" << std::endl;
        q.wait();
        std::cout << "check_test: waited" << std::endl;

        /* print out the result of the test on the host side */
        {
            size_t i;
            sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
            sycl::host_accessor rbuf_acc(rbuf, sycl::read_only);
            sycl::host_accessor sbuf_acc(sbuf, sycl::read_only);
            bool passed = true;
            for (i = 0; i < count; i++) {
                if (check_buf_acc[i] == -1) {
                    int expected = static_cast<int>(i) + 2;
                    std::cout << "FAILED\n";
                    std::cout << "i, sbuf[i], rbuf[i], check_buf_acc[i], expected: " << i << ", "
                              << sbuf_acc[i] << ", " << rbuf_acc[i] << ", " << check_buf_acc[i]
                              << ", " << expected << std::endl;
                    passed = false;
                    break;
                }
            }
            if (passed) {
                std::cout << "PASSED\n";
            }
        }
    }
    void reinit(size_t new_count) {
        TestInstance::reinit(new_count, new_count, new_count);
    }
};

// test runner code ------------------------------------------

static void run_test_scenario(TestInstance& test_instance,
                              size_t iterations_in_batch,
                              size_t iterations_in_recording) {
    if (test_instance.get_comm_rank() == 0) {
        std::cout << "Running size for int buffer count: " << test_instance.get_count()
                  << std::endl;
    }
    try {
        run_test_scenario_unguarded(test_instance, iterations_in_batch, iterations_in_recording);
    }
    catch (const sycl::exception& e) {
        std::cout << "EXCEPTION OCCURRED:" << std::endl
                  << e.what() << std::endl
                  << std::endl
                  << std::endl;
        throw;
    }
}

static void run_test_scenario_unguarded(TestInstance& test_instance,
                                        size_t iterations_in_batch,
                                        size_t iterations_in_recording) {
    std::vector<uint64_t> timestamps;
    timestamps.reserve(10);

    size_t rank = test_instance.get_comm_rank();

    // warmup
    {
        if (rank == 0)
            std::cout << "warmup" << std::endl;
        timestamps.push_back(get_microseconds());
        for (size_t i = 0; i < iterations_in_batch; ++i) {
            auto deps = test_instance.pre_run({});
            test_instance.run_test(deps).wait();
        }

        timestamps.push_back(get_microseconds());
        if (rank == 0)
            std::cout << "warmup done" << std::endl;
    }

    if (rank == 0)
        std::cout << "checking test" << std::endl;
    test_instance.check_test();
    if (rank == 0)
        std::cout << "test checked" << std::endl;
    test_instance.init_clear_buffers();
    if (rank == 0)
        std::cout << "buffers cleared" << std::endl;

    // actual run
    {
        if (rank == 0)
            std::cout << "run" << std::endl;
        timestamps.push_back(get_microseconds());
        for (size_t i = 0; i < iterations_in_batch; ++i) {
            std::vector<ccl::event> deps{};
            for (size_t i = 0; i < iterations_in_recording; ++i) {
                deps = test_instance.pre_run(deps);
                std::vector<ccl::event> tmp;
                tmp.push_back(test_instance.run_test(deps));
                deps = std::move(tmp);
            }
        }
        timestamps.push_back(get_microseconds());
        if (rank == 0)
            std::cout << "run done" << std::endl;
    }
    test_instance.check_test();
    test_instance.init_clear_buffers();

    timestamps.push_back(get_microseconds());

    sycl::ext::oneapi::experimental::command_graph graph(test_instance.get_queue().get_context(),
                                                         test_instance.get_queue().get_device());
    {
        if (rank == 0) {
            std::cout << "record" << std::endl;
        }
        graph.begin_recording(test_instance.get_queue());

        std::vector<ccl::event> deps{};
        for (size_t i = 0; i < iterations_in_recording; ++i) {
            deps = test_instance.pre_run(deps);
            std::vector<ccl::event> tmp;
            tmp.push_back(test_instance.run_test(deps));
            deps = std::move(tmp);
        }

        graph.end_recording();

        if (rank == 0) {
            std::cout << "record done" << std::endl;
        }
    }
    auto executable_graph = graph.finalize();
    if (rank == 0) {
        std::cout << "finalized" << std::endl;
    }

    timestamps.push_back(get_microseconds());

    test_instance.init_clear_buffers();

    {
        if (rank == 0) {
            std::cout << "graph first replay" << std::endl;
        }
        // replay the graph thing
        timestamps.push_back(get_microseconds());
        for (size_t i = 0; i < iterations_in_batch; ++i)
            test_instance.get_queue().ext_oneapi_graph(executable_graph).wait();
        timestamps.push_back(get_microseconds());
        if (rank == 0) {
            std::cout << "graph first replay done" << std::endl;
        }
    }

    test_instance.check_test();
    test_instance.init_clear_buffers();

    {
        if (rank == 0) {
            std::cout << "graph second replay" << std::endl;
        }
        // replay the graph thing
        timestamps.push_back(get_microseconds());
        for (size_t i = 0; i < iterations_in_batch; ++i)
            test_instance.get_queue().ext_oneapi_graph(executable_graph).wait();
        timestamps.push_back(get_microseconds());
        if (rank == 0) {
            std::cout << "graph second replay done" << std::endl;
        }
    }

    test_instance.check_test();
    test_instance.init_clear_buffers();

    uint64_t ptimestamp = timestamps[0];

    std::vector<std::string> names = { "warmup",        "check & clear",   "run",
                                       "check & clear", "create & record", "clear",
                                       "first replay",  "check & clear",   "second replay" };

    if (rank == 0) {
        std::cout << "Timestamp diffs:" << std::endl;
    }
    std::vector<double> timestamp_diffs;
    timestamp_diffs.reserve(timestamps.size());
    for (size_t i = 1; i < timestamps.size(); ++i) {
        timestamp_diffs.push_back(timestamps[i] - ptimestamp);
        if (rank == 0) {
            std::cout << names[i - 1] << ": " << timestamp_diffs.back() << std::endl;
        }
        ptimestamp = timestamps[i];
    }

    size_t regular_idx = 2;
    size_t replay_idx = 8;

    assert(timestamps[replay_idx] > 0);

    double speedup = (static_cast<double>(timestamp_diffs[regular_idx]) -
                      static_cast<double>(timestamp_diffs[replay_idx])) /
                     static_cast<double>(timestamp_diffs[regular_idx]);

    if (rank == 0) {
        std::cout << "Sycl graph speedup [" << test_instance.get_name() << ": "
                  << test_instance.get_count() << "]: " << speedup * 100 << " %" << std::endl;
        std::cout << "Finished scenario" << std::endl;
    }
}
