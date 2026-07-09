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
#ifdef CCL_ENABLE_SYCL
#include "sycl_base.hpp"
#endif // CCL_ENABLE_SYCL
#include "pt2pt_transport.hpp"

#include "oneapi/ccl.hpp"

void run_gpu_backend(user_options_t& options) {
#ifdef CCL_ENABLE_SYCL
    auto& transport = transport_data::instance();
    transport.init_comms(options);

    auto rank = transport.get_rank();
    auto& comms = transport.get_comms();

    print_user_options("Latency", options, comms[0]);

    size_t dtype_size = sizeof(ccl::datatype::int32);

    auto q = transport.get_sycl_queue();
    auto streams = transport.get_streams();
    std::vector<ccl::event> ccl_events{};

    for (auto& count : options.elem_counts) {
        double start_t = 0.0, end_t = 0.0, diff_t = 0.0, total_latency_t = 0.0;

        // create buffers
        auto buf_send = sycl::malloc_device<int>(count, q);
        auto buf_recv = sycl::malloc_device<int>(count, q);

        // oneCCL spec defines attr identifiers that may be used to fill operation
        // attribute objects. It means for every pair of op, we have to keep own unique attr
        // because we may have conflicts between 2 different pairs with one common attr.
        auto attr = create_attr(options.cache, count, std::to_string(0));
        auto attr1 = create_attr(options.cache, count, std::to_string(1));

        for (size_t iter_idx = 0; iter_idx < (options.warmup_iters + options.iters); iter_idx++) {
            // init the buffer
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_send[id] = id + iter_idx;
                    buf_recv[id] = INVALID_VALUE;
                });
            });

            if (options.wait) {
                e.wait_and_throw();
            }

            if (iter_idx == options.warmup_iters - 1) {
                // to ensure that all processes or threads have reached
                // a certain synchronization point before proceeding time
                // calculation
                ccl::barrier(comms[0]);
            }

            if (rank == options.peers[0]) {
                if (iter_idx >= options.warmup_iters) {
                    start_t = MPI_Wtime();
                }

                auto send_event = ccl::send(buf_send,
                                            count,
                                            ccl::datatype::int32,
                                            options.peers[1],
                                            comms[0],
                                            streams[0],
                                            attr);
                if (options.wait) {
                    send_event.wait();
                }
                ccl_events.emplace_back(std::move(send_event));

                auto recv_event = ccl::recv(buf_recv,
                                            count,
                                            ccl::datatype::int32,
                                            options.peers[1],
                                            comms[0],
                                            streams[0],
                                            attr1);
                if (options.wait) {
                    recv_event.wait();
                }
                ccl_events.emplace_back(std::move(recv_event));

                if (iter_idx >= options.warmup_iters) {
                    end_t = MPI_Wtime();
                    diff_t = end_t - start_t;
                    total_latency_t += diff_t;
                }
            }
            else if (rank == options.peers[1]) {
                auto recv_event = ccl::recv(buf_recv,
                                            count,
                                            ccl::datatype::int32,
                                            options.peers[0],
                                            comms[0],
                                            streams[0],
                                            attr);
                if (options.wait) {
                    recv_event.wait();
                }
                ccl_events.emplace_back(std::move(recv_event));

                auto send_event = ccl::send(buf_send,
                                            count,
                                            ccl::datatype::int32,
                                            options.peers[0],
                                            comms[0],
                                            streams[0],
                                            attr1);
                if (options.wait) {
                    send_event.wait();
                }
                ccl_events.emplace_back(std::move(send_event));
            }

            if (options.check == CHECK_ALL_ITERS ||
                (options.check == CHECK_LAST_ITER &&
                 iter_idx == (options.warmup_iters + options.iters) - 1)) {
                ccl::barrier(comms[0]);
                check_gpu_buffers(q, options, count, iter_idx, buf_recv, ccl_events);
            }
        }

        if (rank == options.peers[0]) {
            // test measures the round trip latency, divide by two to get the one-way latency
            double average_t = (total_latency_t * 1e6) / (2.0 * options.iters);
            print_timings(comms[0], options, average_t, count * dtype_size, "#usec(latency)");
        }

        sycl::free(buf_send, q);
        sycl::free(buf_recv, q);
    }

    PRINT_BY_ROOT(comms[0], "\n# All done\n");

    transport.reset_comms();
#endif // CCL_ENABLE_SYCL
}

void run_cpu_backend(user_options_t& options) {
    auto& transport = transport_data::instance();
    transport.init_comms(options);

    auto rank = transport.get_rank();
    auto& comms = transport.get_comms();

    print_user_options("Latency", options, comms[0]);

    size_t dtype_size = sizeof(ccl::datatype::int32);

    std::vector<int> buf_send;
    std::vector<int> buf_recv;

    for (auto& count : options.elem_counts) {
        double start_t = 0.0, end_t = 0.0, diff_t = 0.0, total_latency_t = 0.0;

        // create buffers
        buf_send.reserve(count);
        buf_recv.reserve(count);

        for (size_t iter_idx = 0; iter_idx < (options.warmup_iters + options.iters); iter_idx++) {
            // init the buffer
            for (size_t id = 0; id < count; id++) {
                buf_send[id] = id + iter_idx;
                buf_recv[id] = INVALID_VALUE;
            }

            if (iter_idx == options.warmup_iters - 1) {
                // to ensure that all processes or threads have reached
                // a certain synchronization point before proceeding time
                // calculation
                ccl::barrier(comms[0]);
            }

            if (rank == options.peers[0]) {
                if (iter_idx >= options.warmup_iters) {
                    start_t = MPI_Wtime();
                }

                ccl::send(buf_send.data(), count, ccl::datatype::int32, options.peers[1], comms[0])
                    .wait();

                ccl::recv(buf_recv.data(), count, ccl::datatype::int32, options.peers[1], comms[0])
                    .wait();

                if (iter_idx >= options.warmup_iters) {
                    end_t = MPI_Wtime();
                    diff_t = end_t - start_t;
                    total_latency_t += diff_t;
                }
            }
            else if (rank == options.peers[1]) {
                ccl::recv(buf_recv.data(), count, ccl::datatype::int32, options.peers[0], comms[0])
                    .wait();

                ccl::send(buf_send.data(), count, ccl::datatype::int32, options.peers[0], comms[0])
                    .wait();
            }

            if (options.check == CHECK_ALL_ITERS ||
                (options.check == CHECK_LAST_ITER &&
                 iter_idx == (options.warmup_iters + options.iters) - 1)) {
                ccl::barrier(comms[0]);
                check_cpu_buffers(count, iter_idx, buf_recv);
            }
        }

        if (rank == options.peers[0]) {
            // test measures the round trip latency, divide by two to get the one-way latency
            double average_t = (total_latency_t * 1e6) / (2.0 * options.iters);
            print_timings(comms[0], options, average_t, count * dtype_size, "#usec(latency)");
        }

        buf_send.clear();
        buf_recv.clear();
    }

    PRINT_BY_ROOT(comms[0], "\n# All done\n");

    transport.reset_comms();
}

int main(int argc, char* argv[]) {
    user_options_t options;

    if (parse_user_options(argc, argv, options)) {
        print_help_usage(argv[0]);
        exit(INVALID_RETURN);
    }

    if (options.backend == BACKEND_GPU) {
        run_gpu_backend(options);
    }
    else {
        run_cpu_backend(options);
    }

    return 0;
}
