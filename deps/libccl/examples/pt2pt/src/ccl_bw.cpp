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

    print_user_options("Bandwidth", options, comms[0]);

    double start_t = 0.0, end_t = 0.0, diff_t = 0.0;
    size_t dtype_size = sizeof(ccl::datatype::int32);

    auto q = transport.get_sycl_queue();
    auto streams = transport.get_streams();
    std::vector<ccl::event> ccl_events{};

    for (auto& count : options.elem_counts) {
        auto buf_send = sycl::malloc_device<int>(count, q);
        auto buf_recv = sycl::malloc_device<int>(count, q);

        // oneCCL spec defines attr identifiers that may be used to fill operation
        // attribute objects. It means for every pair of op, we have to keep own unique attr
        // because we may have conflicts between 2 different pairs with one common attr.
        auto attr = create_attr(options.cache, count, std::to_string(0));
        auto attr1 = create_attr(options.cache, count, std::to_string(1));

        if (rank == options.peers[0]) {
            for (size_t iter_idx = 0; iter_idx < (options.warmup_iters + options.iters);
                 iter_idx++) {
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

                if (iter_idx == options.warmup_iters) {
                    ccl::barrier(comms[0]);
                    start_t = MPI_Wtime();
                }

                for (int j = 0; j < options.window_size; j++) {
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
                }

                auto recv_event = ccl::recv(buf_recv,
                                            1,
                                            ccl::datatype::int32,
                                            options.peers[1],
                                            comms[0],
                                            streams[0],
                                            attr1);
                if (options.wait) {
                    recv_event.wait();
                }
                ccl_events.emplace_back(std::move(recv_event));

                end_t = MPI_Wtime();
                diff_t = end_t - start_t;
            }
        }
        else if (rank == options.peers[1]) {
            for (size_t iter_idx = 0; iter_idx < (options.warmup_iters + options.iters);
                 iter_idx++) {
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

                if (iter_idx == options.warmup_iters) {
                    ccl::barrier(comms[0]);
                }

                for (int j = 0; j < options.window_size; j++) {
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
                }

                // we can send 1 count here, this pair is for aligning
                // no need a big count
                auto send_event = ccl::send(buf_send,
                                            1,
                                            ccl::datatype::int32,
                                            options.peers[0],
                                            comms[0],
                                            streams[0],
                                            attr1);
                if (options.wait) {
                    send_event.wait();
                }
                ccl_events.emplace_back(std::move(send_event));

                if (options.check == CHECK_ALL_ITERS ||
                    (options.check == CHECK_LAST_ITER &&
                     iter_idx == (options.warmup_iters + options.iters) - 1)) {
                    check_gpu_buffers(q, options, count, iter_idx, buf_recv, ccl_events);
                }
            }
        }

        if (rank == options.peers[0]) {
            double bandwidth_t =
                (count * dtype_size / 1e6 * options.iters * options.window_size) / diff_t;
            print_timings(comms[0], options, bandwidth_t, count * dtype_size, "Mbytes/sec");
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

    print_user_options("Bandwidth", options, comms[0]);

    double start_t = 0.0, end_t = 0.0, diff_t = 0.0;
    size_t dtype_size = sizeof(ccl::datatype::int32);

    std::vector<int> buf_send;
    std::vector<int> buf_recv;

    for (auto& count : options.elem_counts) {
        buf_send.reserve(count);
        buf_recv.reserve(count);

        if (rank == options.peers[0]) {
            for (size_t iter_idx = 0; iter_idx < (options.warmup_iters + options.iters);
                 iter_idx++) {
                // init the buffer
                for (size_t id = 0; id < count; id++) {
                    buf_send[id] = id + iter_idx;
                    buf_recv[id] = INVALID_VALUE;
                }

                if (iter_idx == options.warmup_iters) {
                    ccl::barrier(comms[0]);
                    start_t = MPI_Wtime();
                }

                for (int j = 0; j < options.window_size; j++) {
                    ccl::send(
                        buf_send.data(), count, ccl::datatype::int32, options.peers[1], comms[0])
                        .wait();
                }

                ccl::recv(buf_recv.data(), 1, ccl::datatype::int32, options.peers[1], comms[0])
                    .wait();

                end_t = MPI_Wtime();
                diff_t = end_t - start_t;
            }
        }
        else if (rank == options.peers[1]) {
            for (size_t iter_idx = 0; iter_idx < (options.warmup_iters + options.iters);
                 iter_idx++) {
                // init the buffer
                for (size_t id = 0; id < count; id++) {
                    buf_send[id] = id + iter_idx;
                    buf_recv[id] = INVALID_VALUE;
                }

                if (iter_idx == options.warmup_iters) {
                    ccl::barrier(comms[0]);
                }

                for (int j = 0; j < options.window_size; j++) {
                    ccl::recv(
                        buf_recv.data(), count, ccl::datatype::int32, options.peers[0], comms[0])
                        .wait();
                }

                // we can send 1 count here, this pair is for aligning
                // no need a big count
                ccl::send(buf_send.data(), 1, ccl::datatype::int32, options.peers[0], comms[0])
                    .wait();

                if (options.check == CHECK_ALL_ITERS ||
                    (options.check == CHECK_LAST_ITER &&
                     iter_idx == (options.warmup_iters + options.iters) - 1)) {
                    check_cpu_buffers(count, iter_idx, buf_recv);
                }
            }
        }

        if (rank == options.peers[0]) {
            double bandwidth_t =
                (count * dtype_size / 1e6 * options.iters * options.window_size) / diff_t;
            print_timings(comms[0], options, bandwidth_t, count * dtype_size, "Mbytes/sec");
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
