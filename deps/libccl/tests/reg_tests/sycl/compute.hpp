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

#include "sycl_base.hpp"

#include <iomanip>

// Few useful terms used inside the code:
//  * kernel pipeline - is a set of steps that simulate any kind of workload.
//    We can run multiple pipelines in parallel, which means that we can have multiple
//    instances of the same kernels working over different sets of data.

// We declare two computation modes:
//  * compute_mode::single - stands for computation that is executed
//    once for all kernel pipelines. For example, one allreduce across
//    all buffers.
//  * compute_mode::multi - means that the computation has to be submitted
//    separately per kernel pipeline. For example, weights update kernel
//    should be created for each pipeline, so number of the kernels is
//    equal to overall kernel pipeline count.
enum class compute_mode { single, multi };

enum class buffer_type { weights, reduction, wide };

// The class is used to gather data about execution of computation steps and
// wait for their completion through `get_event` method.
class compute_result {
public:
    virtual sycl::event get_event() = 0;
    virtual const char *get_name() = 0;
    virtual size_t get_start_timestamp() = 0;
    virtual size_t get_end_timestamp() = 0;
    virtual size_t get_submission_time() = 0;
    virtual unsigned logging_level() {
        return 1;
    }

    virtual ~compute_result() {}
};

class sycl_compute_result : public compute_result {
public:
    sycl_compute_result(sycl::event &&e, size_t submission_time, const char *name)
            : e(std::move(e)),
              name(name),
              submission_time(submission_time) {}

    sycl::event get_event() override {
        return e;
    };
    const char *get_name() override {
        return name;
    }
    size_t get_start_timestamp() override {
        return e.get_profiling_info<sycl::info::event_profiling::command_start>();
    };
    size_t get_end_timestamp() override {
        return e.get_profiling_info<sycl::info::event_profiling::command_end>();
    };
    size_t get_submission_time() override {
        return submission_time;
    }

private:
    const sycl::event e;
    const char *name;
    const size_t submission_time;
};

class ccl_compute_result : public compute_result {
public:
    ccl_compute_result(ccl::event &&e,
                       sycl::event &&start,
                       sycl::event &&end,
                       size_t submission_time,
                       const char *name)
            : e(std::move(e)),
              start(std::move(start)),
              end(std::move(end)),
              submission_time(submission_time),
              name(name) {}

    sycl::event get_event() override {
        return e.get_native();
    };
    const char *get_name() override {
        return name;
    }
    size_t get_start_timestamp() override {
        return start.get_profiling_info<sycl::info::event_profiling::command_end>();
    };
    size_t get_end_timestamp() override {
        return end.get_profiling_info<sycl::info::event_profiling::command_start>();
    };
    size_t get_submission_time() override {
        return submission_time;
    }
    unsigned logging_level() override {
        return 0;
    }

private:
    const ccl::event e;
    const sycl::event start;
    const sycl::event end;
    const size_t submission_time;
    const char *name;
};

class combined_sycl_compute_result : public compute_result {
public:
    combined_sycl_compute_result(sycl::event &&e1,
                                 sycl::event &&e2,
                                 size_t submission_time,
                                 const char *name)
            : e1(std::move(e1)),
              e2(std::move(e2)),
              name(name),
              submission_time(submission_time) {}

    sycl::event get_event() override {
        return e2;
    };
    const char *get_name() override {
        return name;
    }
    size_t get_start_timestamp() override {
        return e1.get_profiling_info<sycl::info::event_profiling::command_start>();
    };
    size_t get_end_timestamp() override {
        return e2.get_profiling_info<sycl::info::event_profiling::command_end>();
    };
    size_t get_submission_time() override {
        return submission_time;
    }

private:
    const sycl::event e1;
    const sycl::event e2;
    const char *name;
    const size_t submission_time;
};

template <typename T>
// The class is responsible for submitting computation to XPU and
// running simulation of the computation on CPU in order to check
// correctness.
class compute {
public:
    virtual std::shared_ptr<compute_result> submit(sycl::queue q,
                                                   size_t iter_idx,
                                                   size_t kernel_idx,
                                                   size_t count,
                                                   std::unordered_map<buffer_type, T *> buffer_for,
                                                   std::vector<sycl::event> deps) = 0;
    virtual void simulate(size_t iter_idx,
                          size_t kernel_idx,
                          size_t count,
                          std::unordered_map<buffer_type, T *> buffer_for) = 0;

    virtual ~compute() {}
};

template <typename T>
void print_array_root(int rank, T *arr, size_t num, const char *name) {
    if (rank != 0) {
        return;
    }
    std::cout << std::setw(30) << std::left << name << " [";
    for (size_t i = 0; i < num; i++) {
        std::cout << arr[i];
        if (i != num - 1) {
            std::cout << ",";
        }
    }
    std::cout << "]" << std::endl;
}

template <typename T>
void print_array(int rank, T *arr, size_t num, const char *name) {
    std::cout << "[" << rank << "]" << name << " [";
    for (size_t i = 0; i < num; i++) {
        std::cout << arr[i];
        if (i != num - 1) {
            std::cout << ",";
        }
    }
    std::cout << "]" << std::endl;
}

template <typename T>
class kernel_operation : public compute<T> {
public:
    kernel_operation(size_t rank, bool random_init) : rank(rank), random_init(random_init) {}

    std::shared_ptr<compute_result> submit(sycl::queue q,
                                           size_t iter_idx,
                                           size_t kernel_idx,
                                           size_t count,
                                           std::unordered_map<buffer_type, T *> buffer_for,
                                           std::vector<sycl::event> deps) override {
        auto weights = buffer_for[buffer_type::weights];
        sycl::event e;

        auto submission_start = std::chrono::high_resolution_clock::now();

        if (iter_idx == 0) {
            auto weights_copy = weights;
            auto rank_copy = rank;

            if (!random_init) {
                auto kernel_idx_copy = kernel_idx;
                e = q.submit([&](auto &h) {
                    h.parallel_for(count, [=](auto id) {
                        // initial weight in first iteration
                        weights_copy[id] = 179 + kernel_idx_copy * (rank_copy + 1);
                    });
                });
            }
            else {
                e = q.submit([&](auto &h) {
                    h.single_task([=]() {
                        unsigned int previous = rank_copy;
                        unsigned int mod = 256;
                        unsigned int multiplier = 1103515245;
                        unsigned int increment = 12345;
                        for (size_t id = 0; id < count; id++) {
                            weights_copy[id] = (multiplier * previous + increment) % mod;
                            previous = weights_copy[id];
                        }
                    });
                });
            }
        }
        else {
            auto weights_copy = weights;
            auto rank_copy = rank;
            auto kernel_idx_copy = kernel_idx;
            e = q.submit([=](auto &h) {
                h.parallel_for(count, [=](auto id) {
                    // make weight differ in each iteration
                    weights_copy[id] = weights_copy[id] + (kernel_idx_copy * (rank_copy + 1));
                });
            });
        }

        auto submission_end = std::chrono::high_resolution_clock::now();
        auto api_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(submission_end - submission_start)
                .count();

        return std::make_shared<sycl_compute_result>(std::move(e), api_time, "kernel");
    }

    void simulate(size_t iter_idx,
                  size_t kernel_idx,
                  size_t count,
                  std::unordered_map<buffer_type, T *> buffer_for) override {
        auto weights = buffer_for[buffer_type::weights];

        if (iter_idx == 0) {
            if (!random_init) {
                auto rank_copy = rank;
                auto kernel_idx_copy = kernel_idx;
                for (size_t id = 0; id < count; id++) {
                    // initial weight in first iteration
                    weights[id] = 179 + kernel_idx_copy * (rank_copy + 1);
                }
            }
            else {
                unsigned int previous = rank;
                unsigned int mod = 256;
                unsigned int multiplier = 1103515245;
                unsigned int increment = 12345;
                for (size_t id = 0; id < count; id++) {
                    weights[id] = (multiplier * previous + increment) % mod;
                    previous = weights[id];
                }
            }
        }
        else {
            auto rank_copy = rank;
            auto kernel_idx_copy = kernel_idx;
            for (size_t id = 0; id < count; id++) {
                // make weight differ in each iteration
                weights[id] = weights[id] + (kernel_idx_copy * (rank_copy + 1));
            }
        }
    }

private:
    size_t rank;
    bool random_init;
};

template <typename T>
class weights_update : public compute<T> {
public:
    weights_update(size_t rank, size_t size) : rank(rank), size(size) {}

    std::shared_ptr<compute_result> submit(sycl::queue q,
                                           size_t iter_idx,
                                           size_t kernel_idx,
                                           size_t count,
                                           std::unordered_map<buffer_type, T *> buffer_for,
                                           std::vector<sycl::event> deps) override {
        auto weights = buffer_for[buffer_type::weights];
        auto size_copy = size;

        auto submission_start = std::chrono::high_resolution_clock::now();

        auto e = q.submit([&](auto &h) {
            h.depends_on(deps);
            h.parallel_for(count, [=](auto id) {
                weights[id] = (weights[id] / size_copy);
            });
        });

        auto submission_end = std::chrono::high_resolution_clock::now();
        auto api_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(submission_end - submission_start)
                .count();

        return std::make_shared<sycl_compute_result>(std::move(e), api_time, "update weights");
    }

    void simulate(size_t iter_idx,
                  size_t kernel_idx,
                  size_t count,
                  std::unordered_map<buffer_type, T *> buffer_for) override {
        auto weights = buffer_for[buffer_type::weights];

        for (size_t id = 0; id < count; id++) {
            weights[id] = (weights[id] / size);
        }
    }

private:
    size_t rank;
    size_t size;
};

template <typename T>
class copy_data : public compute<T> {
public:
    copy_data(size_t rank, size_t size, buffer_type from, buffer_type to)
            : rank(rank),
              size(size),
              from(from),
              to(to) {}

    std::shared_ptr<compute_result> submit(sycl::queue q,
                                           size_t iter_idx,
                                           size_t kernel_idx,
                                           size_t count,
                                           std::unordered_map<buffer_type, T *> buffer_for,
                                           std::vector<sycl::event> deps) override {
        if (from == to) {
            throw std::string("Unreasonable copy operation!");
        }
        auto weights = buffer_for[buffer_type::weights];
        auto reduction = buffer_for[buffer_type::reduction];
        auto wide = buffer_for[buffer_type::wide];

        T *src, *dst;
        size_t src_skip_count = 1;
        size_t dst_skip_count = 1;

        switch (from) {
            case buffer_type::wide:
                src = wide + rank;
                src_skip_count = size;
                break;
            case buffer_type::weights: src = weights; break;
            case buffer_type::reduction: src = reduction; break;
        }

        switch (to) {
            case buffer_type::wide:
                dst = wide + rank;
                dst_skip_count = size;
                break;
            case buffer_type::weights: dst = weights; break;
            case buffer_type::reduction: dst = reduction; break;
        }

        auto submission_start = std::chrono::high_resolution_clock::now();

        auto e = q.submit([&](auto &h) {
            h.depends_on(deps);
            h.parallel_for(count, [=](auto id) {
                dst[id * dst_skip_count] = src[id * src_skip_count];
            });
        });

        auto submission_end = std::chrono::high_resolution_clock::now();
        auto api_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(submission_end - submission_start)
                .count();

        return std::make_shared<sycl_compute_result>(std::move(e), api_time, "copy data");
    }

    void simulate(size_t iter_idx,
                  size_t kernel_idx,
                  size_t count,
                  std::unordered_map<buffer_type, T *> buffer_for) override {
        auto weights = buffer_for[buffer_type::weights];
        auto reduction = buffer_for[buffer_type::reduction];
        auto wide = buffer_for[buffer_type::wide];

        T *src, *dst;
        size_t src_skip_count = 1;
        size_t dst_skip_count = 1;

        switch (from) {
            case buffer_type::wide:
                src = wide + rank;
                src_skip_count = size;
                break;
            case buffer_type::weights: src = weights; break;
            case buffer_type::reduction: src = reduction; break;
        }

        switch (to) {
            case buffer_type::wide:
                dst = wide + rank;
                dst_skip_count = size;
                break;
            case buffer_type::weights: dst = weights; break;
            case buffer_type::reduction: dst = reduction; break;
        }

        for (size_t id = 0; id < count; id++) {
            dst[id * dst_skip_count] = src[id * src_skip_count];
        }
    }

private:
    int rank;
    int size;
    buffer_type from;
    buffer_type to;
};
template <typename T>
class allreduce_operation : public compute<T> {
public:
    allreduce_operation(size_t rank,
                        ccl::communicator *comm,
                        ccl::stream *stream,
                        ccl::datatype dtype,
                        std::vector<ccl::allreduce_attr> attrs,
                        size_t size)
            : rank(rank),
              comm(comm),
              stream(stream),
              dtype(dtype),
              attrs(attrs),
              size(size) {}

    std::shared_ptr<compute_result> submit(sycl::queue q,
                                           size_t iter_idx,
                                           size_t kernel_idx,
                                           size_t count,
                                           std::unordered_map<buffer_type, T *> buffer_for,
                                           std::vector<sycl::event> deps) override {
        auto send_buf = buffer_for[buffer_type::weights];
        auto recv_buf = buffer_for[buffer_type::reduction];
        ccl::event ccl_event;
        std::vector<ccl::event> ccl_deps;

        for (auto e : deps) {
            ccl_deps.push_back(ccl::create_event(e));
        }

        auto start_event = q.submit([&](auto &h) {
            h.depends_on(deps);
            h.single_task([=]() {});
        });

        auto coll_start = std::chrono::high_resolution_clock::now();
        if (ccl_deps.empty()) {
            ccl_event = ccl::allreduce(send_buf,
                                       recv_buf,
                                       count,
                                       dtype,
                                       ccl::reduction::sum,
                                       *comm,
                                       *stream,
                                       attrs[kernel_idx]);
        }
        else {
            ccl_event = ccl::allreduce(send_buf,
                                       recv_buf,
                                       count,
                                       dtype,
                                       ccl::reduction::sum,
                                       *comm,
                                       *stream,
                                       attrs[kernel_idx],
                                       ccl_deps);
        }

        auto coll_end = std::chrono::high_resolution_clock::now();
        auto api_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(coll_end - coll_start).count();

        auto end_event = q.submit([&](auto &h) {
            h.depends_on(ccl_event.get_native());
            h.single_task([=]() {});
        });

        return std::make_shared<ccl_compute_result>(std::move(ccl_event),
                                                    std::move(start_event),
                                                    std::move(end_event),
                                                    api_time,
                                                    "allreduce");
    }

    void simulate(size_t iter_idx,
                  size_t kernel_idx,
                  size_t count,
                  std::unordered_map<buffer_type, T *> buffer_for) override {
        auto send_buf = buffer_for[buffer_type::weights];
        auto recv_buf = buffer_for[buffer_type::reduction];

        if (size == 1) {
            std::memcpy(recv_buf, send_buf, count * sizeof(T));
            return;
        }

        std::vector<T> gather_buf;
        if (rank == 0) {
            gather_buf.resize(count * size);

            for (size_t id = 0; id < count; id++) {
                recv_buf[id] = send_buf[id];
            }
        }

        MPI_Gather(send_buf,
                   count * sizeof(T),
                   MPI_BYTE,
                   gather_buf.data(),
                   count * sizeof(T),
                   MPI_BYTE,
                   0,
                   MPI_COMM_WORLD);

        // Manually executing the reduction step to match oneCCL's behavior closely,
        // especially for low precision data types like bf16 where the result is
        // highly sensitive to the operation order. This code mimics the topology-aware
        // algorithms found in oneCCL, so keep in mind it's tailored for that
        // environment.
        if (rank == 0) {
            for (size_t id = 0; id < count; id++) {
                for (size_t rank_id = 0; rank_id < size; rank_id += 2) {
                    gather_buf[id + count * rank_id] += gather_buf[id + count * (rank_id + 1)];
                }
                float accumulator = 0;
                for (size_t rank_id = 0; rank_id < size; rank_id += 2) {
                    accumulator += gather_buf[id + count * rank_id];
                }
                recv_buf[id] = accumulator;
            }
        }

        MPI_Bcast(recv_buf, count * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

private:
    size_t rank;
    ccl::communicator *comm;
    ccl::stream *stream;
    ccl::datatype dtype;
    std::vector<ccl::allreduce_attr> attrs;
    size_t size;
};

template <typename T>
class allgather_operation : public compute<T> {
public:
    allgather_operation(size_t rank,
                        ccl::communicator *comm,
                        ccl::stream *stream,
                        ccl::datatype dtype,
                        std::vector<ccl::allgatherv_attr> attrs,
                        size_t size)
            : rank(rank),
              comm(comm),
              stream(stream),
              dtype(dtype),
              attrs(attrs),
              size(size) {}

    std::shared_ptr<compute_result> submit(sycl::queue q,
                                           size_t iter_idx,
                                           size_t kernel_idx,
                                           size_t count,
                                           std::unordered_map<buffer_type, T *> buffer_for,
                                           std::vector<sycl::event> deps) override {
        auto send_buf = buffer_for[buffer_type::weights];
        auto recv_buf = buffer_for[buffer_type::wide];
        ccl::event ccl_event;
        std::vector<size_t> recv_counts(size, count);
        std::vector<ccl::event> ccl_deps;
        for (auto e : deps) {
            ccl_deps.push_back(ccl::create_event(e));
        }

        auto start_event = q.submit([&](auto &h) {
            h.depends_on(deps);
            h.single_task([=]() {});
        });

        auto coll_start = std::chrono::high_resolution_clock::now();
        if (ccl_deps.empty()) {
            ccl_event = ccl::allgatherv(
                send_buf, count, recv_buf, recv_counts, dtype, *comm, *stream, attrs[kernel_idx]);
        }
        else {
            ccl_event = ccl::allgatherv(send_buf,
                                        count,
                                        recv_buf,
                                        recv_counts,
                                        dtype,
                                        *comm,
                                        *stream,
                                        attrs[kernel_idx],
                                        ccl_deps);
        }

        auto coll_end = std::chrono::high_resolution_clock::now();
        auto api_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(coll_end - coll_start).count();

        auto end_event = q.submit([&](auto &h) {
            h.depends_on(ccl_event.get_native());
            h.single_task([=]() {});
        });

        return std::make_shared<ccl_compute_result>(std::move(ccl_event),
                                                    std::move(start_event),
                                                    std::move(end_event),
                                                    api_time,
                                                    "allgatherv");
    }

    void simulate(size_t iter_idx,
                  size_t kernel_idx,
                  size_t count,
                  std::unordered_map<buffer_type, T *> buffer_for) override {
        auto send_buf = buffer_for[buffer_type::weights];
        auto recv_buf = buffer_for[buffer_type::wide];

        MPI_Allgather(send_buf,
                      count * sizeof(T),
                      MPI_BYTE,
                      recv_buf,
                      count * sizeof(T),
                      MPI_BYTE,
                      MPI_COMM_WORLD);
    }

private:
    size_t rank;
    ccl::communicator *comm;
    ccl::stream *stream;
    ccl::datatype dtype;
    std::vector<ccl::allgatherv_attr> attrs;
    size_t size;
    ;
};

template <typename T>
class reduce_scatter_operation : public compute<T> {
public:
    reduce_scatter_operation(size_t rank,
                             ccl::communicator *comm,
                             ccl::stream *stream,
                             ccl::datatype dtype,
                             std::vector<ccl::reduce_scatter_attr> attrs,
                             size_t size)
            : rank(rank),
              comm(comm),
              stream(stream),
              dtype(dtype),
              attrs(attrs),
              size(size) {}

    std::shared_ptr<compute_result> submit(sycl::queue q,
                                           size_t iter_idx,
                                           size_t kernel_idx,
                                           size_t count,
                                           std::unordered_map<buffer_type, T *> buffer_for,
                                           std::vector<sycl::event> deps) override {
        auto send_buf = buffer_for[buffer_type::wide];
        auto recv_buf = buffer_for[buffer_type::weights];
        ccl::event ccl_event;
        std::vector<ccl::event> ccl_deps;

        auto start_event = q.submit([&](auto &h) {
            h.depends_on(deps);
            h.single_task([=]() {});
        });

        auto coll_start = std::chrono::high_resolution_clock::now();
        if (ccl_deps.empty()) {
            ccl_event = ccl::reduce_scatter(send_buf,
                                            recv_buf,
                                            count,
                                            dtype,
                                            ccl::reduction::sum,
                                            *comm,
                                            *stream,
                                            attrs[kernel_idx]);
        }
        else {
            ccl_event = ccl::reduce_scatter(send_buf,
                                            recv_buf,
                                            count,
                                            dtype,
                                            ccl::reduction::sum,
                                            *comm,
                                            *stream,
                                            attrs[kernel_idx],
                                            ccl_deps);
        }

        auto coll_end = std::chrono::high_resolution_clock::now();
        auto api_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(coll_end - coll_start).count();

        auto end_event = q.submit([&](auto &h) {
            h.depends_on(ccl_event.get_native());
            h.single_task([=]() {});
        });

        return std::make_shared<ccl_compute_result>(std::move(ccl_event),
                                                    std::move(start_event),
                                                    std::move(end_event),
                                                    api_time,
                                                    "reduce scatter");
    }

    void simulate(size_t iter_idx,
                  size_t kernel_idx,
                  size_t count,
                  std::unordered_map<buffer_type, T *> buffer_for) override {
        size_t wide_count = count * size;
        auto send_buf = buffer_for[buffer_type::wide];
        auto recv_buf = buffer_for[buffer_type::weights];

        std::vector<T> gather_buf;
        if (rank == 0) {
            gather_buf.resize(wide_count * size);
        }

        MPI_Gather(send_buf,
                   wide_count * sizeof(T),
                   MPI_BYTE,
                   gather_buf.data(),
                   wide_count * sizeof(T),
                   MPI_BYTE,
                   0,
                   MPI_COMM_WORLD);

        // Manually executing the reduction step to match oneCCL's behavior closely,
        // especially for low precision data types like bf16 where the result is
        // highly sensitive to the operation order. This code mimics the topology-aware
        // algorithms found in oneCCL, so keep in mind it's tailored for that
        // environment.
        if (rank == 0) {
            for (size_t id = 0; id < wide_count; id++) {
                for (size_t rank_id = 0; rank_id < size; rank_id += 2) {
                    gather_buf[id + wide_count * rank_id] +=
                        gather_buf[id + wide_count * (rank_id + 1)];
                }
                float accumulator = 0;
                for (size_t rank_id = 0; rank_id < size; rank_id += 2) {
                    accumulator += gather_buf[id + wide_count * rank_id];
                }
                gather_buf[id] = accumulator;
            }
        }

        // Where possible we are using `MPI_BYTE`, but nature of scatter requires that we match
        // size of `sendtype` and real size of `T`. Not doing this would for example scatter
        // one float32 over 4 ranks which would yield incorrect values.
        MPI_Datatype type;
        MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(T), &type);
        MPI_Scatter(gather_buf.data(), count, type, recv_buf, count, type, 0, MPI_COMM_WORLD);
    }

private:
    size_t rank;
    ccl::communicator *comm;
    ccl::stream *stream;
    ccl::datatype dtype;
    std::vector<ccl::reduce_scatter_attr> attrs;
    size_t size;
};
