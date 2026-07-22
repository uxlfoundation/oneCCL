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

#include "mpi.h"
#include "oneapi/ccl/types.hpp"
#include "sycl_base.hpp"
#include "compute.hpp"

#include <getopt.h>
#include <iomanip>
#include <numeric>
#include <sstream>

using namespace std;
using namespace sycl;

const char *const GREEN = (getenv("NO_COLOR")) ? "" : "\033[1;32m";
const char *const BOLD = (getenv("NO_COLOR")) ? "" : "\033[1m";
const char *const RESET = (getenv("NO_COLOR")) ? "" : "\033[0m";

/* -------------------------------------------------------------------------- */
/*                              Helper structures                             */
/* -------------------------------------------------------------------------- */

enum class exec_mode { single_allreduce, multi_allreduce, sequence_parallel };

struct attrs_group {
    vector<ccl::allreduce_attr> allreduce = {};
    vector<ccl::allgatherv_attr> allgatherv = {};
    vector<ccl::reduce_scatter_attr> reduce_scatter = {};
};

string convert_arg(bool val) {
    if (val) {
        return "enabled";
    }
    else {
        return "disabled";
    }
}

string convert_arg(exec_mode val) {
    static unordered_map<exec_mode, string> convert = {
        { exec_mode::single_allreduce, "single_allreduce" },
        { exec_mode::multi_allreduce, "multi_allreduce" },
        { exec_mode::sequence_parallel, "sequence_parallel" }
    };
    return convert[val];
}

struct run_args {
    exec_mode mode = exec_mode::multi_allreduce;
    queue_type queue_type = queue_type::in_order;
    string data_type = "fp32";
    size_t count = 2 * 1024 * 1024; // 8mb of floats
    size_t iter_count = 20;
    size_t kernel_count = 15;
    size_t skip_iter_count = 0;
    unsigned verbose = 0;
    bool enable_cache = false;
    bool random = false;

    template <typename T>
    void print_arg(std::stringstream &ss, const std::string &label, const T &value) {
        const int width = 20;
        ss << BOLD << std::setw(width) << std::left << label << ": " << RESET << value << "\n";
    }

    void print() {
        std::stringstream ss;
        ss << GREEN << "Using parameters: \n" << RESET;

        print_arg(ss, "mode", convert_arg(mode));
        print_arg(ss, "queue-type", convert_queue_type(queue_type));
        print_arg(ss, "data-type", data_type);
        print_arg(ss, "count", count);
        print_arg(ss, "iter-count", iter_count);
        print_arg(ss, "kernel-count", kernel_count);
        print_arg(ss, "skip-iter-count", skip_iter_count);
        print_arg(ss, "cache", convert_arg(enable_cache));
        print_arg(ss, "verbose", convert_arg(verbose));
        print_arg(ss, "random", convert_arg(random));

        std::cout << ss.str() << std::endl;
    }
};

/* -------------------------------------------------------------------------- */
/*                        Commandline arguments parsing                       */
/* -------------------------------------------------------------------------- */

static run_args args;

bool process_args(int argc, char *argv[], run_args &test_args) {
    char c;

    const char *short_ops = "b:c:i:k:q:ps:vt:h";
    struct option ops[] = { { "mode", required_argument, nullptr, 'b' },
                            { "count", required_argument, nullptr, 'c' },
                            { "iter-count", required_argument, nullptr, 'i' },
                            { "kernel-count", required_argument, nullptr, 'k' },
                            { "queue-type", required_argument, nullptr, 'q' },
                            { "cache", no_argument, nullptr, 'p' },
                            { "skip-iter-count", required_argument, nullptr, 's' },
                            { "verbose", no_argument, nullptr, 'v' },
                            { "verbose-debug", no_argument, nullptr, '\0' },
                            { "random", no_argument, nullptr, 'r' },
                            { "data-type", required_argument, nullptr, 't' },
                            { "help", no_argument, nullptr, 'h' },
                            { nullptr, 0, nullptr, 0 } };

    while (true) {
        c = getopt_long(argc, argv, short_ops, ops, nullptr);
        if (c == -1)
            break;

        switch (c) {
            case 'b': {
                // 0 and 1 values are supported for backwards compatibility
                if (std::string(optarg) == "single_allreduce" || std::string(optarg) == "0") {
                    args.mode = exec_mode::single_allreduce;
                }
                else if (std::string(optarg) == "multi_allreduce" || std::string(optarg) == "1") {
                    args.mode = exec_mode::multi_allreduce;
                }
                else if (std::string(optarg) == "sequence_parallel") {
                    args.mode = exec_mode::sequence_parallel;
                }
                else {
                    std::cerr << "Invalid mode specified." << std::endl;
                    return false;
                }
                break;
            }
            case 'c': test_args.count = atoi(optarg); break;
            case 'i': test_args.iter_count = atoi(optarg); break;
            case 'k': test_args.kernel_count = atoi(optarg); break;
            case 'p': test_args.enable_cache = true; break;
            case 's': test_args.skip_iter_count = atoi(optarg); break;
            case 'v': test_args.verbose = 1; break;
            case 'r': test_args.random = true; break;
            case 'q':
                // 0 and 1 values are supported for backwards compatibilit
                if (std::string(optarg) == "in_order" || std::string(optarg) == "1") {
                    test_args.queue_type = queue_type::in_order;
                }
                else if (std::string(optarg) == "out_of_order" || std::string(optarg) == "0") {
                    test_args.queue_type = queue_type::out_of_order;
                }
                else {
                    std::cerr << "Invalid queue type specified." << std::endl;
                    return false;
                }
                break;
            case 't': test_args.data_type = static_cast<std::string>(optarg); break;
            case 0: test_args.verbose = 2; break;
            case 'h':
            default: return false;
        }
    }

    return true;
}

void print_help() {
    auto print_line = [=](std::stringstream &ss,
                          const std::string &flag,
                          const std::string &type,
                          const std::string &description,
                          const std::string &default_val) {
        ss << BOLD << std::left << std::setw(25) << flag << RESET << std::left << std::setw(30)
           << type << std::left << std::setw(50) << description << std::left << setfill(' ')
           << (default_val == "" ? "" : " (default: ") << default_val
           << (default_val == "" ? "\n" : ")\n");
    };

    std::stringstream ss;

    ss << GREEN << "Usage:" << RESET << "\n";

    print_line(
        ss, "-m, --mode", "<\"multi_allreduce\"", "Workload mode", std::to_string(args.count));
    print_line(ss, "", "\"single_allreduce\"", "", "");
    print_line(ss, "", "\"sequence_parallel\">", "", "");
    print_line(ss,
               "-q, --queue-type",
               "<\"in_order\"/\"out_of_order\">",
               "SYCL command queue type",
               convert_queue_type(args.queue_type));
    print_line(ss,
               "-t, --data-type",
               "<\"fp32\"/\"fp16\"/\"bf16\">",
               "Data type for computation",
               args.data_type);
    print_line(ss,
               "-c, --count",
               "<size_t>",
               "Number of elements in weights buffer",
               std::to_string(args.count));
    print_line(ss,
               "-i, --iter-count",
               "<size_t>",
               "Number of iterations",
               std::to_string(args.iter_count));
    print_line(ss,
               "-k, --kernel-count",
               "<size_t>",
               "Number of kernel pipelines per iteration",
               std::to_string(args.kernel_count));
    print_line(
        ss, "-p, --cache", "<flag>", "Enable or disable caching", convert_arg(args.enable_cache));
    print_line(ss,
               "-s, --skip-iter-count",
               "<size_t>",
               "Number of iterations to skip before syncing",
               std::to_string(args.skip_iter_count));
    print_line(ss, "-v, --verbose", "<flag>", "Enable verbose output", convert_arg(args.verbose));
    print_line(ss, "--verbose-debug", "<flag>", "Enable all output", convert_arg(args.verbose));
    print_line(ss,
               "-r, --random",
               "<flag>",
               "Initialize buffer with pseudo-random values",
               convert_arg(args.random));
    ss << BOLD << std::setw(55) << std::left << "-h, --help" << RESET
       << "Print this help message\n";

    std::cout << ss.str() << std::endl;
}

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */

template <typename T>
using iteration = vector<pair<compute_mode, std::shared_ptr<compute<T>>>>;

optional<tuple<int, int, queue, unique_ptr<ccl::communicator>, unique_ptr<ccl::stream>>> init() {
    int size = 0;
    int rank = 0;

    ccl::init();
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    sycl::property_list props;
    if (args.queue_type == queue_type::in_order) {
        props = { sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} };
    }
    else {
        props = { sycl::property::queue::enable_profiling{} };
    }

    sycl::queue q;
    if (!create_sycl_queue("gpu", rank, q, props)) {
        return nullopt;
    }

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = make_unique<ccl::communicator>(ccl::create_communicator(size, rank, dev, ctx, kvs));
    auto stream = make_unique<ccl::stream>(ccl::create_stream(q));

    return std::tuple(rank, size, std::move(q), std::move(comm), std::move(stream));
}

template <typename attr_type>
void fill_attrs(vector<attr_type> &attrs, const char *name, bool cache, size_t kernel_count) {
    for (size_t kernel_idx = 0; kernel_idx < kernel_count; ++kernel_idx) {
        attrs.emplace_back(ccl::create_operation_attr<attr_type>());
        if (args.enable_cache) {
            auto &attr = attrs.back();

            attr.set<ccl::operation_attr_id::to_cache>(true);

            ccl::string_class match_id = name;
            match_id = match_id + std::to_string(kernel_idx);
            attr.set<ccl::operation_attr_id::match_id>(match_id);
        }
    }
}

attrs_group declare_attrs(size_t kernel_count) {
    attrs_group attrs_for;
    fill_attrs(attrs_for.allreduce, "allreduce", args.enable_cache, kernel_count);
    fill_attrs(attrs_for.allgatherv, "allgatherv", args.enable_cache, kernel_count);
    fill_attrs(attrs_for.reduce_scatter, "reduce_scatter", args.enable_cache, kernel_count);

    return attrs_for;
}

template <typename T>
iteration<T> declare_iteration(exec_mode mode,
                               int rank,
                               int size,
                               ccl::communicator *comm,
                               ccl::stream *stream,
                               ccl::datatype ccl_data_type,
                               const attrs_group &attrs_for) {
    iteration<T> it;
    auto ker = make_shared<kernel_operation<T>>(rank, args.random);
    auto allreduce = make_shared<allreduce_operation<T>>(
        rank, comm, stream, ccl_data_type, attrs_for.allreduce, size);
    auto update = make_shared<weights_update<T>>(rank, size);
    auto copy_reduction_to_weights =
        make_shared<copy_data<T>>(rank, size, buffer_type::reduction, buffer_type::weights);

    if (mode == exec_mode::multi_allreduce) {
        it.emplace_back(compute_mode::multi, std::move(ker));
        it.emplace_back(compute_mode::multi, std::move(allreduce));
        it.emplace_back(compute_mode::multi, std::move(copy_reduction_to_weights));
        it.emplace_back(compute_mode::multi, std::move(update));
    }
    else if (mode == exec_mode::single_allreduce) {
        it.emplace_back(compute_mode::multi, std::move(ker));
        it.emplace_back(compute_mode::single, std::move(allreduce));
        it.emplace_back(compute_mode::multi, std::move(copy_reduction_to_weights));
        it.emplace_back(compute_mode::multi, std::move(update));
    }
    else if (mode == exec_mode::sequence_parallel) {
        auto allgather = make_shared<allgather_operation<T>>(
            rank, comm, stream, ccl_data_type, attrs_for.allgatherv, size);
        auto reduce_scatter = make_shared<reduce_scatter_operation<T>>(
            rank, comm, stream, ccl_data_type, attrs_for.reduce_scatter, size);
        auto update_after_scatter = make_shared<weights_update<T>>(rank, size);

        it.emplace_back(compute_mode::multi, std::move(ker));
        it.emplace_back(compute_mode::multi, std::move(allgather));
        it.emplace_back(compute_mode::multi, std::move(allreduce));
        it.emplace_back(compute_mode::multi, std::move(copy_reduction_to_weights));
        it.emplace_back(compute_mode::multi, std::move(update));
        it.emplace_back(compute_mode::multi, std::move(reduce_scatter));
        it.emplace_back(compute_mode::multi, std::move(update_after_scatter));
    }
    return std::move(it);
}

template <typename T>
vector<tuple<T *, T *, T *>> allocate_buffers(buf_allocator<T> &allocator,
                                              exec_mode mode,
                                              size_t size,
                                              size_t count,
                                              size_t kernel_count) {
    // Store allocated mem ptrs to free them later
    // for single buffer, only 1 allocation is needed
    vector<tuple<T *, T *, T *>> ptrs(kernel_count);
    // allocate all the buffers
    if (mode == exec_mode::multi_allreduce) {
        for (size_t i = 0; i < kernel_count; i++) {
            T *weight_buf = allocator.allocate(count, usm::alloc::device);
            T *weight_allreduce_buf = allocator.allocate(count, usm::alloc::device);
            T *weight_wide_buf = allocator.allocate(count * size, usm::alloc::device);
            ptrs[i] = { weight_buf, weight_allreduce_buf, weight_wide_buf };
        }
    }
    else {
        T *weight_buf = allocator.allocate(count * kernel_count, usm::alloc::device);
        T *weight_allreduce_buf = allocator.allocate(count * kernel_count, usm::alloc::device);
        T *weight_wide_buf = allocator.allocate(count * size * kernel_count, usm::alloc::device);
        // in case of single buffer set all ptrs with the same buffers for consistency
        for (size_t i = 0; i < kernel_count; ++i) {
            ptrs[i] = { weight_buf + count * i,
                        weight_allreduce_buf + count * i,
                        weight_wide_buf + count * size * i };
        }
    }
    return ptrs;
}

template <typename T>
void deallocate_buffers(buf_allocator<T> &allocator,
                        exec_mode mode,
                        vector<tuple<T *, T *, T *>> ptrs) {
    if (mode == exec_mode::multi_allreduce) {
        for (auto [weights, reduction, wide] : ptrs) {
            allocator.deallocate(weights);
            allocator.deallocate(reduction);
            allocator.deallocate(wide);
        }
    }
    else {
        auto [weights, reduction, wide] = ptrs[0];
        allocator.deallocate(weights);
        allocator.deallocate(reduction);
        allocator.deallocate(wide);
    }
}

template <typename T>
// Submitting horizontally means that we submit ONLY ONE computation
// for all kernel pipelines. An example could be one allreduce after
// all compute kernels. This approach requires strict synchronization
// both before and after the computation, i.e we must wait for all
// previous steps across all pipelines and all pipelines must wait
// for the single computation unit.
void submit_horizontally(
    const iteration<T> &it,
    queue &q,
    size_t iter_idx,
    size_t count,
    size_t kernel_count,
    vector<tuple<T *, T *, T *>> &ptrs,
    unordered_map<size_t, vector<std::shared_ptr<compute_result>>> &computation_pipelines,
    vector<std::shared_ptr<compute_result>> &combined_results) {
    for (auto &[mode, computation] : it) {
        if (mode != compute_mode::single) {
            throw string("Cannot submit computation in mode different than `single` horizontally!");
        }

        // In single mode we wait for all computation steps across all kernels
        vector<event> deps;
        for (auto &[_, computation_list] : computation_pipelines) {
            if (computation_list.empty()) {
                continue;
            }
            deps.push_back(computation_list.back()->get_event());
        }

        auto [weights, reduction, wide] = ptrs[0];
        auto result = computation->submit(q,
                                          iter_idx,
                                          0,
                                          count * kernel_count,
                                          { { buffer_type::weights, weights },
                                            { buffer_type::reduction, reduction },
                                            { buffer_type::wide, wide } },
                                          { deps });

        // .. and each kernel coming after the single mode computation has to wait for the single event
        for (auto &[_, computation_results] : computation_pipelines) {
            computation_results.push_back(result);
        }
    }
}

template <typename T>
// Submitting computation vertically means that we submit all steps for one pipeline
// before we submit the following one. For example, if we have two pipelines with kernels
// and allreduce we will submit to the queue in order like:
//
//┌───────┐ ┌──────────┐ ┌────────┐ ┌───────┐ ┌──────────┐ ┌────────┐
//│KERNEL1│ │ALLREDUCE1│ │WEIGHTS1│ │KERNEL2│ │ALLREDUCE2│ │WEIGHTS2│
//└───────┘ └──────────┘ └────────┘ └───────┘ └──────────┘ └────────┘
//
void submit_vertically(
    const iteration<T> &it,
    queue &q,
    size_t iter_idx,
    size_t count,
    size_t kernel_count,
    vector<tuple<T *, T *, T *>> &ptrs,
    unordered_map<size_t, vector<std::shared_ptr<compute_result>>> &computation_pipelines,
    vector<std::shared_ptr<compute_result>> &combined_results) {
    for (size_t kernel_idx = 0; kernel_idx < kernel_count; kernel_idx++) {
        for (auto &[mode, computation] : it) {
            if (mode == compute_mode::single) {
                throw string("Cannot submit computation in `single` mode vertically!");
            }

            // Fetch vector of events only for one pipeline
            vector<std::shared_ptr<compute_result>> &computation_list =
                computation_pipelines[kernel_idx];
            vector<event> deps = {};
            if (!computation_list.empty() && args.queue_type == queue_type::out_of_order) {
                deps.push_back(computation_list.back()->get_event());
            }

            auto [weights, reduction, wide] = ptrs[kernel_idx];
            auto result = computation->submit(q,
                                              iter_idx,
                                              kernel_idx,
                                              count,
                                              { { buffer_type::weights, weights },
                                                { buffer_type::reduction, reduction },
                                                { buffer_type::wide, wide } },
                                              deps);

            computation_list.push_back(result);
        }
    }
}

template <typename T>
vector<std::shared_ptr<compute_result>> submit_iteration(const iteration<T> &it,
                                                         queue &q,
                                                         int rank,
                                                         size_t iter_idx,
                                                         size_t count,
                                                         size_t kernel_count,
                                                         vector<tuple<T *, T *, T *>> &ptrs) {
    // The map consists keeps record of all `compute_result`s based on
    // kernel_idx they belong to.
    unordered_map<size_t, vector<std::shared_ptr<compute_result>>> computation_pipelines;
    vector<std::shared_ptr<compute_result>> combined_results;

    // This object represents part of iteration containing computations only in `multi` mode,
    // which means that all of the computation units should be submitted vertically.
    iteration<T> vertical_part;
    for (auto &[mode, compute] : it) {
        if (mode == compute_mode::multi) {
            vertical_part.emplace_back(mode, compute);
        }
        else {
            // `compute` should be submitted horizontally, so we must submit
            // all previous compute units in `multi` kernel mode. We do that,
            // so we have all the events necessary to synchronize with single
            // mode compute.
            submit_vertically(vertical_part,
                              q,
                              iter_idx,
                              count,
                              kernel_count,
                              ptrs,
                              computation_pipelines,
                              combined_results);
            submit_horizontally({ { mode, compute } },
                                q,
                                iter_idx,
                                count,
                                kernel_count,
                                ptrs,
                                computation_pipelines,
                                combined_results);
            vertical_part.clear();
        }
    }

    if (!vertical_part.empty()) {
        submit_vertically(vertical_part,
                          q,
                          iter_idx,
                          count,
                          kernel_count,
                          ptrs,
                          computation_pipelines,
                          combined_results);
    }

    auto first = computation_pipelines[0].front();
    auto last = computation_pipelines[kernel_count - 1].back();
    auto iteration_result = make_shared<combined_sycl_compute_result>(
        combined_sycl_compute_result(first->get_event(), last->get_event(), 0, "iteration"));
    combined_results.emplace_back(std::move(iteration_result));

    // Record duration of each pipeline(kernel+allreduce+update)
    for (auto &[_, computation_list] : computation_pipelines) {
        auto pipeline_result = make_shared<combined_sycl_compute_result>(
            combined_sycl_compute_result(computation_list.front()->get_event(),
                                         computation_list.back()->get_event(),
                                         0,
                                         "single pipeline"));
        combined_results.emplace_back(std::move(pipeline_result));
    }

    // Combine all `compute_result`s into one vector for future
    // processing of timestamps
    for (auto &[_, computation_list] : computation_pipelines) {
        for (auto &result : computation_list) {
            combined_results.emplace_back(std::move(result));
        }

        computation_list.clear();
    }

    return combined_results;
}

void print_summary(int rank,
                   int size,
                   unordered_map<const char *, vector<tuple<size_t, size_t>>> timings) {
    for (int cur_rank = 0; cur_rank < size; cur_rank++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank != cur_rank) {
            continue;
        }
        for (const auto &entry : timings) {
            const char *name = entry.first;
            const auto &time_data = entry.second;

            if (time_data.empty())
                continue;

            size_t count = time_data.size();

            // Vectors to store individual execution and submission times
            std::vector<size_t> execution_times, submission_times;

            for (const auto &data : time_data) {
                execution_times.push_back(std::get<0>(data)); // Collect execution times
                submission_times.push_back(std::get<1>(data)); // Collect submission times
            }

            // Sorting to find min, max, and to calculate average from the best 90%
            std::sort(execution_times.begin(), execution_times.end());
            std::sort(submission_times.begin(), submission_times.end());

            // Calculating the average from the best 90%
            size_t best_90_count = static_cast<size_t>(std::ceil(count * 0.9));
            size_t execution_sum_best_90 = std::accumulate(
                execution_times.begin(), execution_times.begin() + best_90_count, 0);
            size_t submission_sum_best_90 = std::accumulate(
                submission_times.begin(), submission_times.begin() + best_90_count, 0);
            double average_execution_best_90 =
                static_cast<double>(execution_sum_best_90) / best_90_count;
            double average_submission_best_90 =
                static_cast<double>(submission_sum_best_90) / best_90_count;

            // Output metrics
            std::cout << "[" << rank << "]"
                      << "Name: " << name << std::endl;
            std::cout << "[" << rank << "]"
                      << "Min execution time: " << execution_times.front() / 1000 << "μs"
                      << std::endl;
            std::cout << "[" << rank << "]"
                      << "Max execution time: " << execution_times.back() / 1000 << "μs"
                      << std::endl;
            std::cout << "[" << rank << "]"
                      << "Min submission time: " << submission_times.front() / 1000 << "μs"
                      << std::endl;
            std::cout << "[" << rank << "]"
                      << "Max submission time: " << submission_times.back() / 1000 << "μs"
                      << std::endl;
            std::cout << "[" << rank << "]"
                      << "Average execution time (best 90%): " << average_execution_best_90 / 1000
                      << "μs" << std::endl;
            std::cout << "[" << rank << "]"
                      << "Average submission time (best 90%): " << average_submission_best_90 / 1000
                      << "μs" << std::endl;
            std::cout << std::endl; // Add an empty line for better readability
        }
    }
}

template <typename T>
int validate_results(const iteration<T> &it,
                     queue &q,
                     int rank,
                     int size,
                     vector<tuple<T *, T *, T *>> &ptrs) {
    // Copy results to compare with host simulation
    vector<T> host_weights(args.count);
    vector<T> host_reduction(args.count);
    vector<T> host_wide(args.count * size);

    vector<T> gpu_weights(args.count);
    vector<T> gpu_reduction(args.count);

    auto [weights, reduction, wide] = ptrs[args.kernel_count - 1];
    q.memcpy(gpu_weights.data(), weights, args.count * sizeof(T));
    q.memcpy(gpu_reduction.data(), reduction, args.count * sizeof(T));
    q.wait();

    // Simulate each computation, currently only for one kernel
    for (size_t iter_idx = 0; iter_idx < args.iter_count; iter_idx++) {
        for (auto &[_, computation] : it) {
            computation->simulate(iter_idx,
                                  args.kernel_count - 1,
                                  args.count,
                                  { { buffer_type::weights, host_weights.data() },
                                    { buffer_type::reduction, host_reduction.data() },
                                    { buffer_type::wide, host_wide.data() } });
        }
        if (args.verbose > 1) {
            print_array(rank, host_weights.data(), 16, "Host weights progress:");
        }
    }

    size_t error_count = 0;
    constexpr auto epsilon = 0.1f;
    for (size_t id = 0; id < args.count; id++) {
        if (std::abs(gpu_weights[id] - host_weights[id]) > epsilon) {
            error_count++;
            // Print each discrepancy.
            cerr << "Mismatch at position " << id << ": GPU weight = " << gpu_weights[id]
                 << ", Expected (Host) weight = " << host_weights[id] << std::endl;
        }
    }

    // Report the total number of discrepancies.
    if (error_count > 0) {
        cerr << "FAIL\n";
    }
    else {
        std::cout << "PASSED\n";
    }
    std::cout.flush();

    MPI_Barrier(MPI_COMM_WORLD);
    if (args.verbose > 1) {
        print_array_root(rank, host_weights.data(), 16, "Host weights:");
        print_array_root(rank, gpu_weights.data(), 16, "GPU weights:");
    }
    return error_count > 0 ? 1 : 0;
}

/* -------------------------------------------------------------------------- */
/*                               Execution logic                              */
/* -------------------------------------------------------------------------- */

template <typename T>
int execute(ccl::datatype ccl_data_type, size_t data_type_size) {
    auto init_result = init();
    if (!init_result.has_value()) {
        return -1;
    }

    auto [rank, size, queue, comm, stream] = std::move(init_result.value());
    auto attrs = declare_attrs(args.kernel_count);
    auto it =
        declare_iteration<T>(args.mode, rank, size, comm.get(), stream.get(), ccl_data_type, attrs);

    buf_allocator<T> allocator(queue);
    auto ptrs = allocate_buffers<T>(allocator, args.mode, size, args.count, args.kernel_count);

    // Store results of each compute for timings processing
    vector<std::shared_ptr<compute_result>> results;
    // Map each compute name onto vector of tuples with execution time and submission time
    unordered_map<const char *, vector<tuple<size_t, size_t>>> timings;

    for (size_t iter_idx = 0; iter_idx < args.iter_count; ++iter_idx) {
        auto result =
            submit_iteration(it, queue, rank, iter_idx, args.count, args.kernel_count, ptrs);
        results.insert(results.end(), result.begin(), result.end());

        if (args.skip_iter_count == 0 || (iter_idx != 0 && iter_idx % args.skip_iter_count == 0) ||
            iter_idx == args.iter_count - 1) {
            queue.wait();

            for (auto &compute : results) {
                if (args.verbose < compute->logging_level())
                    continue;

                timings[compute->get_name()].emplace_back(
                    compute->get_end_timestamp() - compute->get_start_timestamp(),
                    compute->get_submission_time());
            }
            results.clear();

            if (args.verbose > 1) {
                vector<T> gpu_weights(args.count);
                auto [weights, _, __] = ptrs[args.kernel_count - 1];
                queue.memcpy(gpu_weights.data(), weights, args.count * sizeof(T));
                queue.wait();
                print_array(rank, gpu_weights.data(), 16, "GPU weights progress:");
            }
        }
    }

    print_summary(rank, size, timings);

    int result = validate_results(it, queue, rank, size, ptrs);

    deallocate_buffers(allocator, args.mode, ptrs);
    return result;
}

int main(int argc, char *argv[]) {
    if (!process_args(argc, argv, args)) {
        print_help();
        return 1;
    }

    args.print();

    if (args.data_type == "fp16") {
        return execute<sycl::half>(ccl::datatype::float16, 2);
    }
    else if (args.data_type == "bf16") {
        return execute<sycl::ext::oneapi::bfloat16>(ccl::datatype::bfloat16, 2);
    }
    else if (args.data_type == "fp32") {
        return execute<float>(ccl::datatype::float32, 4);
    }
    else {
        printf("Unsupported data_type %s ", args.data_type.c_str());
        return -1;
    }
}
