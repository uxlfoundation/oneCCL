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

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include "benchmark.hpp"
#include "declarations.hpp"
#include "transport_impl.hpp"

inline void prepare_coll(const user_options_t& options,
                         ccl::communicator& service_comm,
                         std::shared_ptr<base_coll> coll,
                         const size_t elem_count) {
    coll->prepare(elem_count);
    ccl::barrier(service_comm);
}

inline void finalize_coll(const user_options_t& options,
                          ccl::communicator& service_comm,
                          std::shared_ptr<base_coll> coll,
                          const size_t elem_count) {
    ccl::barrier(service_comm);
    coll->finalize(elem_count);
}

void run(ccl::communicator& service_comm,
         bench_exec_attr& bench_attr,
         coll_list_t& all_colls,
         req_list_t& reqs,
         const user_options_t& options) {
    std::stringstream match_id_stream;

    for (auto dtype : all_dtypes) {
        coll_list_t colls;
        std::string dtype_name;

        std::copy_if(all_colls.begin(),
                     all_colls.end(),
                     std::back_inserter(colls),
                     [dtype](const typename coll_list_t::value_type coll) {
                         return dtype == coll->get_dtype();
                     });
        if (colls.empty())
            continue;

        dtype_name = find_str_val(dtype_names, dtype);

        for (const auto& reduction : options.reductions) {
            ccl::reduction reduction_op;

            if (!find_key_val(reduction_op, reduction_names, reduction))
                continue;

            PRINT_BY_ROOT(service_comm,
                          "\ndatatype: %s\nreduction: %s\n",
                          dtype_name.c_str(),
                          reduction.c_str());

            reqs.reserve(colls.size() * options.buf_count);

            bench_attr.reduction = reduction_op;
            bench_attr.set<ccl::operation_attr_id::to_cache>((bool)options.cache_ops);

            std::ostringstream scolls;
            std::copy(options.coll_names.begin(),
                      options.coll_names.end(),
                      std::ostream_iterator<std::string>{ scolls, " " });

            ccl::barrier(service_comm);

            /* benchmark with multiple equal sized buffer per collective */
            PRINT_BY_ROOT(service_comm,
                          "#------------------------------------------------------------\n"
                          "# Benchmarking: %s\n"
                          "# #processes: %d\n"
                          "#------------------------------------------------------------\n",
                          scolls.str().c_str(),
                          service_comm.size());

            if (service_comm.rank() == 0) {
                std::stringstream ss;
                ss << std::right << std::setw(COL_WIDTH) << "#bytes" << std::setw(COL_WIDTH)
                   << "#elem_count" << std::setw(COL_WIDTH) << "#repetitions"
                   << std::setw(COL_WIDTH) << "t_min[usec]" << std::setw(COL_WIDTH) << "t_max[usec]"
                   << std::setw(COL_WIDTH) << "t_avg[usec]" << std::setw(COL_WIDTH - 3)
                   << "stddev[%]" << std::setw(COL_WIDTH) << "algbw[GB/s]" << std::setw(COL_WIDTH)
                   << "busbw[GB/s]";

                if (show_extened_info(options.show_additional_info)) {
                    ss << std::right << std::setw(COL_WIDTH + 3) << "wait_t_avg[usec]";
                }
                ss << std::endl;
                printf("%s", ss.str().c_str());
            }

            for (auto& count : options.elem_counts) {
                size_t iter_count = get_iter_count(
                    count * ccl::get_datatype_size(dtype), options.iters, options.iter_policy);

                size_t warmup_iter_count = get_iter_count(count * ccl::get_datatype_size(dtype),
                                                          options.warmup_iters,
                                                          options.iter_policy);

                try {
                    // we store times for each collective separately,
                    // but aggregate over buffers and iterations
                    std::vector<double> total_timers(colls.size(), 0);
                    std::vector<double> wait_timers(colls.size(), 0);
                    for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++) {
                        auto& coll = colls[coll_idx];
                        double coll_time = 0, wait_time = 0;

                        ccl::barrier(service_comm);

                        for (size_t iter_idx = 0; iter_idx < (iter_count + warmup_iter_count);
                             iter_idx++) {
                            if (options.check_values == CHECK_ALL_ITERS) {
                                prepare_coll(options, service_comm, coll, count);
                            }

                            double coll_start_time = when();
                            for (size_t buf_idx = 0; buf_idx < options.buf_count; buf_idx++) {
                                if (options.cache_ops) {
                                    match_id_stream << "coll_" << coll->name() << "_" << coll_idx
                                                    << "_count_" << count << "_buf_" << buf_idx
                                                    << "_dt_" << dtype_name << "_rt_" << reduction;
                                    bench_attr.set<ccl::operation_attr_id::match_id>(
                                        ccl::string_class(match_id_stream.str()));
                                    match_id_stream.str("");
                                }
                                coll->start(count, buf_idx, bench_attr, reqs);
                            }
                            double coll_end_time = when();

                            double wait_start_time = when();
                            for (auto& req : reqs) {
                                req.wait();
                            }
                            double wait_end_time = when();
                            reqs.clear();

                            if (iter_idx >= warmup_iter_count) {
                                coll_time += coll_end_time - coll_start_time;
                                wait_time += wait_end_time - wait_start_time;
                                if (options.verbosity == 1) {
                                    printf("rank: %d count: %ld iter_idx: %ld time: %f\n",
                                           service_comm.rank(),
                                           count,
                                           iter_idx,
                                           coll_end_time - coll_start_time + wait_end_time -
                                               wait_start_time);
                                }
                            }

                            if (options.check_values == CHECK_ALL_ITERS) {
                                finalize_coll(options, service_comm, coll, count);
                            }
                        }

                        total_timers[coll_idx] += coll_time + wait_time;
                        wait_timers[coll_idx] += wait_time;

                        if (options.check_values == CHECK_LAST_ITER) {
                            prepare_coll(options, service_comm, coll, count);

                            for (size_t buf_idx = 0; buf_idx < options.buf_count; buf_idx++) {
                                if (options.cache_ops) {
                                    match_id_stream << "coll_" << coll->name() << "_" << coll_idx
                                                    << "_count_" << count << "_buf_" << buf_idx
                                                    << "_dt_" << dtype_name << "_rt_" << reduction;
                                    bench_attr.set<ccl::operation_attr_id::match_id>(
                                        ccl::string_class(match_id_stream.str()));
                                    match_id_stream.str("");
                                }
                                coll->start(count, buf_idx, bench_attr, reqs);
                            }

                            for (auto& req : reqs) {
                                req.wait();
                            }
                            reqs.clear();

                            finalize_coll(options, service_comm, coll, count);
                        }
                    }

                    print_timings(service_comm,
                                  total_timers,
                                  wait_timers,
                                  options,
                                  count,
                                  iter_count,
                                  dtype,
                                  reduction_op);
                }
                catch (const std::exception& ex) {
                    ASSERT(0, "error on count %zu, reason: %s", count, ex.what());
                }
            }
        }
    }

    ccl::barrier(service_comm);

    PRINT_BY_ROOT(service_comm, "\n# All done\n");
}

template <class Dtype>
void create_cpu_colls(bench_init_attr& init_attr, user_options_t& options, coll_list_t& colls) {
    std::stringstream error_messages_stream;

    for (auto names_it = options.coll_names.begin(); names_it != options.coll_names.end();) {
        const std::string& name = *names_it;
        if (name == allgather_strategy_impl::class_name()) {
            colls.emplace_back(new cpu_allgather_coll<Dtype>(init_attr));
        }
        else if (name == allgatherv_strategy_impl::class_name()) {
            colls.emplace_back(new cpu_allgatherv_coll<Dtype>(init_attr));
        }
        else if (name == allreduce_strategy_impl::class_name()) {
            colls.emplace_back(new cpu_allreduce_coll<Dtype>(init_attr));
        }
        else if (name == alltoall_strategy_impl::class_name()) {
            colls.emplace_back(new cpu_alltoall_coll<Dtype>(init_attr));
        }
        else if (name == alltoallv_strategy_impl::class_name()) {
            colls.emplace_back(new cpu_alltoallv_coll<Dtype>(init_attr));
        }
        else if (name == bcast_strategy_impl::class_name()) {
            colls.emplace_back(new cpu_bcast_coll<Dtype>(init_attr));
        }
        else if (name == broadcast_strategy_impl::class_name()) {
            colls.emplace_back(new cpu_broadcast_coll<Dtype>(init_attr));
        }
        else if (name == reduce_strategy_impl::class_name()) {
            colls.emplace_back(new cpu_reduce_coll<Dtype>(init_attr));
        }
        else if (name == reduce_scatter_strategy_impl::class_name()) {
            colls.emplace_back(new cpu_reduce_scatter_coll<Dtype>(init_attr));
        }
        else {
            ASSERT(0, "create_colls error, unknown coll name: %s", name.c_str());
        }
        ++names_it;
    }

    const std::string& coll_processing_log = error_messages_stream.str();
    if (!coll_processing_log.empty()) {
        std::cerr << "WARNING:\n" << coll_processing_log << std::endl;
    }

    if (colls.empty()) {
        throw std::logic_error(std::string(__FUNCTION__) +
                               " - empty colls, reason: " + coll_processing_log);
    }
}

#ifdef CCL_ENABLE_SYCL
template <class Dtype>
void create_sycl_colls(bench_init_attr& init_attr, user_options_t& options, coll_list_t& colls) {
    std::stringstream error_messages_stream;

    for (auto names_it = options.coll_names.begin(); names_it != options.coll_names.end();) {
        const std::string& name = *names_it;
        if (name == allgather_strategy_impl::class_name()) {
            colls.emplace_back(new sycl_allgather_coll<Dtype>(init_attr));
        }
        else if (name == allgatherv_strategy_impl::class_name()) {
            colls.emplace_back(new sycl_allgatherv_coll<Dtype>(init_attr));
        }
        else if (name == allreduce_strategy_impl::class_name()) {
            colls.emplace_back(new sycl_allreduce_coll<Dtype>(init_attr));
        }
        else if (name == alltoall_strategy_impl::class_name()) {
            colls.emplace_back(new sycl_alltoall_coll<Dtype>(init_attr));
        }
        else if (name == alltoallv_strategy_impl::class_name()) {
            colls.emplace_back(new sycl_alltoallv_coll<Dtype>(init_attr));
        }
        else if (name == bcast_strategy_impl::class_name()) {
            colls.emplace_back(new sycl_bcast_coll<Dtype>(init_attr));
        }
        else if (name == broadcast_strategy_impl::class_name()) {
            colls.emplace_back(new sycl_broadcast_coll<Dtype>(init_attr));
        }
        else if (name == reduce_strategy_impl::class_name()) {
            colls.emplace_back(new sycl_reduce_coll<Dtype>(init_attr));
        }
        else if (name == reduce_scatter_strategy_impl::class_name()) {
            colls.emplace_back(new sycl_reduce_scatter_coll<Dtype>(init_attr));
        }
        else {
            ASSERT(0, "create_colls error, unknown coll name: %s", name.c_str());
        }

        ++names_it;
    }

    const std::string& coll_processing_log = error_messages_stream.str();
    if (!coll_processing_log.empty()) {
        std::cerr << "WARNING: " << coll_processing_log << std::endl;
    }

    if (colls.empty()) {
        throw std::logic_error(std::string(__FUNCTION__) +
                               " - empty colls, reason: " + coll_processing_log);
    }
}
#endif // CCL_ENABLE_SYCL

template <class Dtype>
void create_colls(bench_init_attr& init_attr, user_options_t& options, coll_list_t& colls) {
    switch (options.backend) {
        case BACKEND_HOST: create_cpu_colls<Dtype>(init_attr, options, colls); break;
        case BACKEND_SYCL:
#ifdef CCL_ENABLE_SYCL
            create_sycl_colls<Dtype>(init_attr, options, colls);
#else
            ASSERT(0, "SYCL backend is requested but CCL_ENABLE_SYCL is not defined");
#endif
            break;
        default: ASSERT(0, "unknown backend %d", (int)options.backend); break;
    }
}

void create_all_colls(bench_init_attr& init_attr, user_options_t& options, coll_list_t& colls) {
    for (auto& dtype : options.dtypes) {
        if (dtype == dtype_names[ccl::datatype::int8])
            create_colls<int8_t>(init_attr, options, colls);
        else if (dtype == dtype_names[ccl::datatype::int32])
            create_colls<int32_t>(init_attr, options, colls);
        else if (dtype == dtype_names[ccl::datatype::int64])
            create_colls<int64_t>(init_attr, options, colls);
        else if (dtype == dtype_names[ccl::datatype::uint64])
            create_colls<uint64_t>(init_attr, options, colls);
        else if (dtype == dtype_names[ccl::datatype::float16])
            create_colls<ccl::float16>(init_attr, options, colls);
        else if (dtype == dtype_names[ccl::datatype::float32])
            create_colls<float>(init_attr, options, colls);
        else if (dtype == dtype_names[ccl::datatype::float64])
            create_colls<double>(init_attr, options, colls);
        else if (dtype == dtype_names[ccl::datatype::bfloat16])
            create_colls<ccl::bfloat16>(init_attr, options, colls);
        else
            ASSERT(0, "unexpected datatype %s", dtype.c_str());
    }
}

int main(int argc, char* argv[]) {
    user_options_t options;
    coll_list_t colls;
    req_list_t reqs;

    bench_init_attr init_attr;

    if (parse_user_options(argc, argv, options)) {
        print_help_usage(argv[0]);
        return -1;
    }

    auto& transport = transport_data::instance();
    transport.init_comms(options);

    ccl::communicator& service_comm = transport.get_service_comm();

    print_user_options(options, service_comm);

    init_attr.buf_count = options.buf_count;
    init_attr.max_elem_count = options.max_elem_count;
    init_attr.elem_offset = options.elem_offset;
    init_attr.ranks_per_proc = options.ranks_per_proc;
    init_attr.inplace = options.inplace;
    init_attr.numa_node = options.numa_node;
    init_attr.verbosity = options.verbosity;
#ifdef CCL_ENABLE_SYCL
    init_attr.sycl_mem_type = options.sycl_mem_type;
    init_attr.sycl_usm_type = options.sycl_usm_type;
#endif // CCL_ENABLE_SYCL

    try {
        create_all_colls(init_attr, options, colls);
    }
    catch (const std::runtime_error& e) {
        ASSERT(0, "cannot create coll objects: %s\n", e.what());
    }
    catch (const std::logic_error& e) {
        std::cerr << "Cannot launch benchmark: " << e.what() << std::endl;
        return -1;
    }

    bench_exec_attr bench_attr{};
    bench_attr.init_all();

    // open and truncate CSV file if csv-output is requested
    if (service_comm.rank() == 0 && !options.csv_filepath.empty()) {
        std::ofstream csvf;
        csvf.open(options.csv_filepath, std::ofstream::out | std::ofstream::trunc);
        if (!csvf.is_open()) {
            std::cerr << "cannot open CSV file for writing: " << options.csv_filepath << std::endl;
            abort();
        }

        // write header (column names)
        csvf << "#ranks,"
             << "collective,"
             << "reduction,"
             << "dtype,"
             << "dtype_size,"
             << "#elements/buffer,"
             << "message_size,"
             << "#buffers,"
             << "#repetitions,"
             << "t_min[usec],"
             << "t_max[usec],"
             << "t_avg[usec],"
             << "stddev[%],"
             << "wait_t_avg[usec],"
             << "algbw[GB/s],"
             << "busbw[GB/s]" << std::endl;
        csvf.close();
    }

    ccl::barrier(service_comm);

    run(service_comm, bench_attr, colls, reqs, options);

    ccl::barrier(service_comm);

    colls.clear();
    transport.reset_comms();

    return 0;
}
