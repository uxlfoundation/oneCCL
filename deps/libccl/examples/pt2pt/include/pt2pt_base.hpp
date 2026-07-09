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

#include <getopt.h>

#include <iomanip>
#include <iostream>
#include <sstream>

#include "types.hpp"

typedef struct user_options_t {
    backend_type_t backend;
    uint32_t cache;
    uint32_t iters;

    std::vector<int> peers;
    uint32_t queue;
    size_t min_elem_count;
    size_t max_elem_count;
    std::list<size_t> elem_counts;
    check_values_t check;
    uint32_t warmup_iters;
    uint32_t wait;
    int window_size;

    bool min_elem_count_set;
    bool max_elem_count_set;
    bool elem_counts_set;

    user_options_t() {
        backend = DEFAULT_BACKEND;
        iters = DEFAULT_ITERS;
        warmup_iters = DEFAULT_WARMUP_ITERS;
        cache = DEFAULT_CACHE_OPS;
        queue = DEFAULT_QUEUE;
        wait = DEFAULT_WAIT;
        min_elem_count = DEFAULT_MIN_ELEM_COUNT;
        max_elem_count = DEFAULT_MAX_ELEM_COUNT;
        fill_elem_counts(elem_counts, min_elem_count, max_elem_count);
        check = DEFAULT_CHECK;
        // for bw benchmark
        window_size = DEFAULT_WINDOW_SIZE;

        peers.reserve(2);
        // filling out with the default values
        peers.push_back(0);
        peers.push_back(1);

        min_elem_count_set = false;
        max_elem_count_set = false;
        elem_counts_set = false;
    }
} user_options_t;

void adjust_elem_counts(user_options_t& options) {
    if (options.max_elem_count < options.min_elem_count) {
        options.max_elem_count = options.min_elem_count;
    }

    if (options.elem_counts_set) {
        /* adjust min/max_elem_count or elem_counts */
        if (options.min_elem_count_set) {
            /* apply user-supplied count as limiter */
            options.elem_counts.remove_if([&options](const size_t& count) {
                return (count < options.min_elem_count);
            });
        }
        else {
            if (options.elem_counts.empty()) {
                options.min_elem_count = DEFAULT_MIN_ELEM_COUNT;
            }
            else {
                options.min_elem_count =
                    *(std::min_element(options.elem_counts.begin(), options.elem_counts.end()));
            }
        }
        if (options.max_elem_count_set) {
            /* apply user-supplied count as limiter */
            options.elem_counts.remove_if([&options](const size_t& count) {
                return (count > options.max_elem_count);
            });
        }
        else {
            if (options.elem_counts.empty()) {
                options.max_elem_count = options.min_elem_count;
            }
            else {
                options.max_elem_count =
                    *(std::max_element(options.elem_counts.begin(), options.elem_counts.end()));
            }
        }
    }
    else {
        fill_elem_counts(options.elem_counts, options.min_elem_count, options.max_elem_count);
    }
}

int set_backend(const std::string& option_value, backend_type_t& backend) {
    std::string option_name = "backend";
    std::set<std::string> supported_option_values{ backend_names[BACKEND_CPU] };

#ifdef CCL_ENABLE_SYCL
    supported_option_values.insert(backend_names[BACKEND_GPU]);
#endif // CCL_ENABLE_SYCL

    if (check_supported_options(option_name, option_value, supported_option_values))
        return -1;

    backend = (option_value == backend_names[BACKEND_GPU]) ? BACKEND_GPU : BACKEND_CPU;
    return 0;
}

int parse_user_options(int& argc, char**(&argv), user_options_t& options) {
    int ch;
    int errors = 0;
    std::list<int> elem_counts_int;

    char short_options[1024] = { 0 };
    const char* base_options = "b:i:w:p:e:s:f:t:y:c:m:h";
    memcpy(short_options, base_options, strlen(base_options));

    struct option getopt_options[] = {
        { "backend", required_argument, nullptr, 'b' },
        { "iters", required_argument, nullptr, 'i' },
        { "warmup_iters", required_argument, nullptr, 'w' },
        { "cache", required_argument, nullptr, 'p' },
        { "sycl_queue_type", required_argument, nullptr, 'e' },
        { "wait", required_argument, nullptr, 's' },
        { "min_elem_count", required_argument, nullptr, 'f' },
        { "max_elem_count", required_argument, nullptr, 't' },
        { "elem_counts", required_argument, nullptr, 'y' },
        { "check", required_argument, nullptr, 'c' },
        { "window", required_argument, nullptr, 'm' },
        { "help", no_argument, nullptr, 'h' },
        //TODO: { "peers", required_argument, nullptr, 'p' },
        { nullptr, 0, nullptr, 0 } // required at end of array.
    };

    while ((ch = getopt_long(argc, argv, short_options, getopt_options, nullptr)) != -1) {
        switch (ch) {
            case 'b':
                if (set_backend(optarg, options.backend)) {
                    PRINT("failed to parse 'backend' option");
                    errors++;
                }
                break;
            case 'i':
                if (is_valid_integer_option(optarg)) {
                    options.iters = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 'w':
                if (is_valid_integer_option(optarg)) {
                    options.warmup_iters = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 'p':
                if (is_valid_integer_option(optarg)) {
                    options.cache = atoll(optarg);
                    if (options.cache) {
                        PRINT(
                            "Warning: Caching mode is not supported at the moment. Option has been disabled");
                        options.cache = 0;
                    }
                }
                else
                    errors++;
                break;
            case 'e':
                if (is_valid_integer_option(optarg)) {
                    options.queue = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 'f':
                if (is_valid_integer_option(optarg)) {
                    options.min_elem_count = atoll(optarg);
                    options.min_elem_count_set = true;
                    if (options.min_elem_count == 0) {
                        // bandwidth can only be tested for message sizes > 0
                        options.min_elem_count = DEFAULT_MIN_ELEM_COUNT;
                        PRINT(
                            "Warning: Minimum element count for bandwidth test should be greater than 0");
                    }
                }
                else
                    errors++;
                break;
            case 't':
                if (is_valid_integer_option(optarg)) {
                    options.max_elem_count = atoll(optarg);
                    options.max_elem_count_set = true;
                }
                else
                    errors++;
                break;
            case 'y':
                elem_counts_int = tokenize<int>(optarg, ',');
                elem_counts_int.remove_if([](const size_t& count) {
                    if (count == 0) {
                        PRINT("Warning: Element count for bandwidth test should be greater than 0");
                    }
                    return !(is_valid_integer_option(count) && count > 0);
                });
                options.elem_counts = tokenize<size_t>(optarg, ',');
                if (elem_counts_int.size() == options.elem_counts.size())
                    options.elem_counts_set = true;
                else
                    errors++;
                break;
            case 's':
                if (is_valid_integer_option(optarg)) {
                    options.wait = atoll(optarg);
                    if (options.wait == 0) {
                        PRINT(
                            "Warning: Non-blocking mode is not supported, fallback to blocking mode");
                        options.wait = 1;
                    }
                }
                else
                    errors++;
                break;
            case 'c':
                if (set_check_values(optarg, options.check)) {
                    PRINT("failed to parse 'check' option");
                    errors++;
                }
                break;
            case 'm':
                if (is_valid_integer_option(optarg)) {
                    options.window_size = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 'h': return INVALID_RETURN;
            default:
                PRINT("failed to parse unknown option");
                errors++;
                break;
        }
    }

    if (optind < argc) {
        PRINT("non-option ARGV-elements given");
        errors++;
    }

    if (errors > 0) {
        PRINT("found %d errors while parsing user options", errors);
        for (int idx = 0; idx < argc; idx++) {
            PRINT("arg %d: %s", idx, argv[idx]);
        }
        return -1;
    }

    adjust_elem_counts(options);

    return 0;
}

ccl::pt2pt_attr create_attr(const bool is_cache,
                            const int count,
                            const std::string& match_id_suffix) {
    auto attr = ccl::create_operation_attr<ccl::pt2pt_attr>();
    if (is_cache) {
        std::string matchId = "_len_" + std::to_string(count) + match_id_suffix;
        attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(matchId));
        attr.set<ccl::operation_attr_id::to_cache>(true);
    }
    return attr;
}

void print_timings(ccl::communicator& comm,
                   const user_options_t& options,
                   const double total_time,
                   const int count,
                   const std::string mesure_str) {
    static bool print_once = false;

    if (!print_once && comm.rank() == 0) {
        std::stringstream ss;
        ss << std::right << std::setw(COL_WIDTH - 4) << "#bytes" << std::setw(COL_WIDTH)
           << "#repetitions" << std::setw(COL_WIDTH) << mesure_str << std::endl;
        std::cout << ss.str();
        print_once = true;
    }

    std::stringstream ss;

    ss << std::right << std::fixed << std::setw(COL_WIDTH - 4) << count << std::setw(COL_WIDTH)
       << options.iters << std::setw(COL_WIDTH) << std::setprecision(COL_PRECISION) << total_time
       << std::setw(COL_WIDTH) << std::endl;
    std::cout << ss.str();
}

void check_cpu_buffers(const int count, const size_t iter_idx, std::vector<int>& buf_recv) {
    bool failed = false;
    std::vector<int> check_buf(count);

    for (auto id = 0; id < count; id++) {
        if (buf_recv[id] != static_cast<int>(id + iter_idx)) {
            check_buf[id] = INVALID_VALUE;
        }
    }

    for (int j = 0; j < count; j++) {
        if (check_buf[j] == INVALID_VALUE) {
            failed = true;
            break;
        }
    }

    if (failed) {
        std::cout << "FAILED: iter_idx: " << iter_idx << ", count: " << count << std::endl;
        ASSERT(0, "unexpected value");
    }
}

#ifdef CCL_ENABLE_SYCL
void check_gpu_buffers(sycl::queue q,
                       const user_options_t& options,
                       const int count,
                       const size_t iter_idx,
                       int* buf_recv,
                       std::vector<ccl::event>& ccl_events) {
    bool failed = false;
    sycl::buffer<int> check_buf(count);

    auto e = q.submit([&](auto& h) {
        sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
        if (!options.queue && !options.wait) {
            h.depends_on(ccl_events.back().get_native());
        }
        // check_buf_acc moved, not used any more
        h.parallel_for(count, [=, check_buf_acc = std::move(check_buf_acc)](auto id) {
            if (buf_recv[id] != static_cast<int>(id + iter_idx)) {
                check_buf_acc[id] = INVALID_VALUE;
            }
        });
    });

    if (options.wait) {
        e.wait_and_throw();
    }

    {
        sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
        for (int j = 0; j < count; j++) {
            if (check_buf_acc[j] == INVALID_VALUE) {
                failed = true;
                break;
            }
        }
    }

    if (failed) {
        std::cout << "FAILED: iter_idx: " << iter_idx << ", count: " << count << std::endl;
        ASSERT(0, "unexpected value");
    }
}
#endif // CCL_ENABLE_SYCL

void print_help_usage(const char* app) {
    PRINT(
        "\nUSAGE:\n"
        "\t%s [OPTIONS]\n\n"
        "OPTIONS:\n"
        "\t[-b,--backend <backend>]: %s\n"
        "\t[-i,--iters <iteration count>]: %d\n"
        "\t[-w,--warmup_iters <warm up iteration count>]: %d\n"
        "\t[-p,--cache <use persistent operations>]: %d\n"
        "\t[-e,--sycl_queue_type <sycl queue mode in/out order>]: %d\n"
        "\t[-s,--wait <enable synchronization on sycl and pt2pt level>]: %d\n"
        "\t[-f,--min_elem_count <minimum element count>]: %d\n"
        "\t[-t,--max_elem_count <maximum element count>]: %d\n"
        "\t[-y,--elem_counts <list of element counts>]: [%d-%d]\n"
        "\t[-c,--check <validate result correctness>]: %s\n"
        "\t[-h,--help]\n\n"
        "example:\n\t--backend gpu --sycl_queue_type 1 --cache 0 --check 1 --elem_counts 64,1024\n",
        app,
        backend_names[DEFAULT_BACKEND].c_str(),
        DEFAULT_ITERS,
        DEFAULT_WARMUP_ITERS,
        DEFAULT_CACHE_OPS,
        DEFAULT_QUEUE,
        DEFAULT_WAIT,
        DEFAULT_MIN_ELEM_COUNT,
        DEFAULT_MAX_ELEM_COUNT,
        // elem_counts requires 2 values, min and max
        DEFAULT_MIN_ELEM_COUNT,
        DEFAULT_MAX_ELEM_COUNT,
        check_values_names[DEFAULT_CHECK].c_str());
}

template <class Dtype, class Iter>
std::string get_values_str(Iter first,
                           Iter last,
                           const char* opening,
                           const char* ending,
                           const char* delim) {
    std::stringstream ss;
    ss.str("");
    ss << opening;
    std::copy(first, last, std::ostream_iterator<Dtype>(ss, delim));
    if (*ss.str().rbegin() == ' ') {
        ss.seekp(-1, std::ios_base::end);
    }
    ss << ending;

    return ss.str();
}

void print_user_options(const std::string benchmark,
                        const user_options_t& options,
                        const ccl::communicator& comm) {
    std::stringstream ss;

    std::string backend_str = find_str_val(backend_names, options.backend);

    std::string elem_counts_str = get_values_str<size_t>(
        options.elem_counts.begin(), options.elem_counts.end(), "[", "]", " ");

    std::string check_values_str = find_str_val(check_values_names, options.check);

    ss.str("");
    ss << "\noptions:"
       << "\n  backend:        " << backend_str << "\n  iters:          " << options.iters
       << "\n  warmup_iters:   " << options.warmup_iters << "\n  cache:          " << options.cache;
    if (options.backend == BACKEND_GPU) {
        ss << "\n  sycl_queue_type:          " << options.queue
           << "\n  wait:           " << options.wait;
    }

    ss << "\n  min_elem_count: " << options.min_elem_count
       << "\n  max_elem_count: " << options.max_elem_count
       << "\n  elem_counts:    " << elem_counts_str << "\n  check:       " << check_values_str;

    if (benchmark == "Bandwidth") {
        ss << "\n  window_size:    " << options.window_size;
    }

    if (comm.rank() == 0) {
        std::cout << ss.str() << std::endl;
    }

    ss.str("");

    ss << "#------------------------------------------------------------\n"
       << "# Benchmarking: " << benchmark << "\n"
       << "# #processes: " << comm.size() << "\n"
       << "#------------------------------------------------------------\n";

    if (comm.rank() == 0) {
        std::cout << ss.str() << std::endl;
    }
}
