/*
 Copyright 2016-2020 Intel Corporation
 
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

// enums
typedef enum { CHECK_OFF, CHECK_LAST_ITER, CHECK_ALL_ITERS } check_values_t;

std::map<check_values_t, std::string> check_values_names = {
    std::make_pair(CHECK_OFF, "off"),
    std::make_pair(CHECK_LAST_ITER, "last"),
    std::make_pair(CHECK_ALL_ITERS, "all")
};

typedef enum { BACKEND_CPU, BACKEND_GPU } backend_type_t;
std::map<backend_type_t, std::string> backend_names = { std::make_pair(BACKEND_CPU, "cpu"),
                                                        std::make_pair(BACKEND_GPU, "gpu") };

// defines
#ifdef CCL_ENABLE_SYCL
#define DEFAULT_BACKEND BACKEND_GPU
#else // CCL_ENABLE_SYCL
#define DEFAULT_BACKEND BACKEND_CPU
#endif // CCL_ENABLE_SYCL
#define COL_WIDTH     (18)
#define COL_PRECISION (2)

#define DEFAULT_ITERS        (16)
#define DEFAULT_WARMUP_ITERS (16)

#define DEFAULT_CACHE_OPS   (0)
#define DEFAULT_QUEUE       (1) // in order queue
#define DEFAULT_WAIT        (1)
#define DEFAULT_WINDOW_SIZE (64)

#define DEFAULT_MIN_ELEM_COUNT (1)
#define DEFAULT_MAX_ELEM_COUNT (33554432) // till 128 MBytes
#define DEFAULT_CHECK          (CHECK_LAST_ITER)

#define INVALID_VALUE  (-1)
#define INVALID_RETURN (-1)

int set_check_values(const std::string& option_value, check_values_t& check) {
    std::string option_name = "check";

    std::set<std::string> supported_option_values{ check_values_names[CHECK_OFF],
                                                   check_values_names[CHECK_LAST_ITER],
                                                   check_values_names[CHECK_ALL_ITERS] };

    if (check_supported_options(option_name, option_value, supported_option_values)) {
        return INVALID_RETURN;
    }

    if (option_value == check_values_names[CHECK_OFF]) {
        check = CHECK_OFF;
    }
    else if (option_value == check_values_names[CHECK_LAST_ITER]) {
        check = CHECK_LAST_ITER;
    }
    else if (option_value == check_values_names[CHECK_ALL_ITERS]) {
        check = CHECK_ALL_ITERS;
    }

    return 0;
}

template <typename T>
std::list<T> tokenize(const std::string& input, char delimeter) {
    std::istringstream ss(input);
    std::list<T> ret;
    std::string str;
    while (std::getline(ss, str, delimeter)) {
        std::stringstream converter;
        converter << str;
        T value;
        converter >> value;
        ret.push_back(value);
    }
    return ret;
}

void fill_elem_counts(std::list<size_t>& counts, const size_t min_count, const size_t max_count) {
    counts.clear();
    size_t count = 0;
    for (count = min_count; count <= max_count; count *= 2) {
        counts.push_back(count);
    }
    if (*counts.rbegin() != max_count) {
        counts.push_back(max_count);
    }
}
