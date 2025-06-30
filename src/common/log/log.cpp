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
#include <execinfo.h>
#include <optional>
#include <sys/syscall.h>
#include <unistd.h>

#include "common/log/log.hpp"

ccl_log_level ccl_logger::level = ccl_log_level::warn;
bool ccl_logger::abort_on_throw = false;
ccl_logger logger;

// TODO: add a label named profile?
std::map<ccl_log_level, std::string> ccl_logger::level_names = {
    std::make_pair(ccl_log_level::error, "error"),
    std::make_pair(ccl_log_level::warn, "warn"),
    std::make_pair(ccl_log_level::info, "info"),
    std::make_pair(ccl_log_level::debug, "debug"),
    std::make_pair(ccl_log_level::trace, "trace")
};

std::ostream& operator<<(std::ostream& os, ccl_streambuf& buf) {
    buf.set_eol();
    os << buf.buffer.get();
    buf.reset();
    return os;
}

// Try to fetch int from an environment variable `env_var_name`.
// If It is not present or getenv/stoi calls threw an exception
// returns nullopt.
static int get_int_from_env(const char* env_var_name) {
    try {
        const char* var_str = std::getenv(env_var_name);

        if (var_str) {
            return std::stoi(var_str);
        }
    }
    catch (...) {
    }

    return ccl::utils::invalid_err_code;
}

// Try to fetch rank based on current process environment.
// Returns nullopt in case the rank could not be deduced.
static int get_rank_from_env() {
    int rank = ccl::utils::invalid_rank;

    // Check IMPI
    if ((rank = get_int_from_env("PMI_RANK")) != ccl::utils::invalid_rank) {
        return rank;
    }

    // Check MPICH
    if ((rank = get_int_from_env("PMIX_RANK")) != ccl::utils::invalid_rank) {
        return rank;
    }

    // Check torch
    if ((rank = get_int_from_env("RANK")) != ccl::utils::invalid_rank) {
        return rank;
    }

    // Check rank set manually by the user
    if ((rank = get_int_from_env("CCL_LOCAL_RANK")) != ccl::utils::invalid_rank) {
        return rank;
    }

    return ccl::utils::invalid_rank;
}

CCL_API ccl_logger& ccl_logger::get_instance() {
    return logger;
}

CCL_API bool ccl_logger::is_abort_on_throw_enabled() {
    return abort_on_throw;
}

CCL_API ccl_log_level ccl_logger::get_log_level() noexcept {
    return level;
}

bool ccl_logger::is_root() {
    static thread_local bool rank_read = false;
    static thread_local int rank = ccl::utils::invalid_rank;

    if (!rank_read) {
        rank = get_rank_from_env();
        rank_read = true;
    }

    if (rank == ccl::utils::invalid_rank || rank == 0) {
        // The rank is root or we could not detect rank properly
        return true;
    }
    return false;
}

void ccl_logger::write_prefix(std::ostream& str) {
    constexpr size_t time_buf_size = 20;
    constexpr size_t tid_width = 5;
    time_t timer;
    char time_buf[time_buf_size]{};
    struct tm time_info;
    memset(&time_info, 0, sizeof(time_info));

    time(&timer);
    if (localtime_r(&timer, &time_info)) {
        strftime(time_buf, time_buf_size, "%Y:%m:%d-%H:%M:%S", &time_info);
        str << time_buf;
    }
    str << ":(" << std::setw(tid_width) << gettid() << ") ";
}

void ccl_logger::write_backtrace(std::ostream& str) {
    void* buffer[100];
    auto nptrs = backtrace(buffer, 100);
    str << "backtrace() returned " << nptrs << " addresses\n";

    auto strings = backtrace_symbols(buffer, nptrs);
    if (!strings) {
        str << "backtrace_symbols error\n";
        return;
    }

    for (int j = 0; j < nptrs; j++) {
        str << strings[j] << "\n";
    }
    free(strings);
}
