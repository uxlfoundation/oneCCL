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

#include "internal/debug.h"
#include "oneapi/ccl.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <execinfo.h>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

int debug_level = -1; // not yet read from env
FILE *debug_file = nullptr;
std::mutex debug_lock;
std::string last_error;

constexpr size_t kErrorBufferLength = 256;
thread_local int debug_no_warn = 0;
thread_local char last_error_buffer[kErrorBufferLength];

int get_log_level_from_env() {
    const char *env_str = getenv("CCL_LOG_LEVEL");
    if (env_str == nullptr) {
        return ONECCL_LOG_WARN;
    }
    if (strcasecmp(env_str, "trace") == 0) {
        return ONECCL_LOG_TRACE;
    }
    if (strcasecmp(env_str, "info") == 0) {
        return ONECCL_LOG_INFO;
    }
    if (strcasecmp(env_str, "warn") == 0) {
        return ONECCL_LOG_WARN;
    }
    if (strcasecmp(env_str, "error") == 0) {
        return ONECCL_LOG_ERROR;
    }
    return ONECCL_LOG_WARN;
}

void get_timestamp(char *buffer, size_t maxSize) {
    time_t rawtime = 0;
    time(&rawtime);
    struct tm tm_buf;
    localtime_r(&rawtime, &tm_buf);
    strftime(buffer, maxSize, "%Y-%m-%d %H:%M:%S", &tm_buf);
}

void print_stack_trace(FILE *out) {
    const int max_frames = 64;
    std::vector<void *> addrlist(max_frames + 1);
    int const addrlen = backtrace(addrlist.data(), max_frames);
    if (addrlen == 0) {
        fprintf(out, "  <empty, possibly corrupt>\n");
        return;
    }

    char **symbollist = backtrace_symbols(addrlist.data(), addrlen);
    if (symbollist == nullptr) {
        return;
    }

    fprintf(out, "  [oneCCL ERROR Stacktrace]:\n");
    for (int i = 2; i < addrlen; ++i) {
        fprintf(out, "    %s\n", symbollist[i]);
    }
    fprintf(out, "  [oneCCL ERROR Stacktrace End]\n");
    free(static_cast<void *>(symbollist));
}
} // namespace

void onecclDebugLog(onecclDebugLogLevel_t level, const char *filefunc, int line,
                    const char *fmt, ...) {
    if (debug_level < 0) {
        onecclDebugInit();
    }

    if (debug_no_warn != 0 && level == ONECCL_LOG_WARN) {
        level = ONECCL_LOG_INFO;
    }

    if (level < debug_level || debug_level == ONECCL_LOG_DISABLE) {
        return;
    }

    char message[1024];
    va_list vargs;
    va_start(vargs, fmt);
    vsnprintf(message, sizeof(message), fmt, vargs);
    va_end(vargs);

    if (level == ONECCL_LOG_WARN || level == ONECCL_LOG_ERROR) {
        std::scoped_lock const lock(debug_lock);
        last_error = message;
    }

    if (debug_file == nullptr) {
        const char *log_file_path = getenv("CCL_LOG_FILE");
        if ((log_file_path != nullptr) && strlen(log_file_path) > 0) {
            debug_file = fopen(log_file_path, "w");
            if (debug_file == nullptr) {
                fprintf(stderr,
                        "[oneCCL] Failed to open log file '%s', using stderr\n",
                        log_file_path);
                debug_file = stderr;
            }
        } else {
            debug_file = stderr;
        }
    }

    char timestr[64];
    get_timestamp(timestr, sizeof(timestr));

    pid_t const pid = getpid();

    const char *level_str = "UNKNOWN";
    switch (level) {
    case ONECCL_LOG_TRACE:
        level_str = "TRACE";
        break;
    case ONECCL_LOG_INFO:
        level_str = "INFO";
        break;
    case ONECCL_LOG_WARN:
        level_str = "WARN";
        break;
    case ONECCL_LOG_ERROR:
        level_str = "ERROR";
        break;
    default:
        break;
    }

    fprintf(debug_file, "%s:(%d) |%s| [%s:%d] %s\n", timestr,
            static_cast<int>(pid), level_str, filefunc, line, message);

    if (level == ONECCL_LOG_ERROR) {
        print_stack_trace(debug_file);
    }

    fflush(debug_file);
}

const char *onecclGetLastErrorString() {
    std::scoped_lock const lock(debug_lock);

    strncpy(last_error_buffer, last_error.c_str(), kErrorBufferLength);
    last_error_buffer[kErrorBufferLength - 1] = 0;
    return last_error_buffer;
}

void onecclDebugInit() {
    if (debug_level < 0) {
        debug_level = get_log_level_from_env();
    }
}

void onecclResetDebugInit() { debug_level = -1; }
