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

#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cstdlib>

enum class OcclLogLevel { OFF = 0, CRITICAL, ERROR, WARN, INFO, DEBUG, TRACE };

static inline OcclLogLevel occl_get_log_level() {
    const char* lvl = std::getenv("CCL_OCCP_LOG_LEVEL");
    if (!lvl)
        return OcclLogLevel::OFF; // Default
    std::string l(lvl);
    for (auto& c : l)
        c = static_cast<char>(toupper(c)); // Case-insensitive

    if (l == "OFF")
        return OcclLogLevel::OFF;
    if (l == "CRITICAL")
        return OcclLogLevel::CRITICAL;
    if (l == "ERROR")
        return OcclLogLevel::ERROR;
    if (l == "WARN")
        return OcclLogLevel::WARN;
    if (l == "INFO")
        return OcclLogLevel::INFO;
    if (l == "DEBUG")
        return OcclLogLevel::DEBUG;
    if (l == "TRACE")
        return OcclLogLevel::TRACE;
    return OcclLogLevel::OFF;
}

static inline bool occl_should_log(OcclLogLevel msg_level) {
    static OcclLogLevel env_level = occl_get_log_level();
    return msg_level <= env_level;
}

static inline std::string thread_id_str() {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    std::string id = oss.str();
    // Return last 4 characters for brevity
    return id.length() > 4 ? id.substr(id.length() - 4) : id;
}

static inline std::string short_func(const char* pretty_func) {
    std::string pf(pretty_func);
    size_t colons = pf.rfind("::");
    if (colons == std::string::npos)
        return pf;
    size_t begin = pf.substr(0, colons).find_last_of(" ") + 1;
    size_t end = pf.find('(', colons);
    if (begin == std::string::npos || end == std::string::npos)
        return pf;
    return pf.substr(begin, end - begin);
}

static inline std::string current_time_str() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time_t = std::chrono::high_resolution_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

inline void trace_params(std::ostringstream&) {
    // Base case for recursion, do nothing
}

template <typename T, typename... Args>
inline void trace_params(std::ostringstream& oss, const T& first, const Args&... rest) {
    oss << first << " ";
    trace_params(oss, rest...);
}

#define OCCL_PRINT(PREFIX, ADD_FUNC, OUT, ...) \
    do { \
        if (OUT) { \
            std::ostringstream oss; \
            oss << thread_id_str() << " : " << PREFIX << " "; \
            oss << current_time_str() << " "; \
            if (ADD_FUNC) { \
                oss << short_func(__PRETTY_FUNCTION__) << " "; \
            } \
            trace_params(oss, ##__VA_ARGS__); \
            std::cout << oss.str() << std::endl; \
        } \
    } while (0)

#define OCCL_TRACE(...) \
    do { \
        if (occl_should_log(OcclLogLevel::TRACE)) \
            OCCL_PRINT("[TRACE   ]", true, true, ##__VA_ARGS__); \
    } while (0)

#define OCCL_DEBUG(...) \
    do { \
        if (occl_should_log(OcclLogLevel::DEBUG)) \
            OCCL_PRINT("[DEBUG   ]", true, true, ##__VA_ARGS__); \
    } while (0)

#define OCCL_INFO(...) \
    do { \
        if (occl_should_log(OcclLogLevel::INFO)) \
            OCCL_PRINT("[INFO    ]", true, true, ##__VA_ARGS__); \
    } while (0)

#define OCCL_WARN(...) \
    do { \
        if (occl_should_log(OcclLogLevel::WARN)) \
            OCCL_PRINT("[WARN    ]", true, true, ##__VA_ARGS__); \
    } while (0)

#define OCCL_ERROR(...) \
    do { \
        if (occl_should_log(OcclLogLevel::ERROR)) \
            OCCL_PRINT("[ERROR   ]", true, true, ##__VA_ARGS__); \
    } while (0)

#define OCCL_CRITICAL(...) \
    do { \
        if (occl_should_log(OcclLogLevel::CRITICAL)) \
            OCCL_PRINT("[CRITICAL]", true, true, ##__VA_ARGS__); \
    } while (0)

#define OCCL_OUT(...) OCCL_PRINT("", false, true, ##__VA_ARGS__)

#define COORD_LOG(...) OCCL_TRACE(__VA_ARGS__)
#define COORD_DBG(...) OCCL_DEBUG(__VA_ARGS__)
#define COORD_ERR(...) OCCL_ERROR(__VA_ARGS__)
#define COORD_INF(...) OCCL_INFO(__VA_ARGS__)
#define COORD_CRT(...) OCCL_CRITICAL(__VA_ARGS__)
#define COORD_WRN(...) OCCL_WARN(__VA_ARGS__)

#define VERIFY(cond, msg, ...) \
    do { \
        if (!(cond)) { \
            OCCL_CRITICAL(("VERIFY failed: " #cond " - " msg), ##__VA_ARGS__); \
            abort(); \
        } \
    } while (false)

#define RET_ON_ERR(func) \
    if (func == -1) { \
        COORD_ERR(#func " returned with error. ({}) {}", errno, strerror(errno)); \
        return false; \
    }

#define SYS_FUNC_CALL(func) \
    if (func == -1) { \
        COORD_WRN(#func " returned with error. ({}) {}", errno, strerror(errno)); \
    }

#define _DEF_IMPL_ \
    { COORD_LOG("[ default(empty) implementation ]"); }
