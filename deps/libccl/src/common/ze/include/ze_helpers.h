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

// #define USE_FAST_LOGGER

#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include "ze_errors.h"
#include "occp/inc/log.h"
#include "common/api_wrapper/ze_api_wrapper.hpp"
#include "common/log/log.hpp"

#ifndef FOR_I
#define FOR_I(N) for(decltype(N) i = 0; i < (N); i++)
#endif

#ifdef USE_FAST_LOGGER
#define FLOG_ENABLE_LAZY_LOGGING
#include "fast_logger.hpp"
namespace onceccl{
enum class LoggerTypes
{
    CCL,
    CRASH,
    DFA,
    CCL_TEST,
    LOG_MAX // must be the last value
};
unsigned getPostLogValue();
}

struct MaxRotationFileInfo {
    std::string filename;
    uint32_t    maxFiles;
};
MaxRotationFileInfo getRotationFilesInfo();
void resetRotationFilesInfo();

#define FLOG_ENUM_TYPE_NAME onceccl::LoggerTypes
#define getLoggingLevel fast_logger::getLoggingLevel
FLOG_DECLARE_MODULE_LOGGER()


#endif

#define L0_LOG(PREFIX, PRINT, ADD_THREAD, ADD_TIME, ADD_FUNC, OUT, ...)      \
    do {                                                                       \
        if (OUT) {                                                             \
            std::ostringstream oss;                                            \
            if (ADD_THREAD) {                                                  \
                oss << thread_id_str() << ":";                                 \
            }                                                                  \
            if (ADD_TIME) {                                                    \
                oss << current_time_str() << " ";                              \
            }                                                                  \
            oss << PREFIX;                                                     \
            if (ADD_FUNC) {                                                    \
                oss << short_func(__PRETTY_FUNCTION__) << " ";                 \
            }                                                                  \
            ze_trace_params(oss, ##__VA_ARGS__);                                  \
            if (PRINT) {                                                       \
                std::cout << oss.str() << '\n';                                \
            }                                                                  \
        }                                                                      \
    } while (0)



inline void ze_trace_params(std::ostringstream &) {
    // Base case for recursion, do nothing
}

template <typename T, typename... Args>
inline void ze_trace_params(std::ostringstream &oss, const T &first,
                         const Args &...rest) {
    oss << first;
    ze_trace_params(oss, rest...);
};

#ifndef USE_FAST_LOGGER

    inline void outfunc(unsigned logger_type, unsigned logger_rank, const std::string& ostr) {
            //do nothing
    }

#define L0_PRINT(...) LOG_TRACE(__VA_ARGS__)

#define L0_TRACE(...) LOG_TRACE(__VA_ARGS__)
#define L0_ERROR(...) LOG_ERROR(__VA_ARGS__)
#define L0_WARN(...)  LOG_WARN(__VA_ARGS__)
#define L0_INFO(...)  LOG_INFO(__VA_ARGS__)
#define L0_DEBUG(...) LOG_DEBUG(__VA_ARGS__)
#define L0_CRITICAL(...) CCL_FATAL(__VA_ARGS__)

#else

#define L0_PRINT_X(FLOG_MAC,...) do {                                                 \
                            std::ostringstream ostrs;                      \
                            ostrs << short_func(__PRETTY_FUNCTION__) << " ";         \
                            ze_trace_params(ostrs, ##__VA_ARGS__);            \
                            FLOG_MAC(CCL, "{}",ostrs.str());          \
                     } while (0);

#define L0_PRINT(...) L0_PRINT_X(FLOG_TRACE, ##__VA_ARGS__)

#define L0_TRACE(...) L0_PRINT_X(FLOG_TRACE, ##__VA_ARGS__)
#define L0_ERROR(...) L0_PRINT_X(FLOG_ERR, ##__VA_ARGS__)
#define L0_WARN(...)  L0_PRINT_X(FLOG_WARN, ##__VA_ARGS__)
#define L0_INFO(...)  L0_PRINT_X(FLOG_INFO, ##__VA_ARGS__)
#define L0_DEBUG(...) L0_PRINT_X(FLOG_DEBUG, ##__VA_ARGS__)
#define L0_CRITICAL(...) L0_PRINT_X(FLOG_CRITICAL, ##__VA_ARGS__)


#endif

#define ZE_ERROR(func, res)                                                    \
    L0_ERROR("Error(0x", std::hex, res, std::dec, ") in ", #func, ": ",        \
             onecclZeResultToString(res))

// return ze error code
#define ZE_RET(func)                                                          \
    do {                                                                       \
        ze_result_t r = func;                                                \
        if (r != ZE_RESULT_SUCCESS) {                                        \
            ZE_ERROR(func, r);                                               \
            return r;                                                        \
        }                                                                      \
    } while (0)

// no return value
#define ZE_NO_RET(func)                                                       \
    do {                                                                       \
        ze_result_t r = func;                                                \
        if (r != ZE_RESULT_SUCCESS) {                                        \
            ZE_ERROR(func, r);                                               \
            return;                                                            \
        }                                                                      \
    } while (0)

// return nullptr if error
#define ZE_RET_NULL(func)                                                      \
    do {                                                                       \
        const ze_result_t r = func;                                                \
        if (r != ZE_RESULT_SUCCESS) {                                        \
            ZE_ERROR(func, r);                                               \
            return nullptr;                                                   \
        }                                                                      \
    } while (0)
