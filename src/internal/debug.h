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

#ifndef ONECCL_DEBUG_H
#define ONECCL_DEBUG_H

#ifdef __cplusplus
extern "C" {
#endif

// Log level enumeration
typedef enum {
    ONECCL_LOG_TRACE = 0,
    ONECCL_LOG_INFO = 1,
    ONECCL_LOG_WARN = 2,
    ONECCL_LOG_ERROR = 3,
    ONECCL_LOG_DISABLE = 9999 // for turning off logs
} onecclDebugLogLevel_t;

// Logging function (similar to ncclDebugLog)
void onecclDebugLog(onecclDebugLogLevel_t level, const char *filefunc, int line,
                    const char *fmt, ...) __attribute__((format(printf, 4, 5)));

// Simple macro wrappers for logging at various levels
#define ONECCL_TRACE(...)                                                      \
    onecclDebugLog(ONECCL_LOG_TRACE, __func__, __LINE__, __VA_ARGS__)
#define ONECCL_INFO(...)                                                       \
    onecclDebugLog(ONECCL_LOG_INFO, __func__, __LINE__, __VA_ARGS__)
#define ONECCL_WARN(...)                                                       \
    onecclDebugLog(ONECCL_LOG_WARN, __func__, __LINE__, __VA_ARGS__)
#define ONECCL_ERROR(...)                                                      \
    onecclDebugLog(ONECCL_LOG_ERROR, __func__, __LINE__, __VA_ARGS__)

// Retrieve the last error string
const char *onecclGetLastErrorString();

#ifdef __cplusplus
}
#endif

#endif /* ONECCL_DEBUG_H */
