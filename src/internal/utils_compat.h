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

#include <cstdlib>
#include <cstring>
#include <ctime>

#ifdef _WIN32
#include <crtdbg.h>
#include <windows.h>

inline char *utils_getenv(const char *name) {
    if (name == nullptr) {
        return nullptr;
    }
    size_t required_size = 0;
    getenv_s(&required_size, nullptr, 0, name);
    if (required_size == 0) {
        return nullptr;
    }
    char *buffer = new char[required_size];
    getenv_s(&required_size, buffer, required_size, name);
    return buffer;
}

inline bool utils_localtime(const time_t *timep, struct tm *result) {
    errno_t err = localtime_s(result, timep);
    return err == 0; // Returns 0 on success, -1 on failure
}

inline char *utils_strdup(const char *str) {
    if (str == nullptr) {
        return nullptr;
    }
    return _strdup(str);
}

#else
#include <unistd.h> // For Linux systems

inline char *utils_getenv(const char *name) {
    if (!name) {
        return nullptr;
    }
    return getenv(name); // Use secure_getenv if available on your platform
}

inline int utils_localtime(const time_t *timep, struct tm *result) {
    struct tm *tm_result = localtime_r(timep, result);
    return tm_result != nullptr ? 0 : -1; // Returns 0 on success, -1 on failure
}

inline char *utils_strdup(const char *str) {
    if (str == nullptr) {
        return nullptr;
    }
    return strdup(str);
}

#endif
