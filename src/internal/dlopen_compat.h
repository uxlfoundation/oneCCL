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

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define RTLD_LAZY 0 // No Windows equivalent
#define RTLD_NOW 1  // No Windows equivalent

typedef HMODULE dl_handle;

// Define the UNIX-like dlopen function using LoadLibrary
inline dl_handle dlopen(const char *file, int mode) {
    return LoadLibrary(file);
}

// Define the UNIX-like dlsym function using GetProcAddress
inline void *dlsym(dl_handle handle, const char *symbol) {
    return reinterpret_cast<void *>(GetProcAddress(handle, symbol));
}

// Define the UNIX-like dlerror function using GetLastError and FormatMessage
inline const char *dlerror() {
    static char buffer[256];
    DWORD error = GetLastError();
    if (error != 0U) {
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM |
                           FORMAT_MESSAGE_IGNORE_INSERTS,
                       nullptr, error, 0, buffer, sizeof(buffer), nullptr);
        SetLastError(0);
        return buffer;
    }
    return nullptr;
}

// Define the UNIX-like dlclose function using FreeLibrary
inline int dlclose(dl_handle handle) { return FreeLibrary(handle) ? 0 : 1; }

#else

typedef void *dl_handle;

// On non-Windows, use the system's dl functions directly
#define dlopen(file, mode) dlopen(file, mode)
#define dlsym(handle, symbol) dlsym(handle, symbol)
#define dlerror() dlerror()
#define dlclose(handle) dlclose(handle)

#endif

#ifdef __cplusplus
}
#endif
