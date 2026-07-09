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

#include "oneapi/ccl/config.h"

#include "common/global/global.hpp"
#include "common/log/log.hpp"

#define CCL_LOAD_LB_SUCCESS      0
#define CCL_LOAD_LB_PATH_ERROR   1
#define CCL_LOAD_LB_DLOPEN_ERROR 2

namespace ccl {

typedef void* (*indirect_symbol_fn_t)();

typedef struct lib_info {
    std::string path;
    void* handle;
    void* ops;
    std::vector<std::string> fn_names;
    // All varialbes above are initialized in XX_api_init()
    // Libraries in direct mode load symbols by their names,
    // while libraries with direct == false (indirect),
    // user get functions that return pointer to the function
    // inside loaded library (indirect_symbol_fn_t)
    std::vector<std::string> *additional_fn_names = nullptr;
    bool direct = true;
} lib_info_t;

void api_wrappers_init();
void api_wrappers_fini();

int load_library(lib_info_t& info);
void print_error(int error, lib_info_t& info);
void close_library(lib_info_t& info);

} //namespace ccl
