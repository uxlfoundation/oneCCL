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

#include <map>
#include <string>

typedef int (*ccl_horovod_init_function)(const int*, int);
extern ccl_horovod_init_function horovod_init_function;
static constexpr const char* horovod_init_function_name = "horovod_init";

enum ccl_framework_type {
    ccl_framework_none,
    ccl_framework_horovod,

    ccl_framework_last_value
};

extern std::map<ccl_framework_type, std::string> ccl_framework_type_names;
