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

#include "common/api_wrapper/api_wrapper.hpp"
#include "common/api_wrapper/ze_api_wrapper.hpp"
#include "common/stream/stream.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

namespace ccl {

ccl::lib_info_t ze_lib_info;
ze_lib_ops_t ze_lib_ops;

bool ze_api_init() {
    bool ret = true;

    ze_lib_info.ops = &ze_lib_ops;

    // lib_path specifies the name and full path to the level-zero library
    // it should be absolute and validated path
    // pointing to desired libze_loader library
    ze_lib_info.path = ccl::global_data::env().ze_lib_path;

    if (ze_lib_info.path.empty()) {
        ze_lib_info.path = "libze_loader.so";
    }
    LOG_DEBUG("level-zero lib path: ", ze_lib_info.path);

    ze_lib_info.fn_names = ze_fn_names;
    ze_lib_info.additional_fn_names = &ze_fn_names_optional;
    int error = load_library(ze_lib_info);
    if (error != CCL_LOAD_LB_SUCCESS) {
        print_error(error, ze_lib_info);
        ret = false;
    }

    return ret;
}

void ze_api_fini() {
    LOG_DEBUG("close level-zero lib: handle: ", ze_lib_info.handle);
    close_library(ze_lib_info);
}

} //namespace ccl
