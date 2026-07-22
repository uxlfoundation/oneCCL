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
#include "common/api_wrapper/umf_api_wrapper.hpp"

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_UMF)
namespace ccl {

ccl::lib_info_t umf_lib_info;
umf_lib_ops_t umf_lib_ops;

bool umf_api_init() {
    bool ret = true;

    umf_lib_info.ops = &umf_lib_ops;
    umf_lib_info.fn_names = umf_fn_names;

    // lib_path specifies the name and full path to the level-umfro library
    // it should be absolute and validated path
    // pointing to desired UMF library
    umf_lib_info.path = ccl::global_data::env().umf_lib_path;

    if (umf_lib_info.path.empty()) {
        // Try multiple library versions in order of preference
        // This handles different UMF versions that may be installed
        std::vector<std::string> lib_names = { "libumf.so.1", "libumf.so.0", "libumf.so" };
        int error = CCL_LOAD_LB_DLOPEN_ERROR;
        for (const auto& lib_name : lib_names) {
            umf_lib_info.path = lib_name;
            LOG_DEBUG("trying UMF lib path: ", umf_lib_info.path);
            error = load_library(umf_lib_info);
            if (error == CCL_LOAD_LB_SUCCESS) {
                LOG_DEBUG("successfully loaded UMF lib: ", umf_lib_info.path);
                return true;
            }
        }
        // All attempts failed, print error for the last tried library
        print_error(error, umf_lib_info);
        ret = false;
    }
    else {
        LOG_DEBUG("UMF lib path: ", umf_lib_info.path);

        int error = load_library(umf_lib_info);
        if (error != CCL_LOAD_LB_SUCCESS) {
            print_error(error, umf_lib_info);
            ret = false;
        }
    }

    return ret;
}

void umf_api_fini() {
    LOG_DEBUG("close UMF lib: handle: ", umf_lib_info.handle);
    close_library(umf_lib_info);
}

} //namespace ccl
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE && CCL_ENABLE_UMF
