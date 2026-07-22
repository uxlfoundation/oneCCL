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
#include "common/api_wrapper/openmp_wrapper.hpp"

#if defined(CCL_ENABLE_MPI)

#include <dlfcn.h>

namespace ccl {

lib_info_t openmp_lib_info;
openmp_lib_ops_t openmp_lib_ops;

bool openmp_api_init() {
    bool ret = true;

    openmp_lib_info.ops = &openmp_lib_ops;
    openmp_lib_info.fn_names = openmp_fn_names;
    openmp_lib_info.path = "libccl_openmp.so";
    openmp_lib_info.direct = false;

    int error = load_library(openmp_lib_info);
    if (error != CCL_LOAD_LB_SUCCESS) {
        if (error == CCL_LOAD_LB_PATH_ERROR) {
            LOG_WARN("library path is not valid: ",
                     openmp_lib_info.path.c_str(),
                     " - path contains invalid characters");
        }
        else if (error == CCL_LOAD_LB_DLOPEN_ERROR) {
            LOG_INFO(
                "could not open the library: ", openmp_lib_info.path.c_str(), " - ", dlerror());
        }
        ret = false;
    }

    return ret;
}

void openmp_api_fini() {}

} // namespace ccl

#endif //CCL_ENABLE_MPI
