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
#include "common/api_wrapper/mpi_api_wrapper.hpp"

#if defined(CCL_ENABLE_MPI)

namespace ccl {

lib_info_t mpi_lib_info;
mpi_lib_ops_t mpi_lib_ops;

bool mpi_api_init() {
    bool ret = true;

    mpi_lib_info.ops = &mpi_lib_ops;
    mpi_lib_info.fn_names = mpi_fn_names;

    // lib_path specifies the name and full path to the MPI library
    // it should be absolute and validated path
    // pointing to desired libmpi library
    mpi_lib_info.path = ccl::global_data::env().mpi_lib_path;

    if (mpi_lib_info.path.empty()) {
        mpi_lib_info.path = "libmpi.so.12";
    }
    LOG_DEBUG("MPI lib path: ", mpi_lib_info.path);

    int error = load_library(mpi_lib_info);
    if (error != CCL_LOAD_LB_SUCCESS) {
        print_error(error, mpi_lib_info);
        ret = false;
    }

    return ret;
}

void mpi_api_fini() {
    LOG_DEBUG("close MPI lib: handle: ", mpi_lib_info.handle);
    close_library(mpi_lib_info);
}

const CCL_API mpi_lib_ops_t &get_mpi_lib_ops() {
    return mpi_lib_ops;
}

} //namespace ccl

#endif //CCL_ENABLE_MPI
