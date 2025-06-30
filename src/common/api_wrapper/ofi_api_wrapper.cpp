/*
 Copyright 2016-2020 Intel Corporation
 
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
#include <sys/stat.h>

#include "common/api_wrapper/api_wrapper.hpp"
#include "common/api_wrapper/ofi_api_wrapper.hpp"

namespace ccl {

lib_info_t ofi_lib_info;
ofi_lib_ops_t ofi_lib_ops;

std::string get_ofi_lib_path() {
    // lib_path specifies the name and full path to the OFI library -
    // it should be an absolute and validated path pointing to the
    // desired libfabric library

    // the order of searching for libfabric is:
    // * CCL_OFI_LIBRARY_PATH (ofi_lib_path env)
    // * I_MPI_OFI_LIBRARY
    // * I_MPI_ROOT/opt/mpi/libfabric/lib
    // * LD_LIBRARY_PATH

    auto ofi_lib_path = ccl::global_data::env().ofi_lib_path;
    if (!ofi_lib_path.empty()) {
        LOG_DEBUG("OFI lib path (CCL_OFI_LIBRARY_PATH): ", ofi_lib_path);
    }
    else {
        char* mpi_ofi_path = getenv("I_MPI_OFI_LIBRARY");
        if (mpi_ofi_path) {
            ofi_lib_path = std::string(mpi_ofi_path);
            LOG_DEBUG("OFI lib path (I_MPI_OFI_LIBRARY): ", ofi_lib_path);
        }
        else {
            char* mpi_root = getenv("I_MPI_ROOT");
            std::string mpi_root_ofi_lib_path =
                mpi_root == NULL ? std::string() : std::string(mpi_root);
            mpi_root_ofi_lib_path += "/opt/mpi/libfabric/lib/libfabric.so";
            struct stat buffer {};
            if (mpi_root && stat(mpi_root_ofi_lib_path.c_str(), &buffer) == 0) {
                ofi_lib_path = std::move(mpi_root_ofi_lib_path);
                LOG_DEBUG("OFI lib path (MPI_ROOT/opt/mpi/libfabric/lib/): ", ofi_lib_path);
            }
            else {
                ofi_lib_path = "libfabric.so";
                LOG_DEBUG("OFI lib path (LD_LIBRARY_PATH): ", ofi_lib_path);
            }
        }
    }

    return ofi_lib_path;
}

bool ofi_api_init() {
    bool ret = true;

    ofi_lib_info.ops = &ofi_lib_ops;
    ofi_lib_info.fn_names = ofi_fn_names;
    ofi_lib_info.path = get_ofi_lib_path();

    int error = load_library(ofi_lib_info);
    if (error != CCL_LOAD_LB_SUCCESS) {
        print_error(error, ofi_lib_info);
        ret = false;
    }

    return ret;
}

void ofi_api_fini() {
    LOG_DEBUG("close OFI lib: handle: ", ofi_lib_info.handle);
    close_library(ofi_lib_info);
}

} //namespace ccl
