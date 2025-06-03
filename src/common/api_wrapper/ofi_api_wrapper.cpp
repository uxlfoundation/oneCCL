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
#include <dlfcn.h>
#include <libgen.h>
#include <sys/stat.h>

#include "common/api_wrapper/api_wrapper.hpp"
#include "common/log/log.hpp"
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

static std::string get_relative_ccl_root_path() {
    Dl_info info;

    if (dladdr((void*)ccl::get_library_version, &info)) {
        char libccl_path[PATH_MAX];

        if (realpath(info.dli_fname, libccl_path) != nullptr) {
            // We have to use `realpath`, so the dirname will work correctly,
            // because if there's any symlink like `..` in the path it will not work
            libccl_path[PATH_MAX - 1] = '\0';

            // Remove `libccl.so` from the path to get directory like `$CCL_ROOT/lib`
            char* libccl_dir = dirname(libccl_path);
            // Remove the `lib` from the path the get just the `CCL_ROOT`
            char* ccl_root_cstr = dirname(libccl_dir);

            auto ccl_root = std::string(ccl_root_cstr);

            return ccl_root;
        }
    }

    return {};
}

static bool load_libfabric() {
    ofi_lib_info.ops = &ofi_lib_ops;
    ofi_lib_info.fn_names = ofi_fn_names;
    ofi_lib_info.path = get_ofi_lib_path();

    int error = load_library(ofi_lib_info);
    if (error == CCL_LOAD_LB_SUCCESS) {
        return true;
    }

    print_error(error, ofi_lib_info);
    LOG_INFO("Retrying to load libfabric.so using relative path");

    auto realtive_root = get_relative_ccl_root_path();
    if (realtive_root.empty()) {
        return false;
    }

    // Path up to IMPI 2021.14
    ofi_lib_info.path = realtive_root + "/lib/libfabric/libfabric.so";
    error = load_library(ofi_lib_info);
    if (error == CCL_LOAD_LB_SUCCESS) {
        return true;
    }

    // Path in IMPI 2021.15
    ofi_lib_info.path = realtive_root + "/lib/libfabric.so";
    error = load_library(ofi_lib_info);
    if (error == CCL_LOAD_LB_SUCCESS) {
        return true;
    }

    print_error(error, ofi_lib_info);
    return false;
}

static void setup_providers() {
    const char* fi_provider_path = getenv("FI_PROVIDER_PATH");
    if (fi_provider_path != nullptr) {
        LOG_DEBUG("FI_PROVIDER_PATH is already set to: ", fi_provider_path);
        return;
    }

    char libfabric_path[PATH_MAX];
    dlinfo(ofi_lib_info.handle, RTLD_DI_ORIGIN, &libfabric_path);

    // Add realpath to resolve any symlinks and get the absolute path
    char real_libfabric_path[PATH_MAX];
    if (!realpath(libfabric_path, real_libfabric_path)) {
        LOG_ERROR("Failed to resolve libfabric realpath: ", strerror(errno));
        return;
    }

    std::string primary_path = std::string(real_libfabric_path);
    std::string secondary_path = primary_path + "/prov";

    // Construct the full provider path with colon separator
    std::string full_provider_path = primary_path + ":" + secondary_path;

    if (setenv("FI_PROVIDER_PATH", full_provider_path.c_str(), 1) != 0) {
        LOG_ERROR("Failed to set FI_PROVIDER_PATH with error: ", strerror(errno));
        return;
    }

    LOG_DEBUG("FI_PROVIDER_PATH set to: ", full_provider_path);
}

bool ofi_api_init() {
    if (load_libfabric() == false) {
        return false;
    }

    setup_providers();
    return true;
}

void ofi_api_fini() {
    LOG_DEBUG("close OFI lib: handle: ", ofi_lib_info.handle);
    close_library(ofi_lib_info);
}

} //namespace ccl
