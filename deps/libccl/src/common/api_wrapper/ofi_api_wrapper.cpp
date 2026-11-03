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

#include <dlfcn.h>
#include <libgen.h>
#include <sys/stat.h>

#include "common/api_wrapper/api_wrapper.hpp"
#include "common/log/log.hpp"
#include "common/api_wrapper/ofi_api_wrapper.hpp"

namespace ccl {

lib_info_t ofi_lib_info;
ofi_lib_ops_t ofi_lib_ops;

static std::string get_relative_ccl_root_path();

std::string get_ofi_lib_path() {
    // lib_path specifies the name and full path to the OFI library.
    // Search order:
    // 1. CCL_OFI_LIBRARY_PATH (always)
    // 2. I_MPI_OFI_LIBRARY (only when transport==mpi)
    // 3. $I_MPI_ROOT/opt/mpi/libfabric/lib/libfabric.so.1 (only when transport==mpi)
    // 4. CCL_ROOT-relative bundled libfabric (dladdr-derived CCL_ROOT):
    //      /opt/mpi/libfabric/lib/libfabric.so.1 (where we install it)
    //      /lib/libfabric/libfabric.so            (IMPI <= 2021.14)
    //      /lib/libfabric.so                      (IMPI 2021.15)
    // 5. "libfabric.so.1" (LD_LIBRARY_PATH / ldconfig)

    auto ofi_lib_path = ccl::global_data::env().ofi_lib_path;
    if (!ofi_lib_path.empty()) {
        LOG_DEBUG("OFI lib path (CCL_OFI_LIBRARY_PATH): ", ofi_lib_path);
        return ofi_lib_path;
    }

    // When MPI transport is requested, check Intel MPI environment variables first
    // to ensure oneCCL and atl_mpi agree on the libfabric version (they share a process).
    if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        char* mpi_ofi_path = getenv("I_MPI_OFI_LIBRARY");
        if (mpi_ofi_path && mpi_ofi_path[0] != '\0') {
            ofi_lib_path = std::string(mpi_ofi_path);
            LOG_DEBUG("OFI lib path (I_MPI_OFI_LIBRARY): ", ofi_lib_path);
            return ofi_lib_path;
        }

        char* mpi_root = getenv("I_MPI_ROOT");
        if (mpi_root) {
            std::string mpi_root_ofi_lib_path = std::string(mpi_root) + "/opt/mpi/libfabric/lib/libfabric.so.1";
            struct stat buffer {};
            if (stat(mpi_root_ofi_lib_path.c_str(), &buffer) == 0) {
                LOG_DEBUG("OFI lib path (I_MPI_ROOT/opt/mpi/libfabric/lib/): ", mpi_root_ofi_lib_path);
                return mpi_root_ofi_lib_path;
            }
        }
    }

    // Prefer the libfabric bundled with the package. The relative paths cover the
    // different install layouts (default install is opt/mpi/libfabric/lib).
    auto ccl_root = get_relative_ccl_root_path();
    if (!ccl_root.empty()) {
        const char* relative_paths[] = {
            "/opt/mpi/libfabric/lib/libfabric.so.1", // Our actual install path
            "/lib/libfabric/libfabric.so",            // IMPI <= 2021.14
            "/lib/libfabric.so",                      // IMPI 2021.15
        };
        struct stat buffer {};
        for (const char* rel : relative_paths) {
            std::string candidate = ccl_root + rel;
            if (stat(candidate.c_str(), &buffer) == 0) {
                LOG_DEBUG("OFI lib path (bundled): ", candidate);
                return candidate;
            }
        }
    }

    // Fall back to the system libfabric resolved through LD_LIBRARY_PATH/ldconfig.
    ofi_lib_path = "libfabric.so.1";
    LOG_DEBUG("OFI lib path (LD_LIBRARY_PATH): ", ofi_lib_path);
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

    // If the bundled/explicit path failed, retry the system libfabric as a last resort.
    if (ofi_lib_info.path != "libfabric.so.1") {
        print_error(error, ofi_lib_info);
        LOG_INFO("Retrying to load system libfabric.so.1");
        ofi_lib_info.path = "libfabric.so.1";
        error = load_library(ofi_lib_info);
        if (error == CCL_LOAD_LB_SUCCESS) {
            return true;
        }
    }

    print_error(error, ofi_lib_info);
    return false;
}

static void setup_providers() {
    // An exported-but-empty FI_PROVIDER_PATH is treated as unset: it carries no
    // search path, and CI environments propagate empty values for unset vars.
    const char* fi_provider_path = getenv("FI_PROVIDER_PATH");
    if (fi_provider_path != nullptr && *fi_provider_path != '\0') {
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

    // libfabric honors only the first entry of FI_PROVIDER_PATH, so this must be
    // the directory holding the provider plugins - not a colon-joined list, and
    // not the directory of the core library itself.
    std::string provider_path = std::string(real_libfabric_path) + "/prov";

    // Leave FI_PROVIDER_PATH unset when there is no bundled provider directory:
    // setting it to a non-existent path would override libfabric's own default
    // search and hide providers built into the library we loaded.
    struct stat buffer {};
    if (stat(provider_path.c_str(), &buffer) != 0) {
        LOG_DEBUG("no provider directory at ", provider_path, ", leaving FI_PROVIDER_PATH unset");
        return;
    }

    if (setenv("FI_PROVIDER_PATH", provider_path.c_str(), 1) != 0) {
        LOG_ERROR("Failed to set FI_PROVIDER_PATH with error: ", strerror(errno));
        return;
    }

    LOG_DEBUG("FI_PROVIDER_PATH set to: ", provider_path);
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
