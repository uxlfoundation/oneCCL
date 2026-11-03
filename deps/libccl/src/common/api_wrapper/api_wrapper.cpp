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
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/api_wrapper/ze_api_wrapper.hpp"
#if defined(CCL_ENABLE_UMF)
#include "common/api_wrapper/umf_api_wrapper.hpp"
#endif // CCL_ENABLE_UMF
#endif //CCL_ENABLE_SYCL && CCL_ENABLE_ZE
#if defined(CCL_ENABLE_MPI)
#include "common/api_wrapper/mpi_api_wrapper.hpp"
#endif //CCL_ENABLE_MPI
#include "common/api_wrapper/ofi_api_wrapper.hpp"
#include "common/api_wrapper/openmp_wrapper.hpp"

#include <dlfcn.h>
#include <link.h>

namespace ccl {

static bool g_mpi_was_inited = false;

void api_wrappers_init() {
    bool mpi_inited = false;
    bool ofi_inited = false;

    if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        // Handle MPI transport
        mpi_inited = mpi_api_init();
        if (!mpi_inited)
            LOG_WARN("MPI transport was not initialized, fallback to OFI transport");
    }

    // Either if-switch fallback or MPI initialization failure's fallback
    if (!mpi_inited) {
        ccl::global_data::env().atl_transport = ccl_atl_ofi;

        // Handle OFI transport
        ofi_inited = ofi_api_init();
        if (!ofi_inited) {
            LOG_INFO("could not initialize OFI api");
        }
    }
    g_mpi_was_inited = mpi_inited;

    CCL_THROW_IF_NOT((mpi_inited || ofi_inited), "could not initialize the requested transport library");

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (ccl::global_data::env().backend == backend_mode::native &&
        ccl::global_data::env().ze_enable) {
        LOG_INFO("initializing level-zero api");
        if (ze_api_init()) {
            try {
                ccl::global_data::get().ze_data.reset(new ze::global_data_desc);
            }
            catch (const ccl::exception& e) {
                LOG_INFO("could not initialize level-zero: ", e.what());
            }
            catch (...) {
                LOG_INFO("could not initialize level-zero: unknown error");
            }
        }
    }
    else {
        LOG_INFO("could not initialize level-zero api");
    }
#if defined(CCL_ENABLE_UMF)
    if (ccl::global_data::env().backend == backend_mode::native &&
        ccl::global_data::env().umf_enable) {
        LOG_INFO("initializing umf api");
        if (!umf_api_init()) {
            CCL_THROW("could not initialize umf api");
        }
    }
    else {
        LOG_INFO("could not initialize umf api");
    }
#endif // CCL_ENABLE_UMF

#endif //CCL_ENABLE_SYCL && CCL_ENABLE_ZE
#if defined(CCL_ENABLE_OMP)
    if (ccl::global_data::env().atl_transport == ccl_atl_mpi && openmp_api_init()) {
        ccl::openmp_lib_ops.init(ccl::global_data::env().omp_allreduce_num_threads);
    }
    else {
        if (ccl::global_data::env().atl_transport != ccl_atl_mpi) {
            LOG_WARN("OpenMP collectives require the MPI transport, disabling");
        }
        ccl::openmp_lib_ops.allreduce = nullptr;
        ccl::openmp_lib_ops.thread_num = nullptr;
        ccl::openmp_lib_ops.init = nullptr;
    }
#endif // CCL_ENABLE_OMP
}

void api_wrappers_fini() {
    ofi_api_fini();
    if (g_mpi_was_inited) {
        mpi_api_fini();
    }
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    ze_api_fini();
#if defined(CCL_ENABLE_UMF)
    umf_api_fini();
#endif // CCL_ENABLE_UMF
#endif //CCL_ENABLE_SYCL && CCL_ENABLE_ZE
#if defined(CCL_ENABLE_OMP)
    openmp_api_fini();
#endif // CCL_ENABLE_OMP
}

int load_library(lib_info_t& info) {
    // Check if the path to the library passed in info is correct
    // - if it contains invalid characters log an error and leave the handle empty.
    bool is_invalid_path =
        ((info.path.find("..") != std::string::npos) ||
         (info.path.find("./") != std::string::npos) || (info.path.find("%") != std::string::npos));
    if (is_invalid_path) {
        return CCL_LOAD_LB_PATH_ERROR;
    }

    info.handle = dlopen(info.path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!info.handle) {
        return CCL_LOAD_LB_DLOPEN_ERROR;
    }
    link_map* lmap = nullptr;
    if (dlinfo(info.handle, RTLD_DI_LINKMAP, &lmap) != 0) {
        LOG_WARN("dlinfo failed for library: ", info.path);
    }
    else if (lmap && lmap->l_name && lmap->l_name[0] != '\0')
    {
        LOG_INFO("Loaded library. path: ", lmap->l_name);
    }
    else
    {
        LOG_WARN("Library path is empty for: ", info.path);
    }

    auto ops = reinterpret_cast<void**>((void*)info.ops);
    auto fn_names = info.fn_names;
    auto additional_fn_names = info.additional_fn_names;
    size_t total_size = fn_names.size();
    if (additional_fn_names) {
        total_size += additional_fn_names->size();
        fn_names.insert(fn_names.end(), additional_fn_names->begin(), additional_fn_names->end());
    }
    for (size_t i = 0; i < total_size; ++i) {
        if (info.direct) {
            ops[i] = dlsym(info.handle, fn_names[i].c_str());
        }
        else {
            auto indirect_symbol = reinterpret_cast<ccl::indirect_symbol_fn_t>(
                dlsym(info.handle, fn_names[i].c_str()));
            if (!indirect_symbol) {
                // Log the dlsym error using dlerror before calling the function
                LOG_ERROR("dlsym failed for symbol: ", fn_names[i], ", error: ", dlerror());
                return CCL_LOAD_LB_DLOPEN_ERROR;
            }
            ops[i] = indirect_symbol();
        }
        if (i < info.fn_names.size()) {
            CCL_THROW_IF_NOT(ops[i], "dlsym is failed on: ", fn_names[i], ", error: ", dlerror());
        }
        else if (!ops[i]) {
            LOG_DEBUG(
                "dlsym cannot load the optional symbol: ", fn_names[i], " , reason: ", dlerror());
        }
        else {
            LOG_TRACE("dlsym loaded of ", fn_names.size(), " - ", i + 1, ": ", fn_names[i]);
        }
    }
    info.fn_names = fn_names;
    return CCL_LOAD_LB_SUCCESS;
}

void print_error(int error, lib_info_t& info) {
    if (error == CCL_LOAD_LB_PATH_ERROR) {
        LOG_WARN("library path is not valid: ",
                 info.path.c_str(),
                 " - path contains invalid characters");
    }
    else if (error == CCL_LOAD_LB_DLOPEN_ERROR) {
        // Log as `info`, because in some cases we have to test multiple paths
        // and we do not want to write excess information into users screen
        LOG_INFO("could not open the library: ", info.path.c_str(), " - ", dlerror());
    }
}

void close_library(lib_info_t& info) {
    if (info.handle) {
        dlclose(info.handle);
        info.handle = nullptr;
    }
}

} //namespace ccl
