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

namespace ccl {

void api_wrappers_init() {
    bool ofi_inited = true, mpi_inited = true;
    if (!(ofi_inited = ofi_api_init())) {
        LOG_INFO("could not initialize OFI api");
    }
#if defined(CCL_ENABLE_MPI)
    if (!(mpi_inited = mpi_api_init())) {
        LOG_INFO("could not initialize MPI api");
    }
#endif //CCL_ENABLE_MPI
    CCL_THROW_IF_NOT(ofi_inited || mpi_inited, "could not initialize any transport library");
    if (!ofi_inited && (ccl::global_data::env().atl_transport == ccl_atl_ofi)) {
        ccl::global_data::env().atl_transport = ccl_atl_mpi;
        LOG_WARN("OFI transport was not initialized, fallback to MPI transport");
    }

    if (!mpi_inited && (ccl::global_data::env().atl_transport == ccl_atl_mpi)) {
        ccl::global_data::env().atl_transport = ccl_atl_ofi;
        LOG_WARN("MPI transport was not initialized, fallback to OFI transport");
    }

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
#if defined(CCL_ENABLE_MPI) && defined(CCL_ENABLE_OMP)
    if (!openmp_api_init()) {
        ccl::openmp_lib_ops.allreduce = nullptr;
        ccl::openmp_lib_ops.thread_num = nullptr;
        ccl::openmp_lib_ops.init = nullptr;
    }
    else {
        ccl::openmp_lib_ops.init(ccl::global_data::env().omp_allreduce_num_threads);
    }
#endif // CCL_ENABLE_MPI && CCL_ENABLE_OMP
}

void api_wrappers_fini() {
    ofi_api_fini();
#if defined(CCL_ENABLE_MPI)
    mpi_api_fini();
#endif //CCL_ENABLE_MPI
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    ze_api_fini();
#if defined(CCL_ENABLE_UMF)
    umf_api_fini();
#endif // CCL_ENABLE_UMF
#endif //CCL_ENABLE_SYCL && CCL_ENABLE_ZE
#if defined(CCL_ENABLE_MPI) && defined(CCL_ENABLE_OMP)
    openmp_api_fini();
#endif // CCL_ENABLE_MPI && CCL_ENABLE_OMP
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

    void** ops = (void**)((void*)info.ops);
    auto fn_names = info.fn_names;
    for (size_t i = 0; i < fn_names.size(); ++i) {
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
        CCL_THROW_IF_NOT(ops[i], "dlsym is failed on: ", fn_names[i], ", error: ", dlerror());
        LOG_TRACE("dlsym loaded of ", fn_names.size(), " - ", i + 1, ": ", fn_names[i]);
    }
    return CCL_LOAD_LB_SUCCESS;
}

void print_error(int error, lib_info_t& info) {
    if (error == CCL_LOAD_LB_PATH_ERROR) {
        LOG_WARN("library path is not valid: ",
                 info.path.c_str(),
                 " - path contains invalid characters");
    }
    else if (error == CCL_LOAD_LB_DLOPEN_ERROR) {
        LOG_WARN("could not open the library: ", info.path.c_str(), " - ", dlerror());
    }
}

void close_library(lib_info_t& info) {
    if (info.handle) {
        dlclose(info.handle);
        info.handle = nullptr;
    }
}

} //namespace ccl
