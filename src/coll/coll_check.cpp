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
#include <cstdint>
#include <numeric>

#include "coll/coll.hpp"
#include "coll/coll_check.hpp"
#include "common/env/env.hpp"
#include "common/global/global.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"

void ccl_check_usm_pointers(const ccl_coll_param& param) {
    auto bufs = param.get_all_non_zero_bufs();
    if (bufs.empty()) {
        return;
    }

    auto dev = param.comm->get_device()->get_native();
    auto ctx = param.comm->get_context()->get_native();

    std::set<sycl::usm::alloc> usm_types;
    for (size_t idx = 0; idx < bufs.size(); idx++) {
        auto ptr_type = sycl::get_pointer_type(bufs[idx], ctx);
        usm_types.insert(ptr_type);
        LOG_DEBUG(
            "ptr: ", bufs[idx], ", usm pointer type: ", ccl::utils::usm_type_to_str(ptr_type));
    }

    if (usm_types.size() != 1) {
        auto first_usm_type = *usm_types.begin();
        auto second_usm_type = *(++usm_types.begin());
        CCL_THROW("coll: ",
                  ccl_coll_type_to_str(param.ctype),
                  ", mixed usm pointer types (",
                  ccl::utils::usm_type_to_str(first_usm_type),
                  ", ",
                  ccl::utils::usm_type_to_str(second_usm_type),
                  ") within single operation are not supported, ",
                  "device type: ",
                  ccl::utils::sycl_device_to_str(dev));
    }

    sycl::usm::alloc usm_type = *usm_types.begin();
    bool is_valid_type = true;

    if ((usm_type == sycl::usm::alloc::host) && (dev.is_gpu() || dev.is_accelerator()))
        is_valid_type = false;

    if ((usm_type == sycl::usm::alloc::device) && !(dev.is_gpu() || dev.is_accelerator()))
        is_valid_type = false;

    LOG_DEBUG("coll: ",
              ccl_coll_type_to_str(param.ctype),
              ", usm pointer type: ",
              ccl::utils::usm_type_to_str(usm_type),
              ", device type: ",
              ccl::utils::sycl_device_to_str(dev));

    CCL_THROW_IF_NOT(is_valid_type,
                     "coll: ",
                     ccl_coll_type_to_str(param.ctype),
                     " - invalid usm pointer type: ",
                     ccl::utils::usm_type_to_str(usm_type),
                     " for device type: ",
                     ccl::utils::sycl_device_to_str(dev));
}
#endif // CCL_ENABLE_SYCL

void ccl_coll_validate_user_input(const ccl_coll_param& param, const ccl_coll_attr& attr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().atl_transport == ccl_atl_ofi || !(attr.reduction_fn),
                     "custom reduction is supported for OFI transport only");

    CCL_THROW_IF_NOT(ccl_datatype_storage::is_predefined_datatype(param.dtype.idx()) ||
                         ccl::global_data::env().atl_transport == ccl_atl_ofi,
                     "custom datatype is supported for OFI transport only");

    CCL_THROW_IF_NOT((param.ctype != ccl_coll_allreduce && param.ctype != ccl_coll_reduce) ||
                         ccl_datatype_storage::is_predefined_datatype(param.dtype.idx()) ||
                         attr.reduction_fn,
                     "custom datatype requires custom reduction");

    CCL_THROW_IF_NOT(param.ctype == ccl_coll_allreduce || !(attr.reduction_fn),
                     "custom reduction is supported for allreduce only");

    //TODO: add vectorized support for ccl_coll_alltoall/v, when it's ready
    CCL_THROW_IF_NOT(param.ctype == ccl_coll_allgatherv || !(attr.is_vector_buf),
                     "vector buffer is not supported for ",
                     ccl_coll_type_to_str(param.ctype));

    if (param.ctype == ccl_coll_bcast || param.ctype == ccl_coll_broadcast ||
        param.ctype == ccl_coll_reduce) {
        CCL_THROW_IF_NOT(param.root < param.comm->size(),
                         "unexpected root ",
                         param.root,
                         ", comm size ",
                         param.comm->size());
    }

#ifdef CCL_ENABLE_SYCL
    if (param.comm->get_device() && param.comm->get_context() &&
        (param.comm->get_device()->get_native().is_gpu()) && param.stream) {
        CCL_THROW_IF_NOT(
            param.stream, "specify stream for ", ccl_coll_type_to_str(param.ctype), " operation");

        if (!attr.is_sycl_buf) {
            ccl_check_usm_pointers(param);
        }

        sycl::device comm_dev = param.comm->get_device()->get_native();
        sycl::context comm_ctx = param.comm->get_context()->get_native();
        auto stream_dev = param.stream->get_native_stream().get_device();
        auto stream_ctx = param.stream->get_native_stream().get_context();

        CCL_THROW_IF_NOT(comm_dev == stream_dev,
                         "device should match for communicator and stream"
                         ", comm_dev: ",
                         ccl::utils::sycl_device_to_str(comm_dev),
                         ", stream_dev: ",
                         ccl::utils::sycl_device_to_str(stream_dev));
        CCL_THROW_IF_NOT(comm_ctx == stream_ctx,
                         "context should match for communicator and stream");
    }
#endif // CCL_ENABLE_SYCL
}
