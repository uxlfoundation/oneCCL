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
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "common/stream/stream.hpp"
#include "common/stream/stream_selector_impl.hpp"
#include "common/utils/enums.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

namespace ccl {
std::string to_string(device_family family) {
    switch (family) {
        case device_family::family1: return "family1";
        case device_family::family2: return "family2";
        case device_family::family3: return "family3";
        case device_family::family4: return "family4";
        case device_family::family5: return "family5";
        case device_family::family6: return "family6";
        case device_family::family7: return "family7";
        default: return "unknown";
    }
}
} // namespace ccl

std::string to_string(const stream_type& type) {
    using stream_str_enum =
        ccl::utils::enum_to_str<ccl::utils::enum_to_underlying(stream_type::last_value)>;
    return stream_str_enum({ "host", "cpu", "gpu" }).choose(type, "unknown");
}

ccl_stream::ccl_stream(stream_type type,
                       stream_native_t& stream,
                       const ccl::library_version& version)
        : version(version),
          type(type),
          device_family(ccl::device_family::unknown) {
    native_stream = stream;

#ifdef CCL_ENABLE_SYCL
    sycl::property_list props{};
    if (stream.is_in_order()) {
        props = { sycl::property::queue::in_order{} };
    }
    else {
        props = {};
    }

    native_streams.resize(ccl::global_data::env().worker_count);
    for (auto& native_stream : native_streams) {
        native_stream = stream_native_t(stream.get_context(), stream.get_device(), props);
    }

    backend = stream.get_device().get_backend();
#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_ZE
    if (backend == ccl::utils::get_level_zero_backend() && ccl::global_data::get().ze_data) {
        device = sycl::get_native<ccl::utils::get_level_zero_backend()>(stream.get_device());
        context = sycl::get_native<ccl::utils::get_level_zero_backend()>(stream.get_context());
        device_family = ccl::ze::get_device_family(device);

        ccl::ze::ze_queue_properties_t queue_props;
        ccl::ze::get_queues_properties(device, &queue_props);

        if (!ccl::global_data::env().disable_ze_family_check) {
            if (queue_props.size() == 1 && queue_props.front().numQueues == 1 &&
                (device_family == ccl::device_family::unknown)) {
                ze_device_properties_t dev_props = ccl::ze::default_device_props;
                ZE_CALL(zeDeviceGetProperties, (device, &dev_props));
                bool is_integrated = dev_props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;

                if (!is_integrated) {
                    LOG_WARN("usage of discrete device with unexpected properties"
                             " (id: ",
                             dev_props.deviceId,
                             ", name: ",
                             (strlen(dev_props.name) ? dev_props.name : "<empty>"),
                             ", flags: ",
                             ccl::ze::flags_to_string<ze_device_property_flag_t>(dev_props.flags),
                             ")"
                             ", set ",
                             CCL_ZE_DISABLE_FAMILY_CHECK,
                             "=1 to hide this message");
                }
            }
        }
    }
#endif // CCL_ENABLE_ZE
}

// export attributes
typename ccl_stream::version_traits_t::type ccl_stream::set_attribute_value(
    typename version_traits_t::type val,
    const version_traits_t& t) {
    (void)t;
    throw ccl::exception("set value for 'ccl::stream_attr_id::library_version' is not allowed");
    return version;
}

const typename ccl_stream::version_traits_t::return_type& ccl_stream::get_attribute_value(
    const version_traits_t& id) const {
    return version;
}

typename ccl_stream::native_handle_traits_t::return_type& ccl_stream::get_attribute_value(
    const native_handle_traits_t& id) {
    return native_stream;
}

std::string ccl_stream::to_string() const {
    std::stringstream ss;
#ifdef CCL_ENABLE_SYCL
    ss << "{ "
       << "type: " << ::to_string(type) << ", in_order: " << native_stream.is_in_order()
       << ", device: " << native_stream.get_device().get_info<sycl::info::device::name>()
       << ", device_family: " << ccl::to_string(device_family) << " }";
#else // CCL_ENABLE_SYCL
    ss << reinterpret_cast<void*>(native_stream.get());
#endif // CCL_ENABLE_SYCL
    return ss.str();
}

stream_type ccl_stream::get_type() const {
    return type;
}

ccl::device_family ccl_stream::get_device_family() const {
    return device_family;
}

bool ccl_stream::is_sycl_device_stream() const {
    return (type == stream_type::cpu || type == stream_type::gpu);
}

bool ccl_stream::is_gpu() const {
    return type == stream_type::gpu;
}

#ifdef CCL_ENABLE_SYCL
sycl::backend ccl_stream::get_backend() const {
    return backend;
}
#ifdef CCL_ENABLE_ZE

ze_device_handle_t ccl_stream::get_ze_device() const {
    CCL_THROW_IF_NOT(backend == ccl::utils::get_level_zero_backend());
    CCL_THROW_IF_NOT(device, "no device");
    return device;
}

ze_context_handle_t ccl_stream::get_ze_context() const {
    CCL_THROW_IF_NOT(backend == ccl::utils::get_level_zero_backend());
    CCL_THROW_IF_NOT(context, "no context");
    return context;
}
#endif // CCL_ENABLE_ZE
#endif // CCL_ENBALE_SYCL
