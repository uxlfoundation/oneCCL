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

#include <sstream>

#include "oneapi/ccl/config.h"
#include "oneapi/ccl/lp_types.hpp"
#include "oneapi/ccl/types.hpp"
#include "common/utils/enums.hpp"
#include "oneapi/ccl/device_types.hpp"

namespace ccl {

std::string to_string(const bfloat16& v) {
    return std::to_string(v.get_data());
}

std::string to_string(const float16& v) {
    return std::to_string(v.get_data());
}

CCL_API ccl::string to_string(const device_index_type& device_id) {
    std::stringstream ss;
    ss << "[" << std::get<ccl::device_index_enum::driver_index_id>(device_id) << ":"
       << std::get<ccl::device_index_enum::device_index_id>(device_id) << ":";

    auto subdevice_id = std::get<ccl::device_index_enum::subdevice_index_id>(device_id);
    if (subdevice_id == ccl::unused_index_value) {
        ss << "*";
    }
    else {
        ss << std::get<ccl::device_index_enum::subdevice_index_id>(device_id);
    }
    ss << "]";

    return ss.str();
}

CCL_API device_index_type from_string(const ccl::string& device_id_str) {
    std::string device_id_native_str = std::string(device_id_str);
    std::string::size_type from_pos = device_id_native_str.find('[');
    if (from_pos == std::string::npos) {
        throw std::invalid_argument(std::string("Cannot get ccl::device_index_type from input: ") +
                                    device_id_native_str);
    }

    if (device_id_native_str.size() == 1) {
        throw std::invalid_argument(
            std::string("Cannot get ccl::device_index_type from input, too less: ") +
            device_id_native_str);
    }
    from_pos++;

    device_index_type path(unused_index_value, unused_index_value, unused_index_value);

    size_t cur_index = 0;
    do {
        std::string::size_type to_pos = device_id_native_str.find(':', from_pos);
        std::string::size_type count =
            (to_pos != std::string::npos ? to_pos - from_pos : std::string::npos);
        std::string index_string(device_id_native_str, from_pos, count);
        switch (cur_index) {
            case device_index_enum::driver_index_id: {
                auto index = std::atoll(index_string.c_str());
                if (index < 0) {
                    throw std::invalid_argument(
                        std::string("Cannot get ccl::device_index_type from input, "
                                    "driver index invalid: ") +
                        device_id_native_str);
                }
                std::get<device_index_enum::driver_index_id>(path) = index;
                break;
            }
            case device_index_enum::device_index_id: {
                auto index = std::atoll(index_string.c_str());
                if (index < 0) {
                    throw std::invalid_argument(
                        std::string("Cannot get ccl::device_index_type from input, "
                                    "device index invalid: ") +
                        device_id_native_str);
                }
                std::get<device_index_enum::device_index_id>(path) = index;
                break;
            }
            case device_index_enum::subdevice_index_id: {
                auto index = std::atoll(index_string.c_str());
                std::get<device_index_enum::subdevice_index_id>(path) =
                    index < 0 ? unused_index_value : index;
                break;
            }
            default:
                throw std::invalid_argument(
                    std::string("Cannot get ccl::device_index_type from input, "
                                "unsupported format: ") +
                    device_id_native_str);
        }

        cur_index++;
        if (device_id_native_str.size() > to_pos) {
            to_pos++;
        }
        from_pos = to_pos;
    } while (from_pos < device_id_native_str.size());

    return path;
}
} // namespace ccl
