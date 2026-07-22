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

#include "common/log/log.hpp"
#include "common/utils/utils.hpp"

namespace ccl {
namespace utils {

size_t get_ptr_diff(const void* ptr1, const void* ptr2) {
    return static_cast<const char*>(ptr2) - static_cast<const char*>(ptr1);
}

size_t pof2(size_t number) {
    size_t last_bit_mask = ((size_t)1 << (8 * sizeof(size_t) - 1));
    if (number & last_bit_mask) {
        return last_bit_mask;
    }

    size_t pof2 = 1;
    while (pof2 <= number) {
        pof2 <<= 1;
    }
    pof2 >>= 1;
    return pof2;
}

size_t aligned_sz(size_t size, size_t alignment) {
    return ((size % alignment) == 0) ? size : ((size / alignment) + 1) * alignment;
}

std::string get_substring_between_delims(std::string& full_str,
                                         const std::string& start_delim,
                                         const std::string& stop_delim) {
    auto first_delim_pos = full_str.find(start_delim);
    auto end_pos_of_first_delim = first_delim_pos + start_delim.length();
    auto last_delim_pos = full_str.find(stop_delim);

    CCL_THROW_IF_NOT(last_delim_pos > first_delim_pos,
                     "incorrect delim positions: {first delim: ",
                     first_delim_pos,
                     ", last delim: ",
                     last_delim_pos,
                     "}");

    return full_str.substr(end_pos_of_first_delim, last_delim_pos - end_pos_of_first_delim);
}

void str_to_array(const std::string& input_str,
                  std::string delimiter,
                  std::vector<std::string>& result) {
    size_t last = 0;
    size_t next = 0;
    while ((next = input_str.find(delimiter, last)) != std::string::npos) {
        auto substr = input_str.substr(last, next - last);
        CCL_THROW_IF_NOT(substr.size() != 0, "unexpected string size: ", substr.size());
        result.push_back(input_str.substr(last, next - last));
        last = next + 1;
    }
    result.push_back(input_str.substr(last));
}

uintptr_t get_aligned_offset_byte(const void* ptr,
                                  const size_t buf_size_bytes,
                                  const size_t mem_align_bytes) {
    // find the number of data items to remove to start from aligned bytes
    unsigned long pre_align_offset_byte = reinterpret_cast<uintptr_t>(ptr) % mem_align_bytes;
    if (pre_align_offset_byte != 0) {
        pre_align_offset_byte = mem_align_bytes - pre_align_offset_byte;
    }
    // make sure to use only the required number of threads for very small data count
    if (buf_size_bytes < pre_align_offset_byte) {
        pre_align_offset_byte = buf_size_bytes;
    }
    return pre_align_offset_byte;
}

std::string join_strings(const std::vector<std::string>& tokens, const std::string& delimeter) {
    std::stringstream ss;
    for (size_t i = 0; i < tokens.size(); ++i) {
        ss << tokens[i];
        if (i < tokens.size() - 1) {
            ss << delimeter;
        }
    }
    return ss.str();
}

std::string to_hex(const char* data, size_t size) {
    std::stringstream ss;
    ss << "0x";
    for (size_t i = 0; i < size; ++i) {
        ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

void close_fd(int fd) {
    LOG_DEBUG("closing fd: ", fd);
    close(fd);
}

} // namespace utils
} // namespace ccl
