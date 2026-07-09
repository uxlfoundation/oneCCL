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

#pragma once
#include <array>
#include <iostream>
#include <type_traits>

namespace ccl {
namespace utils {
namespace detail {
struct failure_callback {
    template <class T>
    static const char *invoke(T val, const char *message) noexcept {
        std::cerr << __FUNCTION__ << ": " << message << ": " << val << std::endl;
        return message;
    }
};
} // namespace detail

template <int Limit>
struct enum_to_str {
    template <class... Args>
    constexpr enum_to_str(Args &&... names) : m_names({ { names... } }) {
        static_assert(Limit == sizeof...(names), "Enum size and descriptions are mismatched");
    }

    template <class EnumClass>
    const char *choose(EnumClass value, const char *error_str = "unexpected value") noexcept {
        using underlying_t = typename std::underlying_type<EnumClass>::type;
        underlying_t counted_value = static_cast<underlying_t>(value);
        return (static_cast<typename std::make_unsigned<underlying_t>::type>(value) < Limit
                    ? m_names[counted_value]
                    : error_str);
    }

private:
    std::array<const char *, Limit> m_names;
};

template <int Limit, class Function>
struct enum_to_str_s : public enum_to_str<Limit> {
    using Base = enum_to_str<Limit>;
    template <class... Args>
    constexpr enum_to_str_s(Args &&... names) : Base(std::forward<Args>(names)...) {
        static_assert(Limit == sizeof...(names), "Enum size and descriptions are mismatched");
    }

    template <class EnumClass>
    const char *choose(EnumClass value, const char *error_str = "unexpected value") noexcept(
        noexcept(Function::invoke(value, error_str))) {
        const char *def = "";
        const char *ret = Base::choose(value, def);
        if (ret == def) {
            return Function::invoke(value, error_str);
        }
        return ret;
    }
};

template <class enumeration_type>
constexpr typename std::underlying_type<enumeration_type>::type enum_to_underlying(
    enumeration_type val) noexcept {
    return static_cast<typename std::underlying_type<enumeration_type>::type>(val);
}
} // namespace utils
} // namespace ccl
