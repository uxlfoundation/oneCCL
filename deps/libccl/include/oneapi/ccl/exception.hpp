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

//#include <sycl/sycl.hpp>
#include "oneapi/ccl/string.hpp"
#include <exception>
#include <string>

namespace ccl {

namespace v1 {

class exception : public std::exception {
    ccl::string msg;

public:
    exception(const ccl::string &domain, const ccl::string &function, const ccl::string &info = "")
            : std::exception() {
        msg = ccl::string("oneCCL: ") + domain +
              ((domain.length() != 0 && function.length() != 0) ? "/" : "") + function +
              ((info.length() != 0)
                   ? (((domain.length() + function.length() != 0) ? ": " : "") + info)
                   : "");
    }

    exception(const ccl::string &info = "") : std::exception() {
        msg = ccl::string("oneCCL: ") + info;
    }

    exception(const char *info) : std::exception() {
        msg = ccl::string("oneCCL: ") + ccl::string(info);
    }

    const char *what() const noexcept override {
        return msg.c_str();
    }
};

class invalid_argument : public exception {
public:
    invalid_argument(const ccl::string &domain,
                     const ccl::string &function,
                     const ccl::string &info = "")
            : exception(domain, function, "invalid argument " + info) {}
};

class host_bad_alloc : public exception {
public:
    host_bad_alloc(const ccl::string &domain, const ccl::string &function)
            : exception(domain, function, "cannot allocate memory on host") {}
};

// class device_bad_alloc : public exception {
// public:
//     device_bad_alloc(const ccl::string &domain, const ccl::string &function,
//                      const sycl::device &device)
//             : exception(
//                   domain, function,
//                   "cannot allocate memory on " + device.get_info<sycl::info::device::name>()) {}
// };

class unimplemented : public exception {
public:
    unimplemented(const ccl::string &domain,
                  const ccl::string &function,
                  const ccl::string &info = "")
            : exception(domain, function, "function is not implemented " + info) {}
};

class unsupported : public exception {
public:
    unsupported(const ccl::string &domain,
                const ccl::string &function,
                const ccl::string &info = "")
            : exception(domain, function, "function is not supported " + info) {}
};

} // namespace v1

using v1::exception;
using v1::invalid_argument;
using v1::host_bad_alloc;
// using v1::device_bad_alloc;
using v1::unimplemented;
using v1::unsupported;

} // namespace ccl
