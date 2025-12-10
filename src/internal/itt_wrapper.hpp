/*
 Copyright 2016-2025 Intel Corporation
 
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

#ifndef ITT_WRAPPER_HPP
#define ITT_WRAPPER_HPP

#include <ittnotify.h>
#include <string>

namespace itt {

class Task {
public:
    explicit Task(const std::string& name) 
        : domain_(__itt_domain_create("oneCCL2")),
          handle_(__itt_string_handle_create(name.c_str()))
    {}

    void start() {
        __itt_task_begin(domain_, __itt_null, __itt_null, handle_);
    }

    void end() {
        __itt_task_end(domain_);
    }

private:
    __itt_domain* domain_;
    __itt_string_handle* handle_;
};

} // namespace itt

#endif // ITT_WRAPPER_HPP
