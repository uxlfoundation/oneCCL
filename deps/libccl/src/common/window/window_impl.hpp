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

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/device_api/window.hpp"

class ccl_comm;

namespace ccl {

namespace v1 {

// Single implementation behind both public entry points. In ccl_api_functions.cpp.
// Explicitly hidden: -fvisibility=internal does not take on this library.
#define CCL_WINDOW_LOCAL __attribute__((visibility("hidden")))

CCL_WINDOW_LOCAL window_t
window_register_internal(ccl_comm* comm, void* buffer, size_t size, int win_flags);
CCL_WINDOW_LOCAL void window_deregister_internal(ccl_comm* comm, window_t& win);

} // namespace v1

// PIMPL body of ccl::v1::window. The class name is v1 ABI (it is spelled inside the
// exported window(shared_ptr<ccl_window_impl>&&) symbol); the layout is not, so this
// can be a thin adapter that keeps window RAII over the window_t path.
class ccl_window_impl {
public:
    ccl_window_impl(ccl_comm* comm, v1::window_t win) : comm_(comm), win_(win) {}

    // Runs from noexcept ~window(); a throw here would abort.
    ~ccl_window_impl() {
        try {
            v1::window_deregister_internal(comm_, win_);
        }
        catch (...) {
        }
    }

    ccl_window_impl(const ccl_window_impl&) = delete;
    ccl_window_impl& operator=(const ccl_window_impl&) = delete;

    v1::window_t get_window() const {
        return win_;
    }

private:
    ccl_comm* comm_; // not owned, as in 2022.1.0
    v1::window_t win_;
};

} // namespace ccl
