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

namespace ccl {

enum class group_split_type : int {
    cluster,

    last_value
};

// TODO: refactor core code and remove this enum?
enum status : int {
    success = 0,
    out_of_resource,
    invalid_arguments,
    runtime_error,
    blocked_due_to_resize,

    last_value
};

} // namespace ccl
