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
#pragma once

#include <sycl/sycl.hpp>

#include "coll/algorithms/utils/tvisa/include/gen_visa_templates.hpp"

#define divUp(x, m) (((x) + (m)-1) / (m))

#define alignUp(x, c) (divUp((x), (c)) * (c))

#include "coll/algorithms/utils/rt64.hpp"
#include "coll/algorithms/utils/rt64_128.hpp"

#include "ring_transmit.hpp"

extern uint32_t pattern_counter;
