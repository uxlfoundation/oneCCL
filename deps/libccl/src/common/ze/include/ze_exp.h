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
#include "zex_event.h" 

ze_result_t zexInit(ze_driver_handle_t driver);

// -----------------------  experimental API functions should be called after zexInit()  -----------------------

ze_result_t ZE_APICALL cclCounterBasedEventCreate2(ze_context_handle_t,  ze_device_handle_t, const zex_counter_based_event_desc_t*, ze_event_handle_t*);
