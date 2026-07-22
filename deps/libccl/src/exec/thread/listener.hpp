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

// TODO: Rework to support listener

#include "exec/thread/base_thread.hpp"

//class ccl_listener : public ccl_base_thread {
//public:
//    ccl_listener();
//
//    ccl_listener(const ccl_listener&) = delete;
//    ccl_listener(ccl_listener&&) = delete;
//
//    ccl_listener& operator=(const ccl_listener&) = delete;
//    ccl_listener& operator=(ccl_listener&&) = delete;
//
//    virtual ~ccl_listener() = default;
//
//    virtual void* get_this() override {
//        return static_cast<void*>(this);
//    };
//
//    virtual const std::string& name() const override {
//        static const std::string name("listener");
//        return name;
//    };
//};
