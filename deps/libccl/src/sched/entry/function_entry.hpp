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

#include "sched/entry/entry.hpp"

class function_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "FUNCTION";
    }

    function_entry() = delete;
    function_entry(ccl_sched* sched, ccl_sched_entry_function_t fn, const void* ctx)
            : sched_entry(sched),
              fn(fn),
              ctx(ctx) {}

    void start() override {
        fn(ctx);
        status = ccl_sched_entry_status_complete;
    }

    const char* name() const override {
        return "FUNCTION";
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str, "fn ", (void*)(fn), ", ctx ", ctx, "\n");
    }

private:
    ccl_sched_entry_function_t fn;
    const void* ctx;
};
