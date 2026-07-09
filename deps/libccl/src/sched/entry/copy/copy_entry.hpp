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

#include "sched/entry/copy/copy_helper.hpp"
#include "sched/entry/entry.hpp"

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
class ze_copy_entry;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

enum class copy_type : int { regular, sycl, ze };

class copy_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "COPY";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    copy_entry() = delete;
    copy_entry(ccl_sched* sched,
               ccl_buffer in_buf,
               ccl_buffer out_buf,
               size_t count,
               const ccl_datatype& dtype,
               copy_attr attr = {});

    void start() override;
    void update() override;
    void reset(size_t idx) override;

protected:
    void dump_detail(std::stringstream& str) const override;

private:
    ccl_buffer in_buf{};
    ccl_buffer out_buf{};
    const size_t count;
    const ccl_datatype dtype;
    copy_attr attr;

    copy_type ctype{ copy_type::regular };

#ifdef CCL_ENABLE_SYCL
    int is_sycl_buf{};
    sycl_copier copier{};
#ifdef CCL_ENABLE_ZE
    std::unique_ptr<ze_copy_entry> ze_copier;
#endif // CCL_ENABLE_ZE
#endif // CCL_ENABLE_SYCL

    void do_regular_copy();
};
