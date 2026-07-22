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

#include "common/global/global.hpp"
#include "sched/entry/entry.hpp"

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
class ze_reduce_local_entry;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

class reduce_local_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "REDUCE_LOCAL";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    reduce_local_entry() = delete;
    explicit reduce_local_entry(ccl_sched* sched,
                                const ccl_buffer in_buf,
                                size_t in_cnt,
                                ccl_buffer inout_buf,
                                size_t* out_cnt,
                                const ccl_datatype& dtype,
                                ccl::reduction op);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    void check_use_device();
    void start_on_device();
#else // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    void check_use_device() {}
    void start_on_device() {}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    void start_on_host();
    void start() override;
    void update() override;
    void reset(size_t idx) override;

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", in_buf ",
                           in_buf,
                           ", in_cnt ",
                           in_cnt,
                           ", inout_buf ",
                           inout_buf,
                           ", out_cnt ",
                           out_cnt,
                           ", op ",
                           ccl_reduction_to_str(op),
                           ", red_fn ",
                           fn,
                           "\n");
    }

private:
    const ccl_buffer in_buf;
    const size_t in_cnt;
    const ccl_buffer inout_buf;
    const size_t* out_cnt;
    const ccl_datatype dtype;
    const ccl::reduction op;
    const ccl::reduction_fn fn;

    bool use_device{};

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    std::unique_ptr<ze_reduce_local_entry> ze_reduce_local;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
};
