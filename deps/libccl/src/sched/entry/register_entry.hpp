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
#include "exec/exec.hpp"
#include "sched/entry/entry.hpp"

class register_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "REGISTER";
    }

    register_entry() = delete;
    register_entry(ccl_sched* sched,
                   size_t size,
                   const ccl_buffer ptr,
                   atl_mr_t** mr,
                   ccl_comm* comm)
            : sched_entry(sched, true),
              size(size),
              ptr(ptr),
              mr(mr),
              comm(comm) {}

    void start() override {
        LOG_DEBUG("REGISTER entry size ", size, ", ptr ", ptr);
        CCL_THROW_IF_NOT(
            size > 0 && ptr && mr, "incorrect input, size ", size, ", ptr ", ptr, " mr ", mr);

        atl_status_t atl_status = comm->get_atl_comm()->mr_reg(ptr.get_ptr(size), size, mr);

        sched->add_memory_region(*mr);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("REGISTER entry failed. atl_status: ", atl_status_to_str(atl_status));
        }
        else
            status = ccl_sched_entry_status_complete;
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str, "sz ", size, ", ptr ", ptr, ", mr ", mr, "\n");
    }

private:
    size_t size;
    ccl_buffer ptr;
    atl_mr_t** mr;
    ccl_comm* comm;
};
