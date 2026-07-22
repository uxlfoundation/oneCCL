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

class deregister_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "DEREGISTER";
    }

    deregister_entry() = delete;
    deregister_entry(ccl_sched* sched, std::list<atl_mr_t*>& mr_list, ccl_comm* comm)
            : sched_entry(sched, true),
              mr_list(mr_list),
              comm(comm) {}

    void start() override {
        LOG_DEBUG("DEREGISTER entry sched ", sched, " mr_count ", mr_list.size());
        atl_status_t atl_status;
        std::list<atl_mr_t*>::iterator it;
        for (it = mr_list.begin(); it != mr_list.end(); it++) {
            LOG_DEBUG("deregister mr ", *it);
            atl_status = comm->get_atl_comm()->mr_dereg(*it);
            if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
                CCL_THROW("DEREGISTER entry failed. atl_status: ", atl_status_to_str(atl_status));
            }
        }
        mr_list.clear();
        status = ccl_sched_entry_status_complete;
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str, "sched ", sched, ", mr_count ", sched, mr_list.size(), "\n");
    }

private:
    std::list<atl_mr_t*>& mr_list;
    ccl_comm* comm;
};
