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

class recv_copy_entry final : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "RECV_COPY";
    }

    recv_copy_entry() = delete;
    recv_copy_entry(ccl_sched* sched,
                    ccl_buffer recv_buf,
                    ccl_buffer copy_buf,
                    size_t bytes,
                    int src,
                    ccl_comm* comm,
                    copy_attr attr)
            : sched_entry(sched),
              recv_buf(recv_buf),
              copy_buf(copy_buf),
              bytes(bytes),
              src(src),
              comm(comm),
              attr(attr) {}

    void start() override;
    void update() override;

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override;

private:
    ccl_buffer recv_buf;
    ccl_buffer copy_buf;
    size_t bytes;
    int src;
    ccl_comm* comm;
    copy_attr attr;

    uint64_t atl_tag = 0;
    atl_req_t req{};
};
