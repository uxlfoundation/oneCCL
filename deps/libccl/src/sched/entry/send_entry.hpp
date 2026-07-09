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
#include "sched/queue/queue.hpp"
#include "sched/entry/copy/copy_entry.hpp"

class send_entry : public sched_entry,
                   public postponed_fields<send_entry,
                                           ccl_sched_entry_field_buf,
                                           ccl_sched_entry_field_cnt> {
public:
    static constexpr const char* class_name() noexcept {
        return "SEND";
    }

    send_entry() = delete;
    send_entry(ccl_sched* sched,
               const ccl_buffer buf,
               size_t cnt,
               const ccl_datatype& dtype,
               int dst,
               ccl_comm* comm)
            : sched_entry(sched),
              buf(buf),
              cnt(cnt),
              dtype(dtype),
              dst(dst),
              comm(comm) {
#ifdef CCL_ENABLE_SYCL
        if (sched->coll_param.stream && cnt &&
            (ccl::global_data::env().atl_send_proxy != ccl_atl_send_proxy_none) &&
            (proxy_mode == proxy_copy_mode::unknown)) {
            sycl::usm::alloc ptr_type = sycl::usm::alloc::unknown;
            if (sched->coll_param.stream->is_gpu()) {
                auto sycl_queue = sched->coll_param.stream->get_native_stream();
                ptr_type = sycl::get_pointer_type(buf.get_ptr(), sycl_queue.get_context());
            }
            proxy_mode = (ptr_type == sycl::usm::alloc::device) ? proxy_copy_mode::enabled
                                                                : proxy_copy_mode::disabled;
        }

        if (proxy_mode == proxy_copy_mode::enabled) {
            if (!proxy_buf) {
                ccl::buffer_type buf_type =
                    (ccl::global_data::env().atl_send_proxy == ccl_atl_send_proxy_regular)
                        ? ccl::buffer_type::regular
                        : ccl::buffer_type::sycl;
                ccl::alloc_param alloc_param(
                    cnt * dtype.size(), buf_type, ccl::buffer_place::host, 1);
                proxy_buf = sched->alloc_buffer(alloc_param);
            }
            proxy_copy_entry = std::make_unique<copy_entry>(sched, buf, proxy_buf, cnt, dtype);
        }
#endif // CCL_ENABLE_SYCL
    }

    void start_send() {
        uint16_t sched_id = sched->sched_id;
        if (sched->coll_param.ctype == ccl_coll_send) {
            sched_id = comm->get_atl_comm()->tag_creator->get_pt2pt_sched_id();
        }

        atl_tag = comm->get_atl_comm()->tag_creator->create(
            comm->rank(), comm->get_comm_id(), sched_id, sched->get_op_id());
        size_t bytes = cnt * dtype.size();
        LOG_DEBUG("SEND entry dst ", dst, ", tag ", atl_tag, ", req ", req, ", bytes ", bytes);

        atl_status_t atl_status = comm->get_atl_comm()->send(
            sched->bin->get_atl_ep(), send_buf.get_ptr(bytes), bytes, dst, atl_tag, req);

        update_status(atl_status);
    }

    void reset(size_t idx) override {
        sched_entry::reset(idx);
#ifdef CCL_ENABLE_SYCL
        if (proxy_copy_entry) {
            proxy_copy_entry->reset(idx);
        }
#endif // CCL_ENABLE_SYCL
    }

    void start() override {
        update_fields();

        send_buf = buf;

#ifdef CCL_ENABLE_SYCL
        if (proxy_mode == proxy_copy_mode::enabled) {
            proxy_copy_entry->do_progress();

            if (proxy_copy_entry->get_status() != ccl_sched_entry_status_complete) {
                status = ccl_sched_entry_status_again;
                return;
            }

            send_buf = proxy_buf;
        }
#endif // CCL_ENABLE_SYCL

        start_send();
    }

    void update() override {
        atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("SEND entry failed. atl_status: ", atl_status_to_str(atl_status));
        }

        if (req.is_completed) {
            LOG_DEBUG("SEND entry done, dst ", dst);
            status = ccl_sched_entry_status_complete;
        }
    }

    const char* name() const override {
        return class_name();
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_buf> id) {
        return buf;
    }

    size_t& get_field_ref(field_id_t<ccl_sched_entry_field_cnt> id) {
        return cnt;
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", cnt ",
                           cnt,
                           ", buf ",
                           buf,
                           ", dst ",
                           dst,
                           ", atl_tag ",
                           atl_tag,
                           ", comm_id ",
                           comm->get_comm_id(),
                           ", req ",
                           req,
                           "\n");
    }

private:
    ccl_buffer buf;
    size_t cnt;
    ccl_datatype dtype;
    int dst;
    ccl_comm* comm;
    uint64_t atl_tag = 0;
    atl_req_t req{};

    ccl_buffer send_buf;

#ifdef CCL_ENABLE_SYCL
    enum class proxy_copy_mode { unknown, enabled, disabled };
    proxy_copy_mode proxy_mode = proxy_copy_mode::unknown;
    std::unique_ptr<copy_entry> proxy_copy_entry;
    ccl_buffer proxy_buf{};
#endif // CCL_ENABLE_SYCL
};
