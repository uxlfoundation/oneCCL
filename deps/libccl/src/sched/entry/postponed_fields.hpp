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

#include <map>
#include <set>
#include <tuple>

#include "oneapi/ccl/types.hpp"
#include "common/log/log.hpp"
#include "common/utils/tuple.hpp"
#include "internal_types.hpp"

enum ccl_sched_entry_field_id {
    ccl_sched_entry_field_buf,
    ccl_sched_entry_field_send_buf,
    ccl_sched_entry_field_recv_buf,
    ccl_sched_entry_field_cnt,
    ccl_sched_entry_field_dtype,
    ccl_sched_entry_field_src_mr,
    ccl_sched_entry_field_dst_mr,
    ccl_sched_entry_field_in_buf,
    ccl_sched_entry_field_in_cnt,
    ccl_sched_entry_field_in_dtype,
    ccl_sched_entry_field_idx_buf,
    ccl_sched_entry_field_idx_cnt,
    ccl_sched_entry_field_val_buf,
    ccl_sched_entry_field_val_cnt,
    ccl_sched_entry_field_send_count
};

typedef ccl::status (*ccl_sched_entry_field_function_t)(const void*, void*);

template <ccl_sched_entry_field_id id>
using field_id_t = std::integral_constant<ccl_sched_entry_field_id, id>;

template <ccl_sched_entry_field_id id>
class postponed_field {
public:
    postponed_field() = default;
    postponed_field(ccl_sched_entry_field_function_t fn, const void* ctx, bool update_once)
            : fn(fn),
              ctx(ctx),
              update_once(update_once) {}

    template <class Entry>
    void operator()(Entry entry) {
        CCL_ASSERT(fn);
        fn(ctx, reinterpret_cast<void*>(&(entry->get_field_ref(entry_field_id))));
        if (update_once)
            fn = nullptr;
    }

    bool empty() const noexcept {
        return !fn;
    }

    ccl_sched_entry_field_function_t fn = nullptr;
    const void* ctx = nullptr;
    bool update_once{};
    static constexpr field_id_t<id> entry_field_id{};
};

template <class Entry, ccl_sched_entry_field_id... ids>
struct postponed_fields {
    template <class Arg>
    struct field_functor {
        field_functor(Arg arg, bool& updated) : arg(arg), updated(updated) {}
        template <typename T>
        void operator()(T& t) const {
            if (!t.empty()) {
                t(arg);
                updated = true;
            }
        }
        Arg arg;
        bool& updated;
    };

    using registered_postponed_fields = std::tuple<postponed_field<ids>...>;

    template <ccl_sched_entry_field_id new_id>
    void set_field_fn(ccl_sched_entry_field_function_t fn,
                      const void* ctx,
                      bool update_once = true) {
        auto& field = ccl_tuple_get<postponed_field<new_id>>(fields);
        CCL_ASSERT(field.empty(), "duplicated field_id ", new_id);
        field.fn = fn;
        field.ctx = ctx;
        field.update_once = update_once;

        empty_fields = false;
    }

    bool update_fields() {
        bool updated = false;
        if (!empty_fields) {
            ccl_tuple_for_each(fields, field_functor<Entry*>(static_cast<Entry*>(this), updated));
        }
        empty_fields = !updated;
        return updated;
    }

protected:
    ~postponed_fields() = default;

private:
    registered_postponed_fields fields;
    bool empty_fields = true;
};
