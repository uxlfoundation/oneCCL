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
#include "sched/sched.hpp"

#include <functional>
#include <list>
#include <memory>

// declares interface for all entries creations
namespace entry_factory {
template <class EntryType, class... Arguments>
EntryType* create(ccl_sched* sched, Arguments&&... args);

namespace detail {
template <class EntryType>
struct entry_creator {
    template <class T, class... U>
    friend T* create(ccl_sched* sched, U&&... args);

    template <class T, ccl_sched_add_mode mode, class... U>
    friend T* create(ccl_sched* sched, U&&... args);

    template <ccl_sched_add_mode mode, class... Arguments>
    static EntryType* make_entry(ccl_sched* sched, Arguments&&... args) {
        return static_cast<EntryType*>(sched->add_entry(
            std::unique_ptr<EntryType>(new EntryType(sched, std::forward<Arguments>(args)...)),
            ccl_sched_base::add_entry_mode_t<mode>()));
    }
};
} // namespace detail
} // namespace entry_factory
