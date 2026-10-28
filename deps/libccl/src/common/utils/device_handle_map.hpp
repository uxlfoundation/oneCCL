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

#include "common/log/log.hpp"

#include <mutex>
#include <unordered_map>

namespace ccl {
namespace utils {

/*
 * Maps a device address to the host-side state kept alongside it.
 *
 * The key is whatever device pointer the device API hands out: a window_t, or a
 * device communicator's inbox array. The record holds the host-only state that
 * goes with it — IPC handle lifetime, peer pointers for the host-side resolves,
 * and a mirror of what was published to the device.
 *
 * insert() takes ownership once the allocation is published, at() borrows the
 * record for a lookup, and remove() hands it back and forgets the key. Every
 * access is serialized on one mutex.
 */
template <typename Record>
class device_handle_map {
public:
    /** Takes ownership of the host state of a freshly published handle. */
    void insert(void* handle, Record&& record) {
        std::lock_guard<std::mutex> lock(guard);
        records.emplace(handle, std::move(record));
    }

    /** Borrows a record; keep the returned lock alive for as long as it is read. */
    std::unique_lock<std::mutex> at(const char* caller, void* handle, const Record** out_record) {
        std::unique_lock<std::mutex> lock(guard);
        *out_record = &find_or_throw(caller, handle)->second;
        return lock;
    }

    /** Hands the host state over and forgets the handle. */
    Record remove(const char* caller, void* handle) {
        std::lock_guard<std::mutex> lock(guard);
        auto it = find_or_throw(caller, handle);
        Record record = std::move(it->second);
        records.erase(it);
        return record;
    }

private:
    using map_type = std::unordered_map<void*, Record>;

    typename map_type::iterator find_or_throw(const char* caller, void* handle) {
        auto it = records.find(handle);
        if (it == records.end()) {
            CCL_THROW(caller, ": unknown handle ", handle);
        }
        return it;
    }

    std::mutex guard;
    map_type records;
};

} // namespace utils
} // namespace ccl
