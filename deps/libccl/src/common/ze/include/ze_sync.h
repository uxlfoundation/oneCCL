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

#include "ze_events.h"
#include <vector>

constexpr auto MAX_EVENTS = 32;

/**
 * @brief A collection of event handles.
 */
class ze_event_handles_t : public std::vector<ze_event_handle_t>
{
public:
    using std::vector<ze_event_handle_t>::vector;
    uint32_t size() const { return static_cast<uint32_t>(std::vector<ze_event_handle_t>::size()); }
};

template <unsigned N> class _ze_sync_t
{
protected:
    ze_event_handles_t m_handles;
    _ze_sync_t() { m_handles.reserve(N); }
};

class ze_sync_t : public _ze_sync_t<MAX_EVENTS>
{
private:
    ze_event_ptr m_signal = nullptr;
    ze_events_t m_waits;

    void populate()
    {
        for (const auto &e : m_waits)
        {
            m_handles.push_back(e ? e->handle() : nullptr);
        }
    }

public:
    ze_sync_t() = default;
    ze_sync_t(ze_event_ptr signal, const ze_events_t& waits = {})
        : m_signal(signal), m_waits(waits)
    {
        CCL_ASSERT(m_waits.size() <= MAX_EVENTS, "Too many wait events: ", m_waits.size(), " > ", MAX_EVENTS);
        populate();
    }

    ze_event_handle_t signal() const
    {
        return m_signal ? m_signal->handle() : nullptr;
    }

    const ze_event_handles_t& waits() const { return m_handles; }
};
