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
#include "ze_helpers.h"
#include <functional>
#include "occp/inc/defines.h"   // for untransferable_t


/**
 * @brief Base class for Level Zero handles.
 *
 * A template class that provides a common interface for Level Zero handles.
 * All classes representing Level Zero handles should inherit from this class.
 *
 * @tparam T - Level Zero handle
 */
template<class T>
class ze_handle_t : public untransferable_t
{
    using ze_dtor_t = std::function<void(T)>;

protected:
    T         m_handle = nullptr;
    ze_dtor_t m_dtor   = nullptr;

public:
    ze_handle_t() = default;
    ze_handle_t(T handle, ze_dtor_t dtor) : m_handle(handle), m_dtor(dtor) {}
    ze_handle_t(T handle) : m_handle(handle) {}
    ze_handle_t(ze_dtor_t dtor) : m_dtor(dtor) {}

    virtual ~ze_handle_t()
    {
        if (m_dtor && m_handle) {
            LOG_DEBUG("Destroying handle: ", m_handle);
            m_dtor(m_handle);
            LOG_DEBUG("Handle destroyed: ", m_handle);
        }
    }

    T    handle() const { return m_handle; }
    bool is_valid() const { return m_handle != nullptr; }

    operator const T() const { return m_handle; }
    operator bool() const { return is_valid(); }
};

// The deleter on its own, for classes that cannot use ZE_HANDLE below because ze_handle_t
// is an indirect base of theirs - a mem-initializer may only name a direct base. Those
// assign m_dtor instead, e.g. m_dtor = ZE_DTOR(ze_event_handle_t, zeEventDestroy);
#define ZE_DTOR(type, dtor) [](type handle) { ZE_NO_RET(dtor(handle)); }

#define ZE_HANDLE(type, dtor) ze_handle_t<type>(ZE_DTOR(type, dtor))
