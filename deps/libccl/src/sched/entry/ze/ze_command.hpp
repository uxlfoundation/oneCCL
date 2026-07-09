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

#include <cstdint>
#include <memory>
#include "sched/entry/ze/ze_kernel.hpp"

class ze_command_t {
public:
    virtual const char* name() const = 0;
    virtual void ze_call() = 0;
    virtual ~ze_command_t(){};
};

using ze_commands_t = std::vector<std::unique_ptr<ze_command_t>>;

#define ZE_APPEND_CALL_TO_ENTRY(base_entry, command, params...) \
    ze_command::create<command>(base_entry->get_ze_commands(), \
                                base_entry->get_sched()->get_ze_commands_bypass_flag(), \
                                params);

#define ZE_APPEND_CALL(command, params...) ZE_APPEND_CALL_TO_ENTRY(this, command, params);

namespace ze_command {

template <class CommandType, class... Arguments>
CommandType* create(const ze_commands_t& ze_commands, bool bypass_flag, Arguments&&... args) {
    LOG_DEBUG("creating: ", CommandType::class_name(), " command. bypass: ", bypass_flag);

    if (bypass_flag) {
        auto cmd = std::make_unique<CommandType>(std::forward<Arguments>(args)...);
        cmd->ze_call();
        return nullptr;
    }
    else {
        auto& commands = const_cast<ze_commands_t&>(ze_commands);
        commands.emplace_back(std::make_unique<CommandType>(std::forward<Arguments>(args)...));
        CommandType* ret = static_cast<CommandType*>(commands.back().get());
        return ret;
    }
}

} // namespace ze_command

class ze_cmd_memory_copy : public ze_command_t {
    ze_command_list_handle_t cmdlist{};
    void* dstptr{};
    void* srcptr{};
    size_t size{};
    ze_event_handle_t signal_event{};
    std::vector<ze_event_handle_t> wait_events{};

public:
    static constexpr const char* class_name() noexcept {
        return "ZECMD_MEMCPY";
    }
    const char* name() const override {
        return class_name();
    }

    ze_cmd_memory_copy() = delete;
    ze_cmd_memory_copy(ze_command_list_handle_t cmdlist,
                       void* dstptr,
                       void* srcptr,
                       size_t size,
                       ze_event_handle_t signal_event,
                       const std::vector<ze_event_handle_t>& wait_events)
            : cmdlist(cmdlist),
              dstptr(dstptr),
              srcptr(srcptr),
              size(size),
              signal_event(signal_event),
              wait_events(wait_events) {}

    void ze_call() override;
};

class ze_cmd_launch_kernel : public ze_command_t {
    ze_command_list_handle_t cmdlist{};
    ccl::ze::ze_kernel kernel;
    ze_event_handle_t signal_event{};
    std::vector<ze_event_handle_t> wait_events{};

public:
    static constexpr const char* class_name() noexcept {
        return "ZECMD_LAUNCH_KERNEL";
    }
    const char* name() const override {
        return class_name();
    }

    ze_cmd_launch_kernel() = delete;
    ze_cmd_launch_kernel(ze_command_list_handle_t cmdlist,
                         ccl::ze::ze_kernel kernel,
                         ze_event_handle_t signal_event,
                         const std::vector<ze_event_handle_t>& wait_events)
            : cmdlist(cmdlist),
              kernel(std::move(kernel)),
              signal_event(signal_event),
              wait_events(wait_events) {}

    void ze_call() override;
};

class ze_cmd_barrier : public ze_command_t {
    ze_command_list_handle_t cmdlist{};
    ze_event_handle_t signal_event{};
    std::vector<ze_event_handle_t> wait_events{};

public:
    static constexpr const char* class_name() noexcept {
        return "ZECMD_BARRIER";
    }
    const char* name() const override {
        return class_name();
    }

    ze_cmd_barrier() = delete;
    ze_cmd_barrier(ze_command_list_handle_t cmdlist,
                   ze_event_handle_t signal_event,
                   const std::vector<ze_event_handle_t>& wait_events)
            : cmdlist(cmdlist),
              signal_event(signal_event),
              wait_events(wait_events) {}

    void ze_call() override;
};

class ze_cmd_mem_range_barrier : public ze_command_t {
    ze_command_list_handle_t cmdlist{};

    std::vector<size_t> range_sizes;
    std::vector<const void*> ranges;

    ze_event_handle_t signal_event{};
    std::vector<ze_event_handle_t> wait_events{};

public:
    static constexpr const char* class_name() noexcept {
        return "ZECMD_MEMBARRIER";
    }
    const char* name() const override {
        return class_name();
    }

    ze_cmd_mem_range_barrier() = delete;
    ze_cmd_mem_range_barrier(ze_command_list_handle_t cmdlist,
                             const std::vector<size_t>& range_sizes,
                             const std::vector<const void*>& ranges,
                             ze_event_handle_t signal_event,
                             const std::vector<ze_event_handle_t>& wait_events)
            : cmdlist(cmdlist),
              range_sizes(range_sizes),
              ranges(ranges),
              signal_event(signal_event),
              wait_events(wait_events) {
        CCL_THROW_IF_NOT(range_sizes.size() == ranges.size() && !range_sizes.empty());
    }

    void ze_call() override;
};

class ze_cmd_wait_on_events : public ze_command_t {
    ze_command_list_handle_t cmdlist{};
    std::vector<ze_event_handle_t> wait_events{};

public:
    static constexpr const char* class_name() noexcept {
        return "ZECMD_WAITONEVENTS";
    }
    const char* name() const override {
        return class_name();
    }

    ze_cmd_wait_on_events() = delete;
    ze_cmd_wait_on_events(ze_command_list_handle_t cmdlist,
                          const std::vector<ze_event_handle_t>& wait_events)
            : cmdlist(cmdlist),
              wait_events(wait_events) {}

    void ze_call() override;
};

class ze_cmd_signal_event : public ze_command_t {
    ze_command_list_handle_t cmdlist{};
    ze_event_handle_t signal_event{};

public:
    static constexpr const char* class_name() noexcept {
        return "ZECMD_SIGNALEVENT";
    }
    const char* name() const override {
        return class_name();
    }

    ze_cmd_signal_event() = delete;
    ze_cmd_signal_event(ze_command_list_handle_t cmdlist, ze_event_handle_t signal_event)
            : cmdlist(cmdlist),
              signal_event(signal_event) {}

    void ze_call() override;
};

class ze_cmd_timestamp : public ze_command_t {
    ze_command_list_handle_t cmdlist{};
    uint64_t* timestamp_ptr{};
    ze_event_handle_t signal_event{};
    std::vector<ze_event_handle_t> wait_events;

public:
    static constexpr const char* class_name() noexcept {
        return "ZECMD_TIMESTAMP";
    }
    const char* name() const override {
        return class_name();
    }

    ze_cmd_timestamp() = delete;
    ze_cmd_timestamp(ze_command_list_handle_t cmdlist,
                     uint64_t* timestamp_ptr,
                     ze_event_handle_t signal_event,
                     const std::vector<ze_event_handle_t>& wait_events)
            : cmdlist(cmdlist),
              timestamp_ptr(timestamp_ptr),
              signal_event(signal_event),
              wait_events(wait_events) {}

    void ze_call() override;
};
