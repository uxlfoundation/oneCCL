/*
 Copyright 2016-2025 Intel Corporation
 
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

#include <functional>
#include <optional>
#include <vector>

#include "oneapi/ccl.h"
#include "oneapi/ccl/v2/types.h"

// Enumeration for hook points
typedef enum onecclHookPoint {
    ONECCL_BEFORE_UNIQUE_ID,
    ONECCL_AFTER_UNIQUE_ID,
    ONECCL_BEFORE_DESTROY_COMM,
    ONECCL_AFTER_DESTROY_COMM,
    ONECCL_BEFORE_GET_RANK,
    ONECCL_AFTER_GET_RANK,
    ONECCL_BEFORE_GET_SIZE,
    ONECCL_AFTER_GET_SIZE,
    ONECCL_BEFORE_GET_DEVICE,
    ONECCL_AFTER_GET_DEVICE,
    ONECCL_BEFORE_SET_DEVICE,
    ONECCL_AFTER_SET_DEVICE,
    ONECCL_BEFORE_CREATE_PRE_MUL_SUM,
    ONECCL_AFTER_CREATE_PRE_MUL_SUM,
    ONECCL_BEFORE_REDUCTION_DESTROY,
    ONECCL_AFTER_REDUCTION_DESTROY,
    ONECCL_BEFORE_ALLREDUCE,
    ONECCL_AFTER_ALLREDUCE,
    ONECCL_BEFORE_INIT_COMM,
    ONECCL_AFTER_INIT_COMM,
    ONECCL_BEFORE_PLUGIN_INIT,
    ONECCL_AFTER_PLUGIN_INIT,
    ONECCL_BEFORE_STREAM_INIT,
    ONECCL_AFTER_STREAM_INIT,
    ONECCL_BEFORE_STREAM_DESTROY,
    ONECCL_AFTER_STREAM_DESTROY,
    ONECCL_BEFORE_STREAM_SYNC,
    ONECCL_AFTER_STREAM_SYNC,
    ONECCL_BEFORE_APPEND_CALLBACK,
    ONECCL_AFTER_APPEND_CALLBACK,
    ONECCL_BEFORE_EVENT_WAIT,
    ONECCL_AFTER_EVENT_WAIT,
    ONECCL_BEFORE_EVENT_DESTROY,
    ONECCL_AFTER_EVENT_DESTROY,
    ONECCL_BEFORE_EVENT_RECORD,
    ONECCL_AFTER_EVENT_RECORD,
    ONECCL_BEFORE_WAIT_EVENT,
    ONECCL_AFTER_WAIT_EVENT,
    ONECCL_BEFORE_XPU_INIT,
    ONECCL_AFTER_XPU_INIT,
    ONECCL_BEFORE_CPU_INIT,
    ONECCL_AFTER_CPU_INIT,
    ONECCL_BEFORE_ALLGATHER,
    ONECCL_AFTER_ALLGATHER,
    ONECCL_BEFORE_BROADCAST,
    ONECCL_AFTER_BROADCAST,
    ONECCL_BEFORE_REDUCE,
    ONECCL_AFTER_REDUCE,
    ONECCL_BEFORE_REDUCE_SCATTER,
    ONECCL_AFTER_REDUCE_SCATTER,
    ONECCL_BEFORE_SEND,
    ONECCL_AFTER_SEND,
    ONECCL_BEFORE_RECV,
    ONECCL_AFTER_RECV,
    ONECCL_BEFORE_FINALIZE,
    ONECCL_AFTER_FINALIZE,
    ONECCL_BEFORE_ABORT,
    ONECCL_AFTER_ABORT,
    ONECCL_NEXT_API_CALL
} onecclHookPoint_t;

// Enumeration for hook lifetime
typedef enum HookLifetime {
    ONECCL_PERSISTENT,
    ONECCL_ONE_TIME,
} onecclHookLifetime_t;

// Callback type with optional return result
using CallbackType =
    std::function<std::optional<onecclResult_t>(onecclHookPoint_t)>;

// Structure to hold a hook's metadata
struct onecclHook {
    onecclHookPoint_t hookPoint;
    onecclHookLifetime_t lifetime;
    CallbackType callback;
};

// Store pairs of HookPoint and CallbackType
extern CCL_C_API std::vector<onecclHook> null_callbacks;

// Macro to process hooks
#define PROCESS_HOOKS(hook_point)                                              \
    for (auto it = null_callbacks.begin(); it != null_callbacks.end();) {      \
        if (it->hookPoint == (hook_point) ||                                   \
            (hook_point) == ONECCL_NEXT_API_CALL) {                            \
            auto callback = it->callback;                                      \
            if (it->lifetime == ONECCL_ONE_TIME) {                             \
                it = null_callbacks.erase(it);                                 \
            }                                                                  \
            auto result = callback(hook_point);                                \
            if (result.has_value()) {                                          \
                return result.value();                                         \
            }                                                                  \
        }                                                                      \
        ++it;                                                                  \
    }

// Macro to add hook
#define ADD_HOOK(hook_point, lifetime, callback_fn)                            \
    null_callbacks.emplace_back(onecclHook{hook_point, lifetime, callback_fn})
