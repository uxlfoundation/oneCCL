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

#include "oneapi/ccl/v2/types.h"

#include <cstdint>
#include <cstdlib>

#ifndef ONECCL_PLUGIN_H
#define ONECCL_PLUGIN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum onecclPluginCall {
    ONECCL_PLUGIN_INIT = 1,
    ONECCL_PLUGIN_INIT_ID,
    ONECCL_PLUGIN_INIT_COMM,
    ONECCL_PLUGIN_INIT_DEVICE,
    ONECCL_PLUGIN_INIT_KVS,
    ONECCL_PLUGIN_GROUP_START,
    ONECCL_PLUGIN_GROUP_END,
    ONECCL_PLUGIN_PLATFORM_SCORE,
} onecclPluginCall_t;

// Typedef for the only plugin entrypoint
typedef void *(*onecclPluginEntrypoint_t)(onecclPluginCall_t call_type);

// Typedefs for plugin calls
typedef onecclResult_t (*onecclPluginCallInit_t)();
typedef onecclResult_t (*onecclPluginCallInitId_t)(onecclUniqueId *uniqueId);
typedef onecclResult_t (*onecclPluginCallCommInit_t)(
    onecclComm_t *comm, size_t nranks, onecclUniqueId commId, int rank,
    const onecclConfig_t *config);
typedef onecclResult_t (*onecclPluginCallDeviceInit_t)(uint32_t index);
typedef onecclResult_t (*onecclPluginCallGroupStart_t)();
typedef onecclResult_t (*onecclPluginCallGroupEnd_t)();
typedef onecclResult_t (*onecclPluginCallPlatformScore_t)(int *score_ptr);

// Structure holding pointers to the plugin's implementation functions
typedef struct onecclPlugin {
    onecclPluginEntrypoint_t call;
} onecclPlugin_t;

// Internal function used to load plugins by other plugins
onecclResult_t _onecclLoadPlugin(onecclPlugin_t *plugin,
                                 onecclPluginType_t type);

#ifdef __cplusplus
}
#endif

#endif // ONECCL_PLUGIN_H
