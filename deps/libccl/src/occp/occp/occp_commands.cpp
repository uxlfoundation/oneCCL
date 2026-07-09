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

#include "occp_commands.h"
#include "occp.h"

const char* cmd2str(cmdid_t id) {
    switch (id) {
        case OCCP_RANK_DATA: return "OCCP_RANK_DATA";
        case OCCP_SYNC: return "OCCP_SYNC";
        case OCCP_OFI_CONFIG: return "OCCP_OFI_CONFIG";
        default: return "you forgot to add it to cmd2str";
    }
}
