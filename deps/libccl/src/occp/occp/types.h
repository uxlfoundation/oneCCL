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
#include <netinet/in.h> // for sockaddr_in, sockaddr_in6
#include <cstring>
#include <vector>

using rank_t = int; // rank in OneCCL
using comm_t = int; // OneCCL communicator

constexpr rank_t OCCL_INVALID_RANK = (rank_t)-1;
constexpr rank_t OCCP_SERVER = OCCL_INVALID_RANK;
constexpr comm_t OCCL_INVALID_COMM = (comm_t)-1;

#define HOSTNAME_MAX_LENGTH 256

#define packed_struct struct __attribute__((packed))

packed_struct rank_info_t {
    char hostname[HOSTNAME_MAX_LENGTH] = "UNKNOWN";

    rank_info_t(const char* host) {
        std::strncpy(this->hostname, host, HOSTNAME_MAX_LENGTH - 1);
        this->hostname[HOSTNAME_MAX_LENGTH - 1] = '\0'; // Null-terminate the string
    }

    rank_info_t() = default;
};

packed_struct ofi_endpoint_t {
    uint8_t data[64] = {}; // Example EFI configuration data, size is fixed
};

using ranks_t = std::vector<rank_t>;
using ofi_endpoints_t = std::vector<ofi_endpoint_t>;
using ranks_infos_t = std::vector<rank_info_t>;
