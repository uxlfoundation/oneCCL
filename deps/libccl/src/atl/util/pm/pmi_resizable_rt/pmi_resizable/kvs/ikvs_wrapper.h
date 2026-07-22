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

#include <unistd.h>
#include "util/pm/pmi_resizable_rt/pmi_resizable/def.h"

class ikvs_wrapper {
public:
    virtual ~ikvs_wrapper() noexcept(false){};

    virtual kvs_status_t kvs_set_value(const std::string& kvs_name,
                                       const std::string& kvs_key,
                                       const std::string& kvs_val) = 0;

    virtual kvs_status_t kvs_remove_name_key(const std::string& kvs_name,
                                             const std::string& kvs_key) = 0;

    virtual kvs_status_t kvs_get_value_by_name_key(const std::string& kvs_name,
                                                   const std::string& kvs_key,
                                                   std::string& kvs_val) = 0;

    virtual kvs_status_t kvs_init(const char* main_addr) = 0;

    virtual kvs_status_t kvs_main_server_address_reserve(char* main_addr) = 0;

    virtual kvs_status_t kvs_get_count_names(const std::string& kvs_name, size_t& count_names) = 0;

    virtual kvs_status_t kvs_finalize() = 0;

    virtual kvs_status_t kvs_get_keys_values_by_name(const std::string& kvs_name,
                                                     std::vector<std::string>& kvs_keys,
                                                     std::vector<std::string>& kvs_values,
                                                     size_t& count) = 0;

    virtual kvs_status_t kvs_get_replica_size(size_t& replica_size) = 0;
};
