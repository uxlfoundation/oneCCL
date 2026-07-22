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
#include "common/env/env.hpp"

namespace ccl {

class env_parser {
public:
    env_parser();

    template <class T>
    void env_2_type(const char* env_name, T& val) {
        const char* env_val = getenv(env_name);
        if (env_val) {
            T new_val;
            std::stringstream ss;
            ss << env_val;
            ss >> new_val;
            set_value(env_name, val, std::move(new_val));
        }
    }

    void env_2_type(const char* env_name, bool& val) {
        const char* env_val = getenv(env_name);
        if (env_val) {
            std::string env_val_str(env_val);
            bool new_val = 0;
            if (env_val_str == "1") {
                new_val = true;
            }
            else if (env_val_str == "0") {
                new_val = false;
            }
            else {
                CCL_THROW(env_name, ": unexpected value: ", env_val_str, ", expected values: 0, 1");
            }
            set_value(env_name, val, new_val);
        }
    }

    template <class T>
    void env_2_enum(const char* env_name, const std::map<T, std::string>& values, T& val) {
        const char* env_val = getenv(env_name);
        if (env_val) {
            T const new_val = enum_by_str(env_name, values, env_val);
            set_enum(env_name, val, new_val, values);
        }
    }

    template <class T>
    void env_2_topo(const char* env_name, const std::map<T, std::string>& values, T& val) {
        char* env_to_parse = getenv(env_name);
        if (env_to_parse) {
            std::string const env_str(env_to_parse);
            if (env_str.find(std::string(topo_manager::card_domain_name)) != std::string::npos &&
                env_str.find(std::string(topo_manager::plane_domain_name)) != std::string::npos) {
                set_enum(env_name, val, topo_color_mode::env, values);
            }
            else {
                env_2_enum(env_name, values, val);
            }
        }
    }

    void env_2_atl_transport(std::map<ccl_atl_transport, std::string> const& atl_transport_names,
                             ccl_atl_transport& atl_transport);

    void warn_about_unused_var() const;

private:
    template <class T>
    static T enum_by_str(const std::string& env_name,
                         const std::map<T, std::string>& enum2str,
                         std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        for (const auto& pair : enum2str) {
            if (!str.compare(pair.second)) {
                // string found, return coresponding enum
                return pair.first;
            }
        }
        // string not found, perform error handling:
        // aggregate a list of expected strings and throw it

        std::vector<std::string> env_names;
        std::transform(enum2str.begin(),
                       enum2str.end(),
                       std::back_inserter(env_names),
                       [](const typename std::map<T, std::string>::value_type& pair) {
                           return pair.second;
                       });

        std::string expected_values;
        for (size_t idx = 0; idx < env_names.size(); idx++) {
            expected_values += env_names[idx];
            if (idx != env_names.size() - 1) {
                expected_values += ", ";
            }
        }

        CCL_THROW(env_name, ": unexpected value: ", str, ", expected values: ", expected_values);
    }

    // logs warning when value changes
    template <class T>
    void set_value(const std::string& env_name, T& value, T const& new_value) {
        unused_check_skip.insert(env_name);
        if (value != new_value) { // warn if reset to a different value
            LOG_WARN_ROOT("value of ", env_name, " changed to be ", new_value, " (default:", value, ")");
        }
        value = new_value;
    }

    // logs warning when value changes
    template <class T>
    void set_enum(const std::string& env_name,
                  T& value,
                  T const& new_value,
                  std::map<T, std::string> const& value2name) {
        unused_check_skip.insert(env_name);
        if (value != new_value) {
            LOG_WARN_ROOT("value of ",
                          env_name,
                          " changed to be ",
                          value2name.at(new_value),
                          " (default:",
                          value2name.at(value),
                          ")");
        }
        value = new_value;
    }

    // CCL_* environment variables that should not be reported as "unused":
    // Contains two types of vars:
    //
    //  1. Created and declared by the user, but not parsed or used by oneCCL
    //  2. Already parsed (used) by oneCCL
    //
    //  At the end of parsing, oneCCL checks if all variables belong
    //  to either of the categories above, i.e. are added to this set
    //
    //  All CCL_* variables that are not a part of this set at the end
    //  of parsing are reported to the user as incorrect.
    std::set<std::string> unused_check_skip;
};

} // namespace ccl
