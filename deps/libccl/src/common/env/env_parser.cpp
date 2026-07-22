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

#include "env_parser.hpp"

extern char** environ;

namespace ccl {

// static file functions

static bool is_run_with_mpi() {
    return (getenv("MPI_LOCALRANKID") || getenv("MPI_LOCALNRANKS") || getenv("PMI_RANK") ||
            getenv("PMI_SIZE") || getenv("PMIX_RANK"))
               ? true
               : false;
}

// env_parser class implementation

env_parser::env_parser()
        : // skip variables not parsed_internally
          unused_check_skip({ "CCL_ROOT", // set by vars.sh
                              "CCL_CONFIGURATION", // set by vars.sh
                              "CCL_CONFIGURATION_PATH", // set by vars.sh
                              "CCL_PLATFORM_INFO_HIDE", // used to hide platform info in benchmarks
                              CCL_WORKER_OFFLOAD,
                              CCL_WORKER_AFFINITY,
                              CCL_WORKER_MEM_AFFINITY }) {}

void env_parser::env_2_atl_transport(
    std::map<ccl_atl_transport, std::string> const& atl_transport_names,
    ccl_atl_transport& atl_transport) {
#ifdef CCL_ENABLE_MPI
    if (!getenv(CCL_ATL_TRANSPORT) && !is_run_with_mpi()) {
        LOG_WARN("did not find MPI-launcher specific variables, switch to ATL/OFI, "
                 "to force enable ATL/MPI set CCL_ATL_TRANSPORT=mpi");
        unused_check_skip.insert(CCL_ATL_TRANSPORT);
        atl_transport = ccl_atl_ofi;
    }
    else
#endif // CCL_ENABLE_MPI
        env_2_enum(CCL_ATL_TRANSPORT, atl_transport_names, atl_transport);
}

void env_parser::warn_about_unused_var() const {
    for (char** s = environ; *s; ++s) {
        auto const env_key_and_value = std::string(*s);
        if (env_key_and_value.substr(0, 4) == "CCL_" &&
            unused_check_skip.count(env_key_and_value.substr(0, env_key_and_value.find('='))) == 0) {
            LOG_DEBUG(
                env_key_and_value,
                " is unknown to and unused by oneCCL code but is present"
                " in the environment, check if it is not mistyped.");
        }
    }
}

} // namespace ccl
