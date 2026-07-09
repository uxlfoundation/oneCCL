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

#include "atl/atl_def.h"

std::map<atl_mnic_t, std::string> mnic_type_names = { std::make_pair(ATL_MNIC_NONE, "none"),
                                                      std::make_pair(ATL_MNIC_LOCAL, "local"),
                                                      std::make_pair(ATL_MNIC_GLOBAL, "global") };

std::map<atl_mnic_offset_t, std::string> mnic_offset_names = {
    std::make_pair(ATL_MNIC_OFFSET_NONE, "none"),
    std::make_pair(ATL_MNIC_OFFSET_LOCAL_PROC_IDX, "local_proc_idx")
};

void atl_proc_coord_t::validate(int comm_rank, int comm_size) {
    CCL_THROW_IF_NOT(global_idx >= 0 && global_idx < global_count);
    CCL_THROW_IF_NOT(local_idx >= 0 && local_idx < local_count);
    CCL_THROW_IF_NOT(global_count >= 1);
    CCL_THROW_IF_NOT(local_count >= 1 && local_count <= global_count);

    if (comm_rank != -1 && comm_size != -1) {
        CCL_THROW_IF_NOT(comm_rank < comm_size);
    }
}

std::string to_string(atl_mnic_t type) {
    auto it = mnic_type_names.find(type);
    if (it != mnic_type_names.end()) {
        return it->second;
    }
    else {
        return "unknown";
    }
}

std::string to_string(atl_mnic_offset_t offset) {
    auto it = mnic_offset_names.find(offset);
    if (it != mnic_offset_names.end()) {
        return it->second;
    }
    else {
        return "unknown";
    }
}

std::string to_string(atl_proc_coord_t& coord) {
    std::stringstream ss;
    ss << "coord: global [ idx " << coord.global_idx << ", cnt " << coord.global_count
       << " ], local [ idx " << coord.local_idx << ", cnt " << coord.local_count << " ]";
    return ss.str();
}

std::string to_string(atl_attr_t& attr) {
    std::stringstream ss;
    ss << "atl attrs:\n{\n"
       << "  in: { "
       << "shm: " << attr.in.enable_shm << ", hmem: " << attr.in.enable_hmem
       << ", sync_coll: " << attr.in.enable_sync_coll << ", extra_ep: " << attr.in.enable_extra_ep
       << ", ep_count: " << attr.in.ep_count << ", mnic_type: " << to_string(attr.in.mnic_type)
       << ", mnic_count: " << attr.in.mnic_count
       << ", mnic_offset: " << to_string(attr.in.mnic_offset) << " }\n"
       << "  out: { "
       << "shm: " << attr.out.enable_shm << ", hmem: " << attr.out.enable_hmem
       << ", mnic_type: " << to_string(attr.out.mnic_type)
       << ", mnic_count: " << attr.out.mnic_count << ", tag_bits: " << attr.out.tag_bits
       << ", max_tag: " << attr.out.max_tag << " }\n}";
    return ss.str();
}

std::ostream& operator<<(std::ostream& str, const atl_req_t& req) {
    str << "req: { completed: " << req.is_completed << ", ptr: " << &req << " }";
    return str;
}
