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

/**
 * oneCCL team — a regularly strided subset of a communicator's ranks,
 * described by {nranks, rank, stride}.
 */
#pragma once

#include <cstdint>

#include "oneapi/ccl/device_api/dev_comm.hpp"

namespace ccl {
namespace v1 {

struct team {
    int nranks;  /**< number of ranks in this team */
    int rank;    /**< this rank's index within the team */
    int stride;  /**< stride between consecutive team members in the world */
};

// Tags for compile-time dispatch
struct team_tag_world {};
struct team_tag_lsa {};

/** World team: all ranks, stride 1. */
inline team team_world(const dev_comm_data& dev_comm) {
    return { dev_comm.nranks, dev_comm.rank, 1 };
}

/** LSA team: ranks sharing local memory, stride 1. */
inline team team_lsa(const dev_comm_data& dev_comm) {
    return { dev_comm.lsa_size, dev_comm.lsa_rank, 1 };
}

} // namespace v1

using v1::team;
using v1::team_tag_world;
using v1::team_tag_lsa;
using v1::team_world;
using v1::team_lsa;

} // namespace ccl
