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

// Identifies a collective at compile time. Each *_strategy_impl exposes
// class_id() so the benchmark harness can dispatch on an enum instead of
// comparing class_name() strings. class_name() is kept for the CLI-driven
// collective selection and for printing.
enum class coll_type {
    allgather,
    allgatherv,
    allreduce,
    alltoall,
    alltoallv,
    bcast,
    broadcast,
    reduce,
    reduce_scatter,
    scatter,
};
