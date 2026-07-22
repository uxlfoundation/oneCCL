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

#include "oneapi/ccl/config.h"

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>

namespace ccl {

typedef struct ofi_lib_ops {
    decltype(fi_dupinfo) *fi_dupinfo_ptr;
    decltype(fi_fabric) *fi_fabric_ptr;
    decltype(fi_freeinfo) *fi_freeinfo_ptr;
    decltype(fi_getinfo) *fi_getinfo_ptr;
    decltype(fi_strerror) *fi_strerror_ptr;
    decltype(fi_tostr) *fi_tostr_ptr;
} ofi_lib_ops_t;

static std::vector<std::string> ofi_fn_names = {
    "fi_dupinfo", "fi_fabric", "fi_freeinfo", "fi_getinfo", "fi_strerror", "fi_tostr",
};

extern ccl::ofi_lib_ops_t ofi_lib_ops;

#define fi_allocinfo() (fi_dupinfo)(NULL)
#define fi_dupinfo     ccl::ofi_lib_ops.fi_dupinfo_ptr
#define fi_fabric      ccl::ofi_lib_ops.fi_fabric_ptr
#define fi_freeinfo    ccl::ofi_lib_ops.fi_freeinfo_ptr
#define fi_getinfo     ccl::ofi_lib_ops.fi_getinfo_ptr
#define fi_strerror    ccl::ofi_lib_ops.fi_strerror_ptr
#define fi_tostr       ccl::ofi_lib_ops.fi_tostr_ptr

bool ofi_api_init();
void ofi_api_fini();
std::string get_ofi_lib_path();

} //namespace ccl
