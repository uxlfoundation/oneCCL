/*
 Copyright 2016-2020 Intel Corporation
 
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

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdint>
#include <system_error>

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

namespace ccl {
namespace v1 {}
} // namespace ccl

struct registered_ptr_desc {
    void *ptr;
    size_t size;
};

class ccl_comm;

// owned by ccl_comm
class alignas(CACHELINE_SIZE) ccl_window {
public:
    ccl_window(ccl_comm *comm) : comm(comm) {}

    bool is_registered(const void *ptr, size_t size, size_t &offset);

    std::vector<void *> get_ptrs(size_t offset);

    int register_buf(void *ptr, size_t size);

    int deregister_buf();

private:
    ccl_comm *comm;

    std::vector<registered_ptr_desc> registered_ptrs;
};
