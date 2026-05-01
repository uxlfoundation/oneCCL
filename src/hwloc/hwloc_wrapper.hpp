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

#include "hwloc.h"

#include <vector>
#include <string>

#define CCL_HWLOC_INVALID_NUMA_NODE (-1)

struct ccl_numa_node {
    int os_idx;
    size_t mem_in_mb;
    int core_count;
    std::vector<int> cpus;
    int membind_support;

    ccl_numa_node();
    ccl_numa_node(int os_idx,
                  size_t mem_in_mb,
                  int core_count,
                  const std::vector<int>& cpus,
                  int membind_support);

    std::string to_string();
};

class ccl_hwloc_wrapper {
public:
    ccl_hwloc_wrapper();
    ccl_hwloc_wrapper(const ccl_hwloc_wrapper& other) = delete;
    ccl_hwloc_wrapper& operator=(const ccl_hwloc_wrapper& other) = delete;
    ~ccl_hwloc_wrapper();

    bool is_initialized();

    std::string to_string();

    bool is_dev_close_by_pci(int domain, int bus, int dev, int func);
    int get_numa_node_by_pci(int domain, int bus, int dev, int func);

    void membind_thread(int numa_node);
    int get_numa_node_by_cpu(int cpu);
    ccl_numa_node get_numa_node(int numa_node);

    void* alloc_memory(size_t alignment, size_t size, int numa_node_os_idx);
    void dealloc_memory(void* buffer);

    size_t get_numa_node_count();

private:
    bool is_valid_numa_node(int numa_node);
    bool check_membind(int numa_node);
    hwloc_obj_t get_first_non_io_obj_by_pci(int domain, int bus, int dev, int func);
    std::string obj_to_string(hwloc_obj_t obj);

    std::vector<ccl_numa_node> numa_nodes;

    bool membind_thread_supported;
    hwloc_cpuset_t bindset;
    hwloc_topology_t topology;

    // maps from aligned ptr (which was returned to the user by alloc_memory)
    //      to <misaligned ptr (returned by hwloc_alloc_membind), len>
    std::map<void*, std::pair<void*, size_t>> allocated_memory_map;
};
