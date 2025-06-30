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
#include "common/global/global.hpp"
#include "hwloc/hwloc_wrapper.hpp"

ccl_numa_node::ccl_numa_node()
        : os_idx(CCL_UNDEFINED_NUMA_NODE),
          mem_in_mb(0),
          core_count(0),
          membind_support(0) {}

ccl_numa_node::ccl_numa_node(int os_idx,
                             size_t mem_in_mb,
                             int core_count,
                             const std::vector<int>& cpus,
                             int membind_support)
        : os_idx(os_idx),
          mem_in_mb(mem_in_mb),
          core_count(core_count),
          cpus(cpus),
          membind_support(membind_support) {}

std::string ccl_numa_node::to_string() {
    std::stringstream ss;

    ss << "{"
       << "os_idx: " << os_idx << ", memory: " << mem_in_mb << " MB"
       << ", cores: " << core_count << ", cpus: " << cpus.size() << ", membind: " << membind_support
       << "}";

    return ss.str();
}

ccl_hwloc_wrapper::ccl_hwloc_wrapper()
        : membind_thread_supported(false),
          bindset(nullptr),
          topology(nullptr) {
    /* mandatory checks */

    if (hwloc_topology_init(&topology) < 0) {
        LOG_WARN("hwloc_topology_init failed (", strerror(errno), ")");
        return;
    }

    hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_ALL);

    if (hwloc_topology_load(topology) < 0) {
        LOG_WARN("hwloc_topology_load failed (", strerror(errno), ")");
        return;
    }

    hwloc_obj_t root_obj = hwloc_get_root_obj(topology);
    LOG_DEBUG("hwloc root object: ", obj_to_string(root_obj));

    bindset = hwloc_bitmap_alloc();
    if (hwloc_get_proc_cpubind(topology, getpid(), bindset, HWLOC_CPUBIND_PROCESS) < 0) {
        LOG_WARN("hwloc_get_proc_cpubind failed (", strerror(errno), ")");
        return;
    }

    CCL_THROW_IF_NOT(topology && bindset);

    /* optional checks */

    const struct hwloc_topology_support* topo_support = hwloc_topology_get_support(topology);
    membind_thread_supported = topo_support->membind->set_thisthread_membind;
    if (!membind_thread_supported) {
        LOG_WARN("no support for memory binding of current thread");
    }

    hwloc_obj_t numa_node = nullptr;
    while ((numa_node = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, numa_node)) !=
           nullptr) {
        int os_idx = numa_node->os_index;
        int mem_in_mb =
            (numa_node->attr) ? numa_node->attr->numanode.local_memory / (1024 * 1024) : 0;
        int core_count =
            hwloc_get_nbobjs_inside_cpuset_by_type(topology, numa_node->cpuset, HWLOC_OBJ_CORE);
        std::vector<int> cpus;
        for (int core_idx = 0; core_idx < core_count; core_idx++) {
            hwloc_obj_t core_obj = hwloc_get_obj_inside_cpuset_by_type(
                topology, numa_node->cpuset, HWLOC_OBJ_CORE, core_idx);
            int cpus_per_core =
                hwloc_get_nbobjs_inside_cpuset_by_type(topology, core_obj->cpuset, HWLOC_OBJ_PU);
            for (int cpu_idx = 0; cpu_idx < cpus_per_core; cpu_idx++) {
                hwloc_obj_t cpu_obj = hwloc_get_obj_inside_cpuset_by_type(
                    topology, core_obj->cpuset, HWLOC_OBJ_PU, cpu_idx);
                cpus.push_back(cpu_obj->os_index);
                //LOG_DEBUG("hwloc L#", numa_node->logical_index, " P#", os_idx, /*" subtype (", numa_node->subtype,")",*/ " mb ", mem_in_mb, " core_count ", core_count, " core ", core_idx,".",cpu_idx);
            }

            struct hwloc_location initiator;
            initiator.type = HWLOC_LOCATION_TYPE_CPUSET;
            initiator.location.cpuset = core_obj->cpuset;

            hwloc_uint64_t latency, bandwidth;
            int err = hwloc_memattr_get_value(
                topology, HWLOC_MEMATTR_ID_BANDWIDTH, numa_node, &initiator, 0, &bandwidth);
            if (err < 0) {
                bandwidth = 0;
            }
            err = hwloc_memattr_get_value(
                topology, HWLOC_MEMATTR_ID_LATENCY, numa_node, &initiator, 0, &latency);
            if (err < 0) {
                latency = 0;
            }
            //LOG_DEBUG(" bandwidth is ", bandwidth, " MiB/s   latency is ", latency, " ns");
        }
        numa_nodes.push_back(
            ccl_numa_node(os_idx, mem_in_mb, core_count, cpus, check_membind(os_idx)));
    }
}

ccl_hwloc_wrapper::~ccl_hwloc_wrapper() {
    hwloc_bitmap_free(bindset);
    hwloc_topology_destroy(topology);
}

bool ccl_hwloc_wrapper::is_initialized() {
    return (topology && bindset) ? true : false;
}

std::string ccl_hwloc_wrapper::to_string() {
    std::stringstream ss;
    bool initialized = is_initialized();
    ss << "hwloc initialized: " << initialized << "\n";
    if (initialized) {
        ss << "{\n";
        ss << "  membind_thread_supported: " << membind_thread_supported << "\n";
        for (auto& node : numa_nodes) {
            ss << "  numa: " << node.to_string() << "\n";
        }
        ss << "}";
    }
    return ss.str();
}

bool ccl_hwloc_wrapper::is_dev_close_by_pci(int domain, int bus, int dev, int func) {
    bool is_close = false;

    if (!is_initialized()) {
        LOG_WARN("hwloc is not initialized, skip checking of locality for device: [",
                 domain,
                 ":",
                 bus,
                 ":",
                 dev,
                 ":",
                 func,
                 "]");
        return is_close;
    }

    hwloc_obj_t first_non_io = get_first_non_io_obj_by_pci(domain, bus, dev, func);
    CCL_THROW_IF_NOT(first_non_io);

    LOG_DEBUG("first_non_io object: ", obj_to_string(first_non_io));
    LOG_DEBUG("pci info: [", domain, ":", bus, ":", dev, ":", func, "]");

    /* determine if PCI device is "close" to process by checking if process's affinity is included
     * in PCI device's affinity or if PCI device's affinity is included in process's affinity */
    is_close = (hwloc_bitmap_isincluded(bindset, first_non_io->cpuset) ||
                hwloc_bitmap_isincluded(first_non_io->cpuset, bindset));

    return is_close;
}

void ccl_hwloc_wrapper::membind_thread(int numa_node) {
    if (!is_initialized()) {
        LOG_WARN("hwloc is not initialized, skip thread membind for NUMA node ", numa_node);
        return;
    }

    if (!membind_thread_supported) {
        LOG_WARN(
            "no support for memory binding of current thread, skip thread membind for NUMA node ",
            numa_node);
        return;
    }

    if (!is_valid_numa_node(numa_node)) {
        LOG_WARN("invalid NUMA node ",
                 numa_node,
                 ", NUMA node count ",
                 get_numa_node_count(),
                 ", skip thread membind");
        return;
    }

    if (!get_numa_node(numa_node).membind_support) {
        LOG_WARN("no membind support for NUMA node ", numa_node, ", skip thread membind");
        return;
    }

    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    hwloc_bitmap_only(nodeset, unsigned(numa_node));
    CCL_THROW_IF_NOT(hwloc_bitmap_isset(nodeset, numa_node) == 1, "hwloc_bitmap_isset failed");

    if (hwloc_set_membind(topology,
                          nodeset,
                          HWLOC_MEMBIND_BIND,
                          HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT | HWLOC_MEMBIND_BYNODESET) <
        0) {
        LOG_WARN("failed to bind thread to NUMA node ", numa_node, " (", strerror(errno), ")");
    }
    else {
        LOG_DEBUG("bound thread to NUMA node ", numa_node);
    }

    hwloc_bitmap_free(nodeset);
}

int ccl_hwloc_wrapper::get_numa_node_by_cpu(int cpu) {
    if (!is_initialized()) {
        LOG_WARN("hwloc is not initialized, can't get numa NUMA for CPU ", cpu);
        return CCL_UNDEFINED_NUMA_NODE;
    }

    if (cpu == CCL_UNDEFINED_CPU_ID) {
        return CCL_UNDEFINED_NUMA_NODE;
    }

    for (auto& node : numa_nodes) {
        for (auto cpu_idx : node.cpus) {
            if (cpu_idx == cpu) {
                return node.os_idx;
            }
        }
    }

    return CCL_UNDEFINED_NUMA_NODE;
}

ccl_numa_node ccl_hwloc_wrapper::get_numa_node(int numa_node) {
    if (!is_initialized()) {
        LOG_WARN("hwloc is not initialized, can't get info for NUMA node ", numa_node);
        return {};
    }

    for (auto& node : numa_nodes) {
        if (node.os_idx == numa_node) {
            return node;
        }
    }

    // is_valid_numa_node() iterates through the numa_nodes vector. To avoid
    // iterating through the vector twice, we first check whether we find it
    // (see loop above) before calling is_valid_numa_node().
    if (!is_valid_numa_node(numa_node)) {
        LOG_WARN("invalid NUMA node ", numa_node, ", NUMA node count ", get_numa_node_count());
        return {};
    }

    // We should never reach this point.
    CCL_THROW("invalid NUMA node ",
              numa_node,
              ". (But this should've been caught by is_valid_numa_node)");
}

bool ccl_hwloc_wrapper::is_valid_numa_node(int numa_node) {
    if ((numa_node == CCL_UNDEFINED_NUMA_NODE) || (numa_node < 0)) {
        return false;
    }

    for (auto& node : numa_nodes) {
        if (node.os_idx == numa_node) {
            return true;
        }
    }

    return false;
}

// Note: the pointer returned is not a 'freeable' address.
//       Call ccl_hwloc_wrapper::dealloc_memory instead
void* ccl_hwloc_wrapper::alloc_memory(size_t alignment, size_t size, int numa_node_os_idx) {
    // TODO: How can we determine which numa_node_os_idx is HBM and which one is DDR?
    //       One idea would be to use bandwidth and latency as reported by hwloc (see above)

    CCL_THROW_IF_NOT(is_valid_numa_node(numa_node_os_idx), "Incorrect numa node requested");

    hwloc_obj_t numa_node_obj = hwloc_get_numanode_obj_by_os_index(topology, numa_node_os_idx);
    void* buffer = hwloc_alloc_membind(topology,
                                       size + alignment, // TODO: handle alignment better
                                       numa_node_obj->nodeset,
                                       HWLOC_MEMBIND_BIND,
                                       HWLOC_MEMBIND_STRICT | HWLOC_MEMBIND_BYNODESET);

    if (!buffer) {
        CCL_THROW("Failed to allocate memory on numa_node ", numa_node_os_idx);
    }

    void* ret = buffer;
    if (((uintptr_t)buffer) % alignment) {
        // TODO: handle alignment better
        size_t offset = alignment - (((uintptr_t)buffer) % alignment);
        ret = ((char*)buffer) + offset;
    }
    CCL_THROW_IF_NOT((((uintptr_t)ret) % alignment) == 0);

    // Add size and potential misalignement to allocated_memory_map for later freeing.
    allocated_memory_map[ret] = std::make_pair(buffer, size);

    return ret;
}

void ccl_hwloc_wrapper::dealloc_memory(void* buffer) {
    // TODO: do a better job at allocating with alignment.
    CCL_THROW_IF_NOT(buffer != nullptr, "We were asked to dealloc a nullptr");
    auto it = allocated_memory_map.find(buffer);
    CCL_THROW_IF_NOT(it != allocated_memory_map.end(),
                     "We were asked to dealloc memory that hasn't been allocated");
    if (hwloc_free(topology, it->second.first, it->second.second) < 0) {
        LOG_WARN("hwloc_free failed (", strerror(errno), ")");
    }
}

bool ccl_hwloc_wrapper::check_membind(int numa_node) {
    hwloc_obj_t numa_node_obj = hwloc_get_numanode_obj_by_os_index(topology, numa_node);
    size_t check_buf_len = 8192;
    void* buffer = hwloc_alloc_membind(topology,
                                       check_buf_len,
                                       numa_node_obj->nodeset,
                                       HWLOC_MEMBIND_BIND,
                                       HWLOC_MEMBIND_STRICT | HWLOC_MEMBIND_BYNODESET);

    if (!buffer) {
        return false;
    }

    bool membind_ok = true;

    hwloc_bitmap_t nodeset = hwloc_bitmap_alloc();
    hwloc_bitmap_zero(nodeset);
    hwloc_membind_policy_t policy = HWLOC_MEMBIND_DEFAULT;

    if (hwloc_get_area_membind(
            topology, buffer, check_buf_len, nodeset, &policy, HWLOC_MEMBIND_BYNODESET) < 0) {
        LOG_WARN("NUMA node ", numa_node, ", failed to get nodeset and policy for buffer ", buffer);
        membind_ok = false;
    }

    if (policy != HWLOC_MEMBIND_BIND) {
        LOG_WARN("NUMA node ",
                 numa_node,
                 ", unxpected membind policy ",
                 policy,
                 ", expected ",
                 HWLOC_MEMBIND_BIND);
        membind_ok = false;
    }

    int i = 0, bind_count = 0;
    hwloc_bitmap_foreach_begin(i, nodeset) {
        hwloc_obj_t obj = hwloc_get_numanode_obj_by_os_index(topology, i);
        if (obj) {
            bind_count++;
        }
    }
    hwloc_bitmap_foreach_end();

    if (bind_count != 1) {
        LOG_WARN("buffer should be bound to single NUMA node but actual bind_count", bind_count);
        membind_ok = false;
    }

    if (!hwloc_bitmap_isset(nodeset, numa_node)) {
        LOG_WARN("nodeset doesn't have expected index ", numa_node);
        membind_ok = false;
    }

    if (hwloc_bitmap_first(nodeset) != numa_node) {
        LOG_WARN("nodeset has unexpected first index ",
                 hwloc_bitmap_first(nodeset),
                 ", expected ",
                 numa_node);
        membind_ok = false;
    }

    hwloc_bitmap_free(nodeset);
    hwloc_free(topology, buffer, check_buf_len);

    return membind_ok;
}

size_t ccl_hwloc_wrapper::get_numa_node_count() {
    return numa_nodes.size();
}

hwloc_obj_t ccl_hwloc_wrapper::get_first_non_io_obj_by_pci(int domain, int bus, int dev, int func) {
    hwloc_obj_t io_device = hwloc_get_pcidev_by_busid(topology, domain, bus, dev, func);
    CCL_THROW_IF_NOT(io_device,
                     "failed to get PCI device with domain %d, bus %d, dev %d, func %d",
                     domain,
                     bus,
                     dev,
                     func);

    hwloc_obj_t first_non_io = hwloc_get_non_io_ancestor_obj(topology, io_device);
    CCL_THROW_IF_NOT(first_non_io, "failed to get ancestor of PCI device");
    return first_non_io;
}

std::string ccl_hwloc_wrapper::obj_to_string(hwloc_obj_t obj) {
    std::stringstream ss;
    const size_t obj_str_len = 4096;
    char str[obj_str_len];

    hwloc_obj_type_snprintf(str, obj_str_len, obj, 1);
    ss << "type: " << str << "\n";
    hwloc_obj_attr_snprintf(str, obj_str_len, obj, " :: ", 1);
    ss << "attr: " << str << "\n";
    hwloc_bitmap_taskset_snprintf(str, obj_str_len, obj->cpuset);
    ss << "cpuset: " << str << "\n";
    hwloc_bitmap_taskset_snprintf(str, obj_str_len, obj->nodeset);
    ss << "nodeset: " << str << "\n";

    return ss.str();
}
