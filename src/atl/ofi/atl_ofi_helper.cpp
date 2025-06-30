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
#include "atl_ofi_helper.hpp"
#include "common/api_wrapper/pmix_api_wrapper.hpp"

atl_ofi_global_data_t global_data;

std::string atl_ofi_get_short_nic_name(const struct fi_info* prov) {
    std::stringstream ss;
    ss << prov->domain_attr->name;
    return ss.str();
}

std::string atl_ofi_get_nic_name(const struct fi_info* prov) {
    std::stringstream ss;
    ss << prov->fabric_attr->prov_name << ":";
    // ss << prov->fabric_attr->name << ":";
    ss << atl_ofi_get_short_nic_name(prov);
    return ss.str();
}

const char* atl_ofi_link_state_str(enum fi_link_state state) {
    switch (state) {
        case FI_LINK_DOWN: return "down";
        case FI_LINK_UP: return "up";
        default: return "unknown";
    }
}

std::string atl_ofi_get_nic_info(const struct fi_info* prov) {
    std::stringstream ss;

    ss << "{ ";

    ss << "name " << atl_ofi_get_nic_name(prov);

    if (prov->nic && prov->nic->link_attr) {
        ss << ", state " << atl_ofi_link_state_str(prov->nic->link_attr->state);

        if (prov->nic->link_attr->mtu) {
            ss << ", mtu " << prov->nic->link_attr->mtu << " bytes";
        }

        if (prov->nic->link_attr->speed) {
            const float bits_to_gbytes_coef = 8.0 * 1000 * 1000 * 1000;
            ss << ", speed " << (float)prov->nic->link_attr->speed / bits_to_gbytes_coef << " GB/s";
        }

        if (prov->nic->link_attr->address) {
            ss << ", address " << prov->nic->link_attr->address;
        }

        if (prov->nic->link_attr->network_type) {
            ss << ", network_type " << prov->nic->link_attr->network_type;
        }
    }
    else {
        ss << ", no link attr";
    }

    ss << " }";

    return ss.str();
}

atl_ofi_prov_t* atl_ofi_get_prov(atl_ofi_ctx_t& ctx,
                                 const atl_proc_coord_t& coord,
                                 const atl_ep_t& ep,
                                 int peer_proc_idx,
                                 size_t msg_size) {
    size_t prov_idx;

    CCL_THROW_IF_NOT(
        ctx.prov_count <= ATL_OFI_MAX_PROV_COUNT, "unexpected prov_count ", ctx.prov_count);

    int has_shm = (ctx.prov_count == ctx.nw_prov_count + 1) ? 1 : 0;

    if (has_shm && (coord.global2local_map[peer_proc_idx] != -1) &&
        (msg_size <= ctx.provs[ctx.shm_prov_idx].max_msg_size)) {
        prov_idx = ctx.shm_prov_idx;
    }
    else {
        size_t nw_prov_offset = ep.idx % ctx.nw_prov_count;
        prov_idx = ctx.nw_prov_first_idx + nw_prov_offset;
    }

    LOG_DEBUG("select nic: ep_idx ",
              ep.idx,
              ", local_proc_idx ",
              coord.local_idx,
              ", prov_idx ",
              prov_idx,
              ", my_proc_idx ",
              coord.global_idx,
              ", peer_proc_idx ",
              peer_proc_idx,
              ", msg_size ",
              msg_size,
              ", has_shm ",
              has_shm);

    /* TODO: add segmentation logic */
    CCL_THROW_IF_NOT(msg_size <= ctx.provs[prov_idx].max_msg_size,
                     "msg_size (",
                     msg_size,
                     ") is greater than max_msg_size (",
                     ctx.provs[prov_idx].max_msg_size,
                     "), prov_idx ",
                     prov_idx);

    return &(ctx.provs[prov_idx]);
}

atl_status_t atl_ofi_get_local_proc_coord(atl_proc_coord_t& coord, std::shared_ptr<ipmi> pmi) {
    atl_status_t ret = ATL_STATUS_SUCCESS;
    int i;
    int local_idx = 0, local_count = 0;
    char* all_hostnames = nullptr;
    char my_hostname[ATL_MAX_HOSTNAME_LEN] = { 0 };
    size_t my_hostname_len = 0;
    int my_global_proc_idx = coord.global_idx;

    gethostname(my_hostname, ATL_MAX_HOSTNAME_LEN - 1);
    my_hostname_len = strlen(my_hostname);
    coord.hostname_hash = std::hash<std::string>{}(my_hostname);

    CCL_THROW_IF_NOT(my_hostname_len < ATL_MAX_HOSTNAME_LEN,
                     "unexpected my_hostname_len ",
                     my_hostname_len,
                     ", expected max ",
                     (size_t)(ATL_MAX_HOSTNAME_LEN));

    if (ATL_MAX_HOSTNAME_LEN - my_hostname_len <= 10) {
        LOG_WARN("hostname is quite long, len: ", my_hostname_len, ", name: ", my_hostname);
    }

    snprintf(my_hostname + my_hostname_len,
             ATL_MAX_HOSTNAME_LEN - my_hostname_len,
             "-%d",
             my_global_proc_idx);

    all_hostnames = (char*)calloc(1, coord.global_count * ATL_MAX_HOSTNAME_LEN);
    if (!all_hostnames) {
        LOG_ERROR("Failed to allocate memory for all_hostnames");
        goto fn_err;
    }

    if (ccl::global_data::env().kvs_init_mode == ccl::kvs_mode::pmix_ofi &&
        ccl::global_data::env().enable_init_hostname_sharing) {
        LOG_DEBUG("using PMIx for hostname exchange");

        // PMIx-based sharing
        pmix_value_t value;
        PMIX_VALUE_CONSTRUCT(&value);
        ccl_pmix::value_load(value, my_hostname, PMIX_STRING);

        int key_put = my_global_proc_idx * ATL_OFI_PMI_PROC_MULTIPLIER;
        std::string key_put_str = std::string(ATL_OFI_HOSTNAME_PM_KEY) + std::to_string(key_put);

        ccl_pmix::put(key_put_str, value);

        ccl_pmix::commit();
        ccl_pmix::fence(ccl::global_proc.nspace, pmi->get_rank(), pmi->get_size());

        for (int i = 0; i < coord.global_count; i++) {
            int key_get = i * ATL_OFI_PMI_PROC_MULTIPLIER;
            std::string key_get_str =
                std::string(ATL_OFI_HOSTNAME_PM_KEY) + std::to_string(key_get);
            pmix_value_t* ret_value = nullptr;
            ccl_pmix::get(ccl::global_proc.nspace, i, key_get_str, &ret_value);
            strncpy(all_hostnames + i * ATL_MAX_HOSTNAME_LEN,
                    ret_value->data.string,
                    ATL_MAX_HOSTNAME_LEN - 1);
            PMIX_VALUE_RELEASE(ret_value);
        }
    }
    else if (ccl::global_data::env().enable_init_hostname_sharing) {
        LOG_DEBUG("using PMI for hostname exchange");
        // PMI-based sharing
        ret = pmi->pmrt_kvs_put((char*)ATL_OFI_HOSTNAME_PM_KEY,
                                my_global_proc_idx * ATL_OFI_PMI_PROC_MULTIPLIER,
                                my_hostname,
                                ATL_MAX_HOSTNAME_LEN);

        if (ret) {
            LOG_ERROR("pmrt_kvs_put: ret: ", ret);
            goto fn_err;
        }

        ATL_CHECK_STATUS(pmi->pmrt_barrier(), "barrier failed");

        for (i = 0; i < coord.global_count; i++) {
            ret = pmi->pmrt_kvs_get((char*)ATL_OFI_HOSTNAME_PM_KEY,
                                    i * ATL_OFI_PMI_PROC_MULTIPLIER,
                                    all_hostnames + i * ATL_MAX_HOSTNAME_LEN,
                                    ATL_MAX_HOSTNAME_LEN);
            if (ret) {
                LOG_ERROR("pmrt_kvs_get: ret: ", ret);
                goto fn_err;
            }
        }
    }
    else {
        LOG_DEBUG("disable hostname exchange to get local coords");
    }

    // Initialize the map with all local ids invalid
    coord.global2local_map.clear();
    coord.global2local_map.resize(coord.global_count, -1);

    if (ccl::global_data::env().enable_init_hostname_sharing) {
        for (i = 0; i < coord.global_count; i++) {
            if (!strncmp(my_hostname,
                         all_hostnames + i * ATL_MAX_HOSTNAME_LEN,
                         my_hostname_len + 1 /* including "-" at the end */)) {
                local_count++;
                int peer_global_proc_idx;
                sscanf(all_hostnames + i * ATL_MAX_HOSTNAME_LEN + my_hostname_len + 1,
                       "%d",
                       &peer_global_proc_idx);
                // Mark the global ids of local node ranks as valid
                coord.global2local_map[peer_global_proc_idx] = 1;
                if (my_global_proc_idx > peer_global_proc_idx)
                    local_idx++;
            }
        }
    }
    // The initial approach for obtaining local coordinates involves hostname
    // sharing, where the current hostname is compared with shared hostnames to
    // calculate the local count and rank. The environment variable enable_init_hostname_sharing
    // determines whether this method is used or if the coordinates are instead retrieved
    // from PMIX_LOCAL_RANK and PMIX_LOCAL_SIZE, queried via the PMIx API during the initialization
    // phase of OneCCL in the global data. By default, the second approach is preferred
    // as it eliminates the need for additional data exchange.
    coord.local_idx = ccl::global_data::env().enable_init_hostname_sharing
                          ? local_idx
                          : ccl::global_data::get().get_local_proc_idx();
    coord.local_count = ccl::global_data::env().enable_init_hostname_sharing
                            ? local_count
                            : ccl::global_data::get().get_local_proc_count();

    local_idx = 0;
    // Assign local ids to local node ranks
    for (i = 0; i < coord.global_count; i++) {
        if (coord.global2local_map[i] == 1 ||
            !ccl::global_data::env().enable_init_hostname_sharing) {
            coord.global2local_map[i] = local_idx;
            local_idx++;
        }
    }

fn_exit:
    free(all_hostnames);
    return ret;

fn_err:
    ret = ATL_STATUS_FAILURE;
    goto fn_exit;
}

atl_status_t atl_ofi_prov_update_addr_table(atl_ofi_ctx_t& ctx,
                                            const atl_proc_coord_t& coord,
                                            size_t prov_idx,
                                            std::shared_ptr<ipmi> pmi,
                                            ep_names_t& ep_names,
                                            void* shared_memory = nullptr,
                                            size_t length = 0,
                                            size_t total_named_eps = 0,
                                            size_t local_addr_offset = 0,
                                            size_t global_addr_offset = 0,
                                            int cur_comm_id = 0) {
    atl_ofi_prov_t* prov = &(ctx.provs[prov_idx]);
    atl_status_t ret = ATL_STATUS_SUCCESS;
    size_t addr_idx = 0;
    int insert_count = 0;
    char* ep_names_table = nullptr;
    char* next_ep_name = nullptr;
    int addr_len = prov->is_shm ? static_cast<int>(FI_NAME_MAX) : prov->addr_len;
    size_t named_ep_count = (prov->sep ? 1 : ctx.ep_count);
    int proc_count = prov->is_shm ? coord.local_count : coord.global_count;
    if (proc_count == 0)
        return ATL_STATUS_SUCCESS;

    size_t ep_names_table_len = addr_len * named_ep_count * proc_count;
    if (ep_names_table_len == 0) {
        LOG_ERROR("ep_names_table_len == 0, addr_len ",
                  prov->addr_len,
                  ", named_ep_count ",
                  named_ep_count,
                  ", proc_count ",
                  proc_count);
        return ATL_STATUS_FAILURE;
    }

    LOG_DEBUG("name ",
              atl_ofi_get_nic_name(prov->info),
              ", is_shm ",
              prov->is_shm,
              ", addr_len ",
              prov->addr_len,
              ", local_count ",
              coord.local_count,
              ", global_count ",
              coord.global_count,
              ", proc_count ",
              proc_count);

    if (ccl::global_data::env().kvs_init_mode == ccl::kvs_mode::pmix_ofi_shm) {
        int global_size = pmi->get_size();
        int global_rank = coord.global_idx;
        int local_rank = coord.local_idx;
        int local_size = coord.local_count;

        char* shm_base = static_cast<char*>(shared_memory);

        // Now, each rank fetches the remote addresses it needs via PMIx individually.
        // The logic: For all global ranks that this process cares about,
        // do a PMIx Get for each "RANK_EP_ADDRS_<global_rank>" key.
        //
        // After fetching, store them into the global section of the shared memory so that other
        // local ranks can also read them without additional PMIx gets.
        auto* bar_mem = static_cast<barrier_mem_t*>(shared_memory);
        shm_barrier((void*)&bar_mem->all_comms, local_size, 0);
        LOG_DEBUG("Barrier before fetching remote data from PMIx");

        ATL_CHECK_STATUS(fetch_and_populate_remote_data(shared_memory,
                                                        local_size,
                                                        global_size,
                                                        global_rank,
                                                        local_rank,
                                                        total_named_eps,
                                                        addr_len,
                                                        local_addr_offset,
                                                        global_addr_offset,
                                                        shm_base,
                                                        length),
                         "fetch_and_populate_remote_data failed");
        shm_barrier((void*)&bar_mem->all_comms, local_size, 0);
        LOG_DEBUG("Barrier after fetching all needed remote data");

        // Now build ep_names_table from shared memory for the given provider
        ep_names_table = (char*)calloc(1, ep_names_table_len);
        if (!ep_names_table) {
            LOG_ERROR("can't allocate epnames_table");
            return ATL_STATUS_FAILURE;
        }

        std::vector<char> ret_ep_name(addr_len, '\0');

        for (int i = 0; i < coord.global_count; i++) {
            if (prov->is_shm && coord.global2local_map[i] == -1) {
                // Skip ranks not on this node for SHM provider
                continue;
            }
            for (size_t j = 0; j < named_ep_count; j++) {
                int global_ep_idx = (int)j;
                char* addr_src = shm_base + global_addr_offset +
                                 (i * total_named_eps + global_ep_idx) * addr_len;
                ret_ep_name.assign(addr_src, addr_src + addr_len);

                if (prov->is_shm) {
                    ret_ep_name.resize(strnlen(ret_ep_name.data(), FI_NAME_MAX) + 1);
                }

                std::memcpy(ep_names_table + addr_idx * addr_len, ret_ep_name.data(), addr_len);
                addr_idx++;
            }
        }
    }
    else {
        int i;
        size_t j;

        /* allocate OFI EP names table that will contain all published names */
        ep_names_table_len = addr_len * named_ep_count * proc_count;
        if (ep_names_table_len == 0) {
            LOG_ERROR("ep_names_table_len == 0, addr_len ",
                      prov->addr_len,
                      ", named_ep_count ",
                      named_ep_count,
                      ", proc_count ",
                      proc_count);
            return ATL_STATUS_FAILURE;
        }

        ep_names_table = (char*)calloc(1, ep_names_table_len);
        if (!ep_names_table) {
            LOG_ERROR("can't allocate epnames_table");
            return ATL_STATUS_FAILURE;
        }

        /* variable initialization must happen before the first *goto* statement */
        std::vector<char> ret_ep_name(addr_len, '\0');

        if (ccl::global_data::env().kvs_init_mode != ccl::kvs_mode::pmix_ofi) {
            if (pmi->pmrt_barrier() != ATL_STATUS_SUCCESS) {
                LOG_ERROR("PMI barrier failed");
                ret = ATL_STATUS_FAILURE;
                goto err_ep_names;
            }
        }

        next_ep_name = ep_names_table;
        // Retrieve all OFI EP names in order
        for (i = 0; i < coord.global_count; i++) {
            if (prov->is_shm) {
                if (coord.global2local_map[i] == -1) {
                    continue;
                }
            }

            for (j = 0; j < named_ep_count; j++) {
                if (prov->is_shm) {
                    ret_ep_name.resize(FI_NAME_MAX, '\0');
                }

                int key =
                    i * ATL_OFI_PMI_PROC_MULTIPLIER + prov_idx * ATL_OFI_PMI_PROV_MULTIPLIER + j;
                if (ccl::global_data::env().kvs_init_mode == ccl::kvs_mode::pmix_ofi) {
                    // PMIx-specific logic for getting addresses
                    std::string key_str = std::string(ATL_OFI_FI_ADDR_PM_KEY) + std::to_string(key);
                    pmix_value_t* ret_value = nullptr;
                    ccl_pmix::get(ccl::global_proc.nspace, i, key_str, &ret_value);

                    ret_ep_name.assign(ret_value->data.bo.bytes,
                                       ret_value->data.bo.bytes + ret_value->data.bo.size);
                    PMIX_VALUE_RELEASE(ret_value);

                    if (prov->is_shm) {
                        size_t original_size = ret_ep_name.size();
                        size_t resized_size = strnlen(ret_ep_name.data(), FI_NAME_MAX) + 1;

                        if (resized_size < original_size) {
                            LOG_WARN(
                                "Endpoint name truncated from ",
                                original_size,
                                " to ",
                                resized_size,
                                " bytes due to FI_NAME_MAX limit. "
                                "This might cause collisions if names differ only in the truncated portion.");
                        }
                        ret_ep_name.resize(strnlen(ret_ep_name.data(), FI_NAME_MAX) + 1);
                    }
                }
                else {
                    ret = pmi->pmrt_kvs_get((char*)ATL_OFI_FI_ADDR_PM_KEY,
                                            key,
                                            (void*)ret_ep_name.data(),
                                            ret_ep_name.size());

                    if (prov->is_shm) {
                        size_t original_size = ret_ep_name.size();
                        size_t resized_size = strnlen(ret_ep_name.data(), FI_NAME_MAX) + 1;

                        if (resized_size < original_size) {
                            LOG_WARN(
                                "Endpoint name truncated from ",
                                original_size,
                                " to ",
                                resized_size,
                                " bytes due to FI_NAME_MAX limit. "
                                "This might cause collisions if names differ only in the truncated portion.");
                        }
                        ret_ep_name.resize(strnlen(ret_ep_name.data(), FI_NAME_MAX) + 1);
                    }

                    if (ret) {
                        LOG_ERROR("pmrt_kvs_get failed: ret: ", ret);
                        goto err_ep_names;
                    }
                }

                auto it = std::find(ep_names.begin(), ep_names.end(), ret_ep_name);
                if (it == ep_names.end()) {
                    ep_names.push_back(ret_ep_name);
                }
                memcpy(next_ep_name, ret_ep_name.data(), ret_ep_name.size());
                if (ret) {
                    LOG_ERROR("kvs_get error: ret ",
                              ret,
                              ", proc_idx ",
                              i,
                              ", ep_idx ",
                              j,
                              ", addr_idx ",
                              addr_idx);
                    goto err_ep_names;
                }

                addr_idx++;
                next_ep_name += ret_ep_name.size();
            }
        }
    }

    LOG_DEBUG("Get: ep_count ", named_ep_count, ", proc_count ", proc_count, ", got ", addr_idx);

    if (addr_idx != static_cast<size_t>(named_ep_count * proc_count)) {
        LOG_ERROR(
            "unexpected get results: expected ", named_ep_count * proc_count, ", got ", addr_idx);
        ret = ATL_STATUS_FAILURE;
        goto err_addr_table;
    }

    if (prov->addr_table != nullptr)
        free(prov->addr_table);

    prov->addr_table = (fi_addr_t*)calloc(1, ctx.ep_count * proc_count * sizeof(fi_addr_t));
    if (!prov->addr_table)
        goto err_ep_names;

    /* insert all the EP names into the AV */
    insert_count = fi_av_insert(
        prov->av, ep_names_table, named_ep_count * proc_count, prov->addr_table, 0, nullptr);

    LOG_DEBUG("av_insert: ep_count ",
              named_ep_count,
              ", proc_count ",
              proc_count,
              ", inserted ",
              insert_count);

    if (insert_count != (int)(named_ep_count * proc_count)) {
        LOG_ERROR("unexpected av_insert results: expected ",
                  named_ep_count * proc_count,
                  " got ",
                  insert_count);
        ret = ATL_STATUS_FAILURE;
        goto err_addr_table;
    }
    else {
        ret = ATL_STATUS_SUCCESS;
    }

    if (prov->sep) {
        if (named_ep_count != 1) {
            LOG_ERROR("unexpected named_ep_count ", named_ep_count);
            goto err_addr_table;
        }

        fi_addr_t* table = (fi_addr_t*)calloc(1, proc_count * sizeof(fi_addr_t));
        if (table == nullptr) {
            LOG_ERROR("memory allocaion failed");
            ret = ATL_STATUS_FAILURE;
            goto err_addr_table;
        }

        memcpy(table, prov->addr_table, proc_count * sizeof(fi_addr_t));
        for (int i = 0; i < proc_count; i++) {
            for (size_t j = 0; j < ctx.ep_count; j++) {
                prov->addr_table[i * ctx.ep_count + j] = fi_rx_addr(table[i], j, prov->rx_ctx_bits);
            }
        }
        free(table);
    }

    free(ep_names_table);
    return ret;
    /* abnormal end of execution */
err_addr_table:
    free(prov->addr_table);

err_ep_names:
    free(ep_names_table);
    return ret;
}

atl_status_t atl_ofi_prov_ep_get_name(atl_ofi_prov_t* prov, size_t ep_idx) {
    int ret;

    atl_ofi_prov_ep_t* ep = &(prov->eps[ep_idx]);
    struct fid_ep* fi_ep = (prov->sep) ? prov->sep : ep->tx;

    ep->name.addr = nullptr;
    ep->name.len = 0;

    ret = fi_getname(&fi_ep->fid, ep->name.addr, &(ep->name.len));
    if ((ret != -FI_ETOOSMALL) || ep->name.len <= 0)
        ep->name.len = FI_NAME_MAX;

    if (ep->name.addr)
        free(ep->name.addr);

    ep->name.addr = calloc(1, ep->name.len);

    if (!(ep->name.addr)) {
        LOG_ERROR("can't allocate addr");
        ret = ATL_STATUS_FAILURE;
        goto err_addr;
    }

    ret = fi_getname(&fi_ep->fid, ep->name.addr, &(ep->name.len));
    if (ret) {
        LOG_ERROR("fi_getname error");
        goto err_getname;
    }

    prov->addr_len = MAX(prov->addr_len, ep->name.len);

    return ATL_STATUS_SUCCESS;

err_getname:
    free(ep->name.addr);
    ep->name.addr = nullptr;
    ep->name.len = 0;

err_addr:
    return ATL_OFI_RET(ret);
}

atl_status_t atl_ofi_prov_eps_connect(atl_ofi_ctx_t& ctx,
                                      const atl_proc_coord_t& coord,
                                      size_t prov_idx,
                                      std::shared_ptr<ipmi> pmi,
                                      ep_names_t& ep_names) {
    int ret;
    size_t ep_idx;

    atl_ofi_prov_t* prov = &(ctx.provs[prov_idx]);
    size_t named_ep_count = (prov->sep ? 1 : ctx.ep_count);

    prov->addr_len = 0;

    for (ep_idx = 0; ep_idx < ctx.ep_count; ep_idx++) {
        ret = atl_ofi_prov_ep_get_name(prov, ep_idx);
        if (ret) {
            LOG_ERROR("atl_ofi_prov_ep_get_name error");
            return ATL_STATUS_FAILURE;
        }
    }

    if (ccl::global_data::env().kvs_init_mode == ccl::kvs_mode::pmix_ofi_shm) {
        int global_rank = coord.global_idx;
        int local_rank = coord.local_idx;
        int local_size = coord.local_count;
        int global_size = pmi->get_size();
        bool is_node_root = (local_rank == 0);

        size_t total_named_eps = named_ep_count;
        size_t addr_len = prov->addr_len;

        // Shared memory layout:
        // [COUNTER_OFFSET] + [local_size * total_named_eps * addr_len for local addresses]
        // + [global_size * total_named_eps * addr_len for global addresses (if needed)]
        size_t local_addrs_size = local_size * total_named_eps * addr_len;
        size_t global_addrs_size = global_size * total_named_eps * addr_len;
        size_t length = COUNTER_OFFSET + local_addrs_size + global_addrs_size;

        void* shared_memory = nullptr;
        ATL_CHECK_STATUS(setup_shared_memory(get_shm_filename("/dev/shm/EPS-shm-file"),
                                             local_size,
                                             is_node_root,
                                             length,
                                             &shared_memory,
                                             0),
                         "setup_shared_memory failed");

        char* shm_base = static_cast<char*>(shared_memory);
        size_t local_addr_offset = COUNTER_OFFSET;
        size_t global_addr_offset = COUNTER_OFFSET + local_addrs_size;

        // Write local endpoint addresses into shared memory
        for (size_t ep_idx = 0; ep_idx < named_ep_count; ep_idx++) {
            char* local_dest =
                shm_base + local_addr_offset + (local_rank * total_named_eps + ep_idx) * addr_len;
            std::memcpy(local_dest, prov->eps[ep_idx].name.addr, addr_len);
        }

        auto bar_mem = static_cast<barrier_mem_t*>(shared_memory);
        shm_barrier((void*)&bar_mem->all_comms, local_size, 0);
        LOG_DEBUG("Barrier after writing local addresses");

        // PMIx Put by Each Rank Individually
        // Each rank publishes its local endpoints directly with a rank-specific key:
        {
            std::string key_str = "RANK_EP_ADDRS_" + std::to_string(global_rank);
            pmix_value_t value;
            PMIX_VALUE_CONSTRUCT(&value);
            char* rank_src =
                shm_base + local_addr_offset + (local_rank * total_named_eps * addr_len);
            pmix_byte_object_t bo;
            bo.bytes = rank_src;
            bo.size = total_named_eps * addr_len;
            ccl_pmix::value_load(value, &bo, PMIX_BYTE_OBJECT);
            LOG_DEBUG("PMIx PUT key: ", key_str);
            ccl_pmix::put(key_str, value);
        }

        ccl_pmix::commit();
        ccl_pmix::fence(ccl::global_proc.nspace, pmi->get_rank(), pmi->get_size());

        // After fence, all ranks have published their endpoints.
        // The next step (in the update_addr_table) will be to fetch needed addresses.
        ret = atl_ofi_prov_update_addr_table(ctx,
                                             coord,
                                             prov_idx,
                                             pmi,
                                             ep_names,
                                             shared_memory,
                                             length,
                                             total_named_eps,
                                             local_addr_offset,
                                             global_addr_offset);
    }
    else {
        for (ep_idx = 0; ep_idx < named_ep_count; ep_idx++) {
            atl_ofi_prov_ep_t* ep = &(prov->eps[ep_idx]);
            int key = coord.global_idx * ATL_OFI_PMI_PROC_MULTIPLIER +
                      prov_idx * ATL_OFI_PMI_PROV_MULTIPLIER + ep_idx;
            if (ccl::global_data::env().kvs_init_mode == ccl::kvs_mode::pmix_ofi) {
                // PMIx-specific logic for putting addresses
                pmix_value_t value;
                PMIX_VALUE_CONSTRUCT(&value);

                pmix_byte_object_t bo;
                bo.bytes = static_cast<char*>(ep->name.addr);
                bo.size = ep->name.len;
                ccl_pmix::value_load(value, &bo, PMIX_BYTE_OBJECT);

                std::string key_str = std::string(ATL_OFI_FI_ADDR_PM_KEY) + std::to_string(key);
                LOG_DEBUG("PMIx PUT key: ", key);
                ccl_pmix::put(key_str, value);
            }
            else {
                ret = pmi->pmrt_kvs_put((char*)ATL_OFI_FI_ADDR_PM_KEY,
                                        coord.global_idx * ATL_OFI_PMI_PROC_MULTIPLIER +
                                            prov_idx * ATL_OFI_PMI_PROV_MULTIPLIER + ep_idx,
                                        ep->name.addr,
                                        ep->name.len);
                if (ret) {
                    LOG_ERROR("pmrt_kvs_put: ret: ", ret);
                    return ATL_STATUS_FAILURE;
                }
            }
        }

        if (ccl::global_data::env().kvs_init_mode == ccl::kvs_mode::pmix_ofi) {
            ccl_pmix::commit();
            ccl_pmix::fence(ccl::global_proc.nspace, pmi->get_rank(), pmi->get_size());
        }
        ret = atl_ofi_prov_update_addr_table(ctx, coord, prov_idx, pmi, ep_names);
    }

    return ATL_OFI_RET(ret);
}

void atl_ofi_prov_ep_destroy(atl_ofi_prov_t* prov, atl_ofi_prov_ep_t* ep) {
    if (ep->rx)
        fi_close(&ep->rx->fid);

    if (prov->sep && ep->tx)
        fi_close(&ep->tx->fid);

    if (ep->cq)
        fi_close(&ep->cq->fid);

    if (ep->name.addr)
        free(ep->name.addr);

    ep->rx = ep->tx = nullptr;
    ep->cq = nullptr;
    ep->name.addr = nullptr;
    ep->name.len = 0;
}

void atl_ofi_prov_destroy(atl_ofi_ctx_t& ctx, atl_ofi_prov_t* prov) {
    size_t i;

    for (i = 0; i < ctx.ep_count; i++) {
        atl_ofi_prov_ep_destroy(prov, &(prov->eps[i]));
    }

    free(prov->eps);
    free(prov->addr_table);

    if (prov->sep)
        fi_close(&prov->sep->fid);

    if (prov->av)
        fi_close(&prov->av->fid);

    if (prov->domain)
        fi_close(&prov->domain->fid);

    if (prov->fabric)
        fi_close(&prov->fabric->fid);

    if (prov->info) {
        fi_freeinfo(prov->info);
    }
}

int atl_ofi_wait_cancel_cq(struct fid_cq* cq) {
    struct fi_cq_err_entry err_entry;
    int ret, i;
    struct fi_cq_tagged_entry entries[ATL_OFI_CQ_BUNCH_SIZE];

    double time = 0;
    clock_t start, end;

    while (time < ATL_OFI_WAIT_SEC) {
        for (i = 0; i < ATL_OFI_CQ_READ_ITERS; i++) {
            start = clock();
            ret = fi_cq_read(cq, entries, ATL_OFI_CQ_BUNCH_SIZE);

            if (ret < 0 && ret != -FI_EAGAIN) {
                ret = fi_cq_readerr(cq, &err_entry, 0);

                if (err_entry.err != FI_ECANCELED) {
                    LOG_ERROR(
                        "fi_cq_readerr: err: ",
                        err_entry.err,
                        ", prov_err: ",
                        fi_cq_strerror(cq, err_entry.prov_errno, err_entry.err_data, nullptr, 0),
                        "(",
                        err_entry.prov_errno,
                        ")");
                    return ATL_STATUS_FAILURE;
                }
                return ATL_STATUS_SUCCESS;
            }
        }
        end = clock();
        time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    LOG_ERROR("too long for cancel");

    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi_prov_ep_init(atl_ofi_prov_t* prov, size_t ep_idx) {
    ssize_t ret = 0;

    struct fi_cq_attr cq_attr;
    struct fi_tx_attr tx_attr;
    struct fi_rx_attr rx_attr;

    atl_ofi_prov_ep_t* ep = &(prov->eps[ep_idx]);

    memset(&cq_attr, 0, sizeof(cq_attr));
    cq_attr.format = FI_CQ_FORMAT_TAGGED;

    ATL_OFI_CALL(
        fi_cq_open(prov->domain, &cq_attr, &ep->cq, nullptr), ret, return ATL_STATUS_FAILURE);

    if (prov->sep) {
        rx_attr = *prov->info->rx_attr;
        rx_attr.caps |= FI_TAGGED;

        ATL_OFI_CALL(fi_rx_context(prov->sep, ep_idx, &rx_attr, &ep->rx, nullptr), ret, goto err);

        ATL_OFI_CALL(fi_ep_bind(ep->rx, &ep->cq->fid, FI_RECV), ret, goto err);

        tx_attr = *prov->info->tx_attr;
        tx_attr.caps |= FI_TAGGED;

        ATL_OFI_CALL(fi_tx_context(prov->sep, ep_idx, &tx_attr, &ep->tx, nullptr), ret, goto err);

        ATL_OFI_CALL(fi_ep_bind(ep->tx, &ep->cq->fid, FI_SEND), ret, goto err);

        fi_enable(ep->rx);
        fi_enable(ep->tx);
    }
    else {
        struct fid_ep* endpoint;

        ATL_OFI_CALL(fi_endpoint(prov->domain, prov->info, &endpoint, nullptr), ret, goto err);

        ep->tx = ep->rx = endpoint;

        ATL_OFI_CALL(fi_ep_bind(endpoint, &ep->cq->fid, FI_SEND | FI_RECV), ret, goto err);

        ATL_OFI_CALL(fi_ep_bind(endpoint, &prov->av->fid, 0), ret, goto err);

        fi_enable(endpoint);
    }

    return ATL_STATUS_SUCCESS;

err:
    atl_ofi_prov_ep_destroy(prov, ep);
    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi_try_to_drain_cq_err(struct fid_cq* cq) {
    struct fi_cq_err_entry err_entry;
    int ret = fi_cq_readerr(cq, &err_entry, 0);
    if (ret != 1) {
        LOG_DEBUG("unable to fi_cq_readerr");
        return ATL_STATUS_FAILURE;
    }
    else {
        if (err_entry.err != FI_ENOMSG && err_entry.err != FI_ECANCELED &&
            err_entry.err != FI_ETRUNC) {
            LOG_ERROR("fi_cq_readerr: err: ",
                      err_entry.err,
                      ", prov_err: ",
                      fi_cq_strerror(cq, err_entry.prov_errno, err_entry.err_data, nullptr, 0),
                      "(",
                      err_entry.prov_errno,
                      ")");
            return ATL_STATUS_FAILURE;
        }
        return ATL_STATUS_SUCCESS;
    }
}

int atl_ofi_try_to_drain_cq(struct fid_cq* cq) {
    int ret = -FI_EAGAIN, i;
    double time = 0;
    clock_t start, end;
    struct fi_cq_tagged_entry entries[ATL_OFI_CQ_BUNCH_SIZE];

    while (time < ATL_OFI_WAIT_SEC) {
        start = clock();
        for (i = 0; i < ATL_OFI_CQ_READ_ITERS; i++) {
            ret = fi_cq_read(cq, entries, ATL_OFI_CQ_BUNCH_SIZE);

            if (ret < 0 && ret != -FI_EAGAIN) {
                atl_ofi_try_to_drain_cq_err(cq);
                return ret;
            }

            if (ret > 0)
                return ret;
        }
        end = clock();
        time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    return ret;
}

void atl_ofi_reset(atl_ofi_ctx_t& ctx) {
    int again = 1;
    size_t prov_idx, ep_idx;
    int recv_buf_len = sizeof(char);
    char* recv_buf;
    struct fi_context fi_ctx;
    recv_buf = (char*)malloc(recv_buf_len);
    for (prov_idx = 0; prov_idx < ctx.prov_count; prov_idx++) {
        atl_ofi_prov_t* prov = &(ctx.provs[prov_idx]);

        for (ep_idx = 0; ep_idx < ctx.ep_count; ep_idx++) {
            atl_ofi_prov_ep_t* ep = &(prov->eps[ep_idx]);

            /* complete active sends and receives */
            while (atl_ofi_try_to_drain_cq(ep->cq) != -FI_EAGAIN) {
            }

            /* try to complete active incoming sends */
            while (again) {
                again = 0;
                /* post recv to complete incoming send */
                while (fi_trecv(ep->rx,
                                recv_buf,
                                recv_buf_len,
                                nullptr,
                                FI_ADDR_UNSPEC,
                                0,
                                UINTMAX_MAX,
                                &fi_ctx) == -FI_EAGAIN) {
                }

                /* wait until recv will be completed or finished by timeout */
                while (atl_ofi_try_to_drain_cq(ep->cq) != -FI_EAGAIN) {
                    /* something is completed -> send queue not empty */
                    again = 1;
                }
            }

            /* nothing to recv -> cancel last recv */
            fi_cancel(&ep->rx->fid, &fi_ctx);

            atl_ofi_wait_cancel_cq(ep->cq);
        }
    }

    free(recv_buf);
}

atl_status_t atl_ofi_adjust_env(const atl_attr_t& attr) {
    char* prov_env = getenv("FI_PROVIDER");

    if (prov_env && strlen(prov_env)) {
        CCL_THROW_IF_NOT(strlen(prov_env) < sizeof(global_data.prov_env_copy),
                         "too long FI_PROVIDER value, max expected length ",
                         sizeof(global_data.prov_env_copy));
        memcpy(global_data.prov_env_copy, prov_env, strlen(prov_env));
    }

    if (attr.in.enable_shm) {
        /* add shm provider in the list of allowed providers */
        if (prov_env && !strstr(prov_env, ATL_OFI_SHM_PROV_NAME)) {
            /* whether single provider will be in the final env variable */
            int single_prov = (strlen(prov_env) == 0) ? 1 : 0;

            size_t prov_env_new_size = strlen(prov_env) + strlen(ATL_OFI_SHM_PROV_NAME) +
                                       (single_prov ? 0 : 1) + /* for delimeter */
                                       1; /* for terminating null symbol */

            char* prov_env_new = (char*)calloc(prov_env_new_size, sizeof(char));
            if (prov_env_new == nullptr) {
                LOG_ERROR("memory allocaion failed");
                return ATL_STATUS_FAILURE;
            }

            if (single_prov)
                snprintf(prov_env_new, prov_env_new_size, "%s", ATL_OFI_SHM_PROV_NAME);
            else {
                snprintf(prov_env_new, prov_env_new_size, "%s,%s", prov_env, ATL_OFI_SHM_PROV_NAME);
            }

            LOG_INFO("atl-ofi-shm is requested, modify FI_PROVIDER: old value: ",
                     prov_env,
                     ", new value: ",
                     prov_env_new);

            setenv("FI_PROVIDER", prov_env_new, 1);

            free(prov_env_new);
        }
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi_set_env(const atl_attr_t& attr) {
    if (global_data.is_env_inited) {
        return ATL_STATUS_SUCCESS;
    }

    setenv("FI_PSM2_DELAY", "0", 0);
    setenv("FI_PSM2_TIMEOUT", "0", 0);
    setenv("FI_PSM2_LOCK_LEVEL", "1", 0);
    setenv("FI_PSM2_NAME_SERVER", "0", 0);
    setenv("HFI_NO_CPUAFFINITY", "1", 0);
    setenv("PSM2_MULTI_EP", "1", 0);

    setenv("FI_PSM3_DELAY", "0", 0);
    setenv("FI_PSM3_TIMEOUT", "0", 0);
    setenv("FI_PSM3_LOCK_LEVEL", "1", 0);
    setenv("FI_PSM3_NAME_SERVER", "0", 0);
    setenv("PSM3_NO_CPUAFFINITY", "1", 0);
    setenv("PSM3_RDMA", "2", 0);
    setenv("PSM3_MR_CACHE_MODE", "0", 0); //TODO temporary
    setenv("PSM3_MULTI_EP", "1", 0);
    setenv("PSM3_HAL", "verbs", 0);
    if (attr.in.mnic_type == ATL_MNIC_NONE)
        setenv("PSM3_NIC", "any", 0);

    char* hydra_uuid_env = getenv("I_MPI_HYDRA_UUID");
    if (hydra_uuid_env) {
        setenv("FI_PSM2_UUID", hydra_uuid_env, 0);
        setenv("FI_PSM3_UUID", hydra_uuid_env, 0);
    }

    setenv("FI_OFI_RXM_USE_HASH", "0", 0);
    setenv("FI_OFI_RXM_USE_SRX", "0", 0);
    setenv("FI_OFI_RXM_RX_SIZE", "8192", 0);
    setenv("FI_OFI_RXM_TX_SIZE", "8192", 0);
    setenv("FI_OFI_RXM_MSG_RX_SIZE", "128", 0);
    setenv("FI_OFI_RXM_MSG_TX_SIZE", "128", 0);

    setenv("FI_SHM_TX_SIZE", "8192", 0);
    setenv("FI_SHM_RX_SIZE", "8192", 0);

#ifdef CCL_ENABLE_SYCL
    setenv("FI_SHM_DISABLE_CMA", "1", 0);
    setenv("FI_HMEM", "ze", 0);
#endif // CCL_ENABLE_SYCL

    setenv("FI_MLX_MULTI_EP", "1", 0);

    atl_ofi_adjust_env(attr);

#ifdef CCL_ENABLE_OFI_OOT_PROV
    /*
       load libfabric symbols into global namespace
       to workaround issue with undefined symbols
       in case of out-of-tree providers, like OFI/PSM3
    */
    global_data.dlhandle = dlopen(ccl::get_ofi_lib_path().c_str(), RTLD_GLOBAL | RTLD_NOW);
    if (global_data.dlhandle == nullptr) {
        LOG_WARN("dlopen (libfabric.so): ", dlerror());
    }
#endif // CCL_ENABLE_OFI_OOT_PROV

    global_data.is_env_inited = 1;

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi_get_prov_list(atl_ofi_ctx_t& ctx,
                                   const char* prov_name,
                                   struct fi_info* base_hints,
                                   struct fi_info** out_prov_list) {
    struct fi_info* hints = nullptr;
    struct fi_info* prov_list = nullptr;
    ssize_t ret = 0;
    int fi_version = FI_VERSION(global_data.fi_major_version, global_data.fi_minor_version);
    const char* prov_name_str = (prov_name) ? prov_name : "<default>";

    hints = fi_dupinfo(base_hints);
    if (!hints) {
        LOG_ERROR("fi_dupinfo error");
        goto err;
    }

    *out_prov_list = nullptr;

    LOG_DEBUG("request providers with name: ", prov_name_str);

    hints->fabric_attr->prov_name = (prov_name) ? strdup(prov_name) : nullptr;

    ret = fi_getinfo(fi_version, nullptr, nullptr, 0ULL, hints, &prov_list);

    if (ret || !prov_list) {
        LOG_ERROR("fi_getinfo error: ret ", ret, ", providers ", (void*)prov_list);
        goto err;
    }
    if (!strcmp(prov_list->fabric_attr->prov_name, ATL_OFI_SHM_PROV_NAME) &&
        prov_list->caps & FI_HMEM) {
        LOG_ERROR("skip OFI/SHM with HMEM capability");
        goto err;
    }

    if (prov_list->domain_attr->max_ep_tx_ctx > 1) {
        hints->ep_attr->tx_ctx_cnt = ctx.ep_count;
        hints->ep_attr->rx_ctx_cnt = ctx.ep_count;
    }
    else {
        hints->ep_attr->tx_ctx_cnt = 1;
        hints->ep_attr->rx_ctx_cnt = 1;
    }

    fi_freeinfo(prov_list);
    prov_list = nullptr;

    ret = fi_getinfo(fi_version, nullptr, nullptr, 0ULL, hints, &prov_list);
    if (ret || !prov_list) {
        LOG_ERROR("fi_getinfo error, prov_name ", prov_name_str);
        goto err;
    }

    fi_freeinfo(hints);
    hints = nullptr;

    *out_prov_list = prov_list;
    return ATL_STATUS_SUCCESS;

err:
    if (hints) {
        fi_freeinfo(hints);
    }
    if (prov_list) {
        fi_freeinfo(prov_list);
    }
    LOG_ERROR("can't create providers for name ", prov_name_str);
    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi_prov_init(atl_ofi_ctx_t& ctx,
                               const atl_proc_coord_t& coord,
                               struct fi_info* info,
                               atl_ofi_prov_t* prov,
                               atl_attr_t* attr,
                               std::shared_ptr<ipmi> pmi,
                               ep_names_t& ep_names) {
    struct fi_av_attr av_attr;
    size_t ep_idx = 0;
    ssize_t ret = 0;

    memset(&av_attr, 0, sizeof(av_attr));

    if (coord.global_idx == 0) {
        LOG_INFO("provider: ", info->fabric_attr->prov_name);
        LOG_INFO("  nic: ", atl_ofi_get_nic_info(info));
        LOG_INFO("  mr_mode: ", info->domain_attr->mr_mode);
        LOG_INFO("  threading: ", info->domain_attr->threading);
        LOG_INFO("  tx_ctx_cnt: ", info->domain_attr->tx_ctx_cnt);
        LOG_INFO("  max_ep_tx_ctx: ", info->domain_attr->max_ep_tx_ctx);
        LOG_INFO("  max_msg_size: ", info->ep_attr->max_msg_size);
    }

    prov->info = fi_dupinfo(info);

    if (!prov->info) {
        LOG_ERROR("fi_dupinfo error");
        goto err;
    }

    prov->max_msg_size = info->ep_attr->max_msg_size;

    ATL_OFI_CALL(fi_fabric(info->fabric_attr, &prov->fabric, nullptr), ret, goto err);

    ATL_OFI_CALL(fi_domain(prov->fabric, info, &prov->domain, nullptr), ret, goto err);

    av_attr.type = FI_AV_TABLE;
    av_attr.rx_ctx_bits = prov->rx_ctx_bits = (int)ceil(log2(prov->info->ep_attr->rx_ctx_cnt));

    ATL_OFI_CALL(fi_av_open(prov->domain, &av_attr, &prov->av, nullptr), ret, goto err);

    if (info->domain_attr->max_ep_tx_ctx > 1) {
        ATL_OFI_CALL(fi_scalable_ep(prov->domain, info, &prov->sep, nullptr), ret, goto err);
        ATL_OFI_CALL(fi_scalable_ep_bind(prov->sep, &prov->av->fid, 0), ret, goto err);
    }

    prov->eps = (atl_ofi_prov_ep_t*)calloc(1, sizeof(atl_ofi_prov_ep_t) * ctx.ep_count);
    if (!prov->eps) {
        LOG_ERROR("can't allocate prov->eps");
        goto err;
    }

    for (ep_idx = 0; ep_idx < ctx.ep_count; ep_idx++) {
        ret = atl_ofi_prov_ep_init(prov, ep_idx);
        if (ret) {
            LOG_ERROR("atl_ofi_prov_ep_init error");
            goto err;
        }
    }

    if (prov->sep) {
        fi_enable(prov->sep);
    }

    /* TODO: make separate function to be called on CCL comm creation */
    ret = atl_ofi_prov_eps_connect(ctx, coord, prov->idx, std::move(pmi), ep_names);
    if (ret) {
        LOG_ERROR("atl_ofi_prov_eps_connect error, prov_idx ", prov->idx);
        goto err;
    }

    ATL_CALL(atl_ofi_adjust_out_tag(prov, attr), goto err);

    return ATL_STATUS_SUCCESS;

err:
    if (prov->info) {
        fi_freeinfo(prov->info);
        prov->info = nullptr;
    }
    LOG_ERROR("can't init provider ", atl_ofi_get_nic_name(info));
    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi_adjust_out_tag(atl_ofi_prov_t* prov, atl_attr_t* attr) {
    size_t tag_bits = 64;
    uint64_t mem_tag_format = prov->info->ep_attr->mem_tag_format;
    while (tag_bits && !(mem_tag_format & ((uint64_t)1 << (tag_bits - 1)))) {
        tag_bits--;
    }

    attr->out.tag_bits = std::min(attr->out.tag_bits, tag_bits);

    if (attr->out.tag_bits == 64) {
        attr->out.max_tag = 0xFFFFFFFFFFFFFFFF;
    }
    else {
        attr->out.max_tag = (((uint64_t)1 << attr->out.tag_bits) - 1);
    }

    const char* prov_name = prov->info->fabric_attr->prov_name;

    if (!(attr->out.tag_bits > 0)) {
        LOG_ERROR("unexpected tag_bits ", attr->out.tag_bits, " for prov ", prov_name);
        return ATL_STATUS_FAILURE;
    }

    if (!(attr->out.max_tag > 0)) {
        LOG_ERROR("unexpected max_tag ", attr->out.max_tag, " for prov ", prov_name);
        return ATL_STATUS_FAILURE;
    }
    LOG_INFO(prov_name,
             " tag_bits: ",
             attr->out.tag_bits,
             ", max_tag: ",
             attr->out.max_tag,
             ", mem_tag_format: ",
             mem_tag_format);

    return ATL_STATUS_SUCCESS;
}

static bool atl_ofi_is_nic_down(struct fi_info* prov) {
    if (prov->nic && prov->nic->link_attr->state == FI_LINK_DOWN) {
        return true;
    }

    return false;
}

/* determine if NIC has already been included in others */
int atl_ofi_nic_already_used(const struct fi_info* prov,
                             const std::vector<struct fi_info*>& others,
                             bool check_pci = false) {
    for (size_t i = 0; i < others.size(); i++) {
        if (check_pci && prov->nic && others[i]->nic &&
            prov->nic->bus_attr->bus_type == FI_BUS_PCI &&
            others[i]->nic->bus_attr->bus_type == FI_BUS_PCI) {
            struct fi_pci_attr pci = prov->nic->bus_attr->attr.pci;
            struct fi_pci_attr other_pci = others[i]->nic->bus_attr->attr.pci;
            LOG_TRACE("compare nic ",
                      prov->fabric_attr->prov_name,
                      " pci ",
                      (int)pci.domain_id,
                      ":",
                      (int)pci.bus_id,
                      ":",
                      (int)pci.device_id,
                      ":",
                      (int)pci.function_id,
                      " with nic ",
                      others[i]->fabric_attr->prov_name,
                      " pci ",
                      (int)other_pci.domain_id,
                      ":",
                      (int)other_pci.bus_id,
                      ":",
                      (int)other_pci.device_id,
                      ":",
                      (int)other_pci.function_id);
            if (pci.domain_id == other_pci.domain_id && pci.bus_id == other_pci.bus_id &&
                pci.device_id == other_pci.device_id && pci.function_id == other_pci.function_id)
                return 1;
        }
        else {
            LOG_TRACE("compare nic ",
                      atl_ofi_get_nic_name(prov),
                      " with nic ",
                      atl_ofi_get_nic_name(others[i]));
            if (atl_ofi_get_short_nic_name(prov) == atl_ofi_get_short_nic_name(others[i]))
                return 1;
        }
    }
    return 0;
}

/* return true if the NIC is bound to the same socket as calling process */
int atl_ofi_is_nic_local(struct fi_info* info) {
    if (info->nic && info->nic->bus_attr->bus_type == FI_BUS_PCI) {
        struct fi_pci_attr pci = info->nic->bus_attr->attr.pci;
        return ccl::global_data::get().hwloc_wrapper->is_dev_close_by_pci(
            pci.domain_id, pci.bus_id, pci.device_id, pci.function_id);
    }
    return 0;
}

atl_status_t atl_ofi_parse_mnic_name(atl_ofi_ctx_t& ctx, std::string str_to_parse) {
    atl_status_t ret = ATL_STATUS_SUCCESS;

    std::string include_str;
    std::string exclude_str;

    auto pos = str_to_parse.find('^');

    if (pos == 0) {
        exclude_str = str_to_parse.substr(1);
    }
    else {
        if (pos != std::string::npos) {
            include_str = str_to_parse.substr(0, pos - 1);
            exclude_str = str_to_parse.substr(pos + 1);
        }
        else {
            include_str = str_to_parse.substr(0, pos);
        }
    }

    if (!include_str.empty()) {
        LOG_DEBUG("include names str: ", include_str);
    }

    if (!exclude_str.empty()) {
        LOG_DEBUG("exclude names str: ", exclude_str);
    }

    auto include_names = ccl::utils::tokenize<std::vector<std::string>>(include_str, ',');
    auto exclude_names = ccl::utils::tokenize<std::vector<std::string>>(exclude_str, ',');

    if (!include_names.empty() && !exclude_names.empty()) {
        auto include_set = std::set<std::string>(include_names.begin(), include_names.end());
        auto exclude_set = std::set<std::string>(exclude_names.begin(), exclude_names.end());

        std::set<std::string> intersect;
        std::set_intersection(include_set.begin(),
                              include_set.end(),
                              exclude_set.begin(),
                              exclude_set.end(),
                              std::inserter(intersect, intersect.begin()));
        if (!intersect.empty()) {
            LOG_ERROR("include and exclude sets can not intersect");
            ret = ATL_STATUS_FAILURE;
        }

        for (auto& include_name : include_names) {
            for (auto& exclude_name : exclude_names) {
                std::string& larger_name =
                    (include_name.size() > exclude_name.size()) ? include_name : exclude_name;
                std::string& smaller_name =
                    (include_name.size() > exclude_name.size()) ? exclude_name : include_name;
                if (larger_name.substr(0, smaller_name.size()) == smaller_name) {
                    LOG_ERROR("include name ",
                              include_name,
                              " and exclude name ",
                              exclude_name,
                              " have commom prefix");
                    ret = ATL_STATUS_FAILURE;
                    break;
                }
            }
        }
    }

    if (ret == ATL_STATUS_SUCCESS) {
        LOG_DEBUG("include names: ", ccl::utils::vec_to_string(include_names));
        LOG_DEBUG("exclude names: ", ccl::utils::vec_to_string(exclude_names));
        ctx.mnic_include_names = include_names;
        ctx.mnic_exclude_names = exclude_names;
    }

    return ret;
}

int atl_ofi_is_allowed_nic_name(atl_ofi_ctx_t& ctx, struct fi_info* info) {
    auto& include_names = ctx.mnic_include_names;
    auto& exclude_names = ctx.mnic_exclude_names;
    std::string nic_name = atl_ofi_get_short_nic_name(info);

    int should_include = 0;
    int should_exclude = 0;

    if (include_names.empty()) {
        should_include = 1;
    }

    for (auto& name : include_names) {
        if (nic_name.substr(0, name.size()) == name) {
            should_include = 1;
            break;
        }
    }

    for (auto& name : exclude_names) {
        if (nic_name.substr(0, name.size()) == name) {
            should_exclude = 1;
            break;
        }
    }

    return (should_include && !should_exclude);
}

bool atl_ofi_compare_nics(const struct fi_info* nic1, const struct fi_info* nic2) {
    if (nic1->nic && !nic2->nic) {
        return true;
    }
    else if (!nic1->nic && nic2->nic) {
        return false;
    }
    return (atl_ofi_get_short_nic_name(nic1) < atl_ofi_get_short_nic_name(nic2));
}

atl_status_t atl_ofi_open_nw_provs(atl_ofi_ctx_t& ctx,
                                   const atl_proc_coord_t& coord,
                                   struct fi_info* base_hints,
                                   atl_attr_t* attr,
                                   std::shared_ptr<ipmi> pmi,
                                   std::vector<ep_names_t>& ep_names,
                                   bool log_on_error) {
    atl_status_t ret = ATL_STATUS_SUCCESS;
    struct fi_info* prov_list = nullptr;
    struct fi_info* prov_iter = nullptr;
    size_t idx = 0, prov_idx = 0;
    char* prov_name = nullptr;
    atl_ofi_prov_t* prov = nullptr;
    std::vector<struct fi_info*> name_provs;
    std::vector<struct fi_info*> topo_provs;
    std::vector<struct fi_info*> final_provs;
    std::set<std::string> all_nic_names;
    int prov_offset = 0;

    ctx.nw_prov_count = 0;

    /* 1. get full list of providers */
    if (strlen(global_data.prov_env_copy) && !strstr(global_data.prov_env_copy, ","))
        prov_name = global_data.prov_env_copy;
    else
        prov_name = nullptr;
    ret = atl_ofi_get_prov_list(ctx, prov_name, base_hints, &prov_list);
    if (ret != ATL_STATUS_SUCCESS) {
        if (log_on_error) {
            LOG_ERROR(
                "atl_ofi_get_prov_list(ctx, prov_name, base_hints, &prov_list)\n fails with status: ",
                ret);
        }
        goto err;
    }

    /* 2. filter out by names */
    prov_iter = prov_list;
    while (prov_iter) {
        LOG_DEBUG("name filter: check nic ", atl_ofi_get_nic_name(prov_iter));
        if (atl_ofi_is_nic_down(prov_iter)) {
            LOG_DEBUG("nic ", atl_ofi_get_nic_name(prov_iter), " is in down state, skip");
        }
        else if (!atl_ofi_nic_already_used(prov_iter, name_provs)) {
            all_nic_names.insert(atl_ofi_get_short_nic_name(prov_iter));
            if (atl_ofi_is_allowed_nic_name(ctx, prov_iter)) {
                LOG_DEBUG("name filter: found suitable nic ", atl_ofi_get_nic_name(prov_iter));
                name_provs.push_back(fi_dupinfo(prov_iter));
            }
        }
        prov_iter = prov_iter->next;
    }

    /* sort by names */
    std::sort(name_provs.begin(), name_provs.end(), atl_ofi_compare_nics);

    if (name_provs.empty()) {
        LOG_ERROR("name filter: can not find network providers",
                  ", include names: ",
                  ccl::utils::vec_to_string(ctx.mnic_include_names),
                  ", exclude names: ",
                  ccl::utils::vec_to_string(ctx.mnic_exclude_names),
                  ", all names: ",
                  ccl::utils::vec_to_string(all_nic_names));
        goto err;
    }

    /* 3. filter out by topo */
    if (ctx.mnic_type == ATL_MNIC_NONE) {
        topo_provs.push_back(fi_dupinfo(name_provs[0]));
    }
    else {
        struct fid_nic* nic = nullptr;
        for (idx = 0; idx < name_provs.size(); idx++) {
            prov_iter = name_provs[idx];
            LOG_DEBUG("topo filter: check nic ", atl_ofi_get_nic_name(prov_iter));
            nic = prov_iter->nic;

            LOG_DEBUG("topo filter: check nic ",
                      atl_ofi_get_nic_name(prov_iter),
                      ", has nic_attr ",
                      (nic != nullptr));

            if (!atl_ofi_nic_already_used(prov_iter, topo_provs)) {
                int is_local = atl_ofi_is_nic_local(prov_iter);
                LOG_DEBUG(
                    "topo filter: nic ", atl_ofi_get_nic_name(prov_iter), ", is_local ", is_local);
                if (ctx.mnic_type == ATL_MNIC_GLOBAL ||
                    (ctx.mnic_type == ATL_MNIC_LOCAL && is_local)) {
                    LOG_DEBUG("topo filter: found suitable nic ", atl_ofi_get_nic_name(prov_iter));
                    topo_provs.push_back(fi_dupinfo(prov_iter));
                }
            }
            else {
                LOG_DEBUG("topo filter: nic ", atl_ofi_get_nic_name(prov_iter), " already used");
            }
        }
    }

    if (topo_provs.empty()) {
        LOG_ERROR("topo filter: can not find network providers, mnic_type ", ctx.mnic_type);
        goto err;
    }

    /* 4. reorder according to desired offset */
    if (ctx.mnic_offset == ATL_MNIC_OFFSET_LOCAL_PROC_IDX) {
        prov_offset = coord.local_idx % topo_provs.size();
    }
    LOG_DEBUG("rotate: prov_offset ", prov_offset, ", vec_size ", topo_provs.size());
    std::rotate(topo_provs.begin(), topo_provs.begin() + prov_offset, topo_provs.end());

    /* 5. filter out by count */
    for (idx = 0; idx < topo_provs.size(); idx++) {
        prov_iter = topo_provs[idx];
        LOG_DEBUG("count filter: check nic ", atl_ofi_get_nic_name(prov_iter));
        if (final_provs.size() < ctx.mnic_count) {
            LOG_DEBUG("count filter: found suitable nic ",
                      atl_ofi_get_nic_name(prov_iter),
                      ", nic idx ",
                      final_provs.size());
            final_provs.push_back(fi_dupinfo(prov_iter));
        }
        else {
            break;
        }
    }

    if (final_provs.empty()) {
        LOG_ERROR("count filter: can not find network providers, mnic_count ", ctx.mnic_count);
        goto err;
    }

    /* 6. create network providers */
    LOG_INFO("found ", final_provs.size(), " nic(s) according to all filters");
    ctx.nw_prov_count = final_provs.size();
    if (ep_names.size() < ctx.nw_prov_count + ctx.nw_prov_first_idx) {
        ep_names.resize(ctx.nw_prov_count + ctx.nw_prov_first_idx);
    }
    for (idx = 0; idx < ctx.nw_prov_count; idx++) {
        prov_idx = ctx.nw_prov_first_idx + idx;
        prov = &ctx.provs[prov_idx];
        prov->idx = prov_idx;
        prov->is_shm = 0;
        ATL_CALL(
            atl_ofi_prov_init(ctx, coord, final_provs[idx], prov, attr, pmi, ep_names[prov->idx]),
            goto err);
    }

exit:
    for (idx = 0; idx < final_provs.size(); idx++) {
        if (final_provs[idx])
            fi_freeinfo(final_provs[idx]);
    }

    for (idx = 0; idx < topo_provs.size(); idx++) {
        if (topo_provs[idx])
            fi_freeinfo(topo_provs[idx]);
    }

    for (idx = 0; idx < name_provs.size(); idx++) {
        if (name_provs[idx])
            fi_freeinfo(name_provs[idx]);
    }

    fi_freeinfo(prov_list);

    ctx.prov_count += ctx.nw_prov_count;

    return ret;

err:
    if (log_on_error) {
        LOG_ERROR("can not open network providers");
    }
    else {
        LOG_DEBUG("can not open network providers");
    }
    ret = ATL_STATUS_FAILURE;
    goto exit;
}

void atl_ofi_init_req(atl_req_t& req, atl_ofi_prov_ep_t* prov_ep, struct fid_ep* fi_ep) {
    atl_ofi_req_t* ofi_req = ((atl_ofi_req_t*)req.internal);
    ofi_req->prov_ep = prov_ep;
    ofi_req->fi_ep = fi_ep;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;
    req.is_completed = 0;
}

std::string get_shm_filename(std::string filename) {
    uid_t uid = getuid();
    std::stringstream ss;
    const char* pals_apid = getenv("PALS_APID");
    if (!pals_apid) {
        LOG_WARN("PALS_APID is not set, using default value");
        pals_apid = "default";
    }

    ss << filename << "-" << pals_apid << "-" << std::to_string(uid);
    return ss.str();
}

void shm_barrier(void* mem, int local_size, int comm_id) {
    // Interpret mem as ShmBarrierAllComms
    ShmBarrierAllComms* all_comms = reinterpret_cast<ShmBarrierAllComms*>(mem);

    // Check comm_id is in range
    if (comm_id < 0 || comm_id >= MAX_COMM) {
        std::cerr << "Error: comm_id out of range " << comm_id << std::endl;
        exit(-1);
    }

    // Each communicator uses its own ShmBarrierData slot
    ShmBarrierData* d = &all_comms->comm_barriers[comm_id];

    std::atomic<int>& barrier_counter = d->barrier_counter;
    std::atomic<int>& call_count = d->call_count;

    // Light sleep to avoid tight spinning
    usleep(1000);

    // Current round
    int current_round = call_count.load(std::memory_order_acquire);

    // Increment barrier_counter
    int old_val = barrier_counter.fetch_add(1, std::memory_order_acq_rel);

    // If we are the last process to arrive in this round
    if (old_val + 1 == local_size) {
        // Reset for next round
        barrier_counter.store(0, std::memory_order_release);
        // Indicate next round
        call_count.fetch_add(1, std::memory_order_release);
    }
    else {
        // Wait until call_count increments
        while (call_count.load(std::memory_order_acquire) == current_round) {
            usleep(1000); // short sleep to reduce CPU usage
        }
    }
}

atl_status_t setup_shared_memory(std::string shm_name,
                                 int local_size,
                                 bool is_node_root,
                                 size_t length,
                                 void** out_shared_memory,
                                 int my_comm_id) {
    LOG_DEBUG("Shared memory filename: " + shm_name);
    int fd = -1;

    if (is_node_root) {
        // Root process creates the shared memory file
        fd = open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) {
            LOG_ERROR("Failed to open shared memory file: ", strerror(errno));
            return ATL_STATUS_FAILURE;
        }
        if (ftruncate(fd, length) != 0) {
            LOG_ERROR("Failed to set size of shared memory file: ", strerror(errno));
            close(fd);
            return ATL_STATUS_FAILURE;
        }
    }
    else {
        // Non-root processes open the existing shared memory file with retries
        struct stat file_stat;
        int retry_count = 0;
        constexpr int max_retries = 20;
        constexpr int backoff_base = 10000;

        while (retry_count < max_retries) {
            if (stat(shm_name.c_str(), &file_stat) == 0) {
                fd = open(shm_name.c_str(), O_RDWR, 0666);
                if (fd >= 0)
                    break;
            }
            usleep(backoff_base * (1 << retry_count));
            retry_count++;
        }
        if (fd < 0) {
            LOG_ERROR("Non-root rank failed to open shared memory file: ", strerror(errno));
            return ATL_STATUS_FAILURE;
        }
    }

    // Map shared memory
    void* shared_memory = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (shared_memory == MAP_FAILED) {
        LOG_ERROR("Failed to map shared memory: ", strerror(errno));
        close(fd);
        return ATL_STATUS_FAILURE;
    }
    close(fd);

    auto* bar_mem = static_cast<barrier_mem_t*>(shared_memory);
    if (is_node_root) {
        bar_mem->all_comms.comm_barriers[my_comm_id].barrier_counter.store(
            0, std::memory_order_relaxed);
        bar_mem->all_comms.comm_barriers[my_comm_id].call_count.store(0, std::memory_order_release);
        LOG_DEBUG("Barrier counter initialized by node root");
    }

    shm_barrier((void*)&bar_mem->all_comms, local_size, my_comm_id);

    if (is_node_root) {
        unlink(shm_name.c_str());
    }
    shm_barrier((void*)&bar_mem->all_comms, local_size, my_comm_id);

    *out_shared_memory = shared_memory;
    return ATL_STATUS_SUCCESS;
}

atl_status_t fetch_and_populate_remote_data(void* shared_memory,
                                            size_t local_size,
                                            size_t global_size,
                                            int global_rank,
                                            int local_rank,
                                            size_t total_named_eps,
                                            size_t addr_len,
                                            size_t local_addr_offset,
                                            size_t global_addr_offset,
                                            char* shm_base,
                                            size_t length) {
    // Calculate the number of nodes and node_id
    int number_of_nodes = global_size / local_size;
    int node_id = global_rank / local_size;

    // Fetch data for the corresponding rank on each remote node
    for (int nid = 0; nid < number_of_nodes; nid++) {
        if (nid == node_id) {
            // Skip the local node
            continue;
        }

        // Compute the global rank on the remote node corresponding to this local rank
        int remote_rank = nid * local_size + local_rank;
        if (remote_rank == global_rank) {
            // Skip our own rank
            continue;
        }

        std::string key_str = std::string("RANK_EP_ADDRS_") + std::to_string(remote_rank);
        pmix_value_t* ret_value = nullptr;

        LOG_DEBUG("PMIx GET key: " + key_str + ", remote_rank: " + std::to_string(remote_rank) +
                  ", node_id: " + std::to_string(node_id) + ", nid: " + std::to_string(nid));

        // Fetch the data for the remote rank
        ccl_pmix::get(ccl::global_proc.nspace, remote_rank, key_str, &ret_value);

        if (!ret_value || ret_value->type != PMIX_BYTE_OBJECT) {
            LOG_ERROR("PMIx Key not found or invalid: ", key_str);
            if (ret_value)
                PMIX_VALUE_RELEASE(ret_value);
            munmap(shared_memory, length);
            return ATL_STATUS_FAILURE;
        }

        // Write fetched data into the global portion of shared memory
        char* dest = shm_base + global_addr_offset + (remote_rank * total_named_eps * addr_len);
        std::memcpy(dest, ret_value->data.bo.bytes, ret_value->data.bo.size);
        PMIX_VALUE_RELEASE(ret_value);
    }

    // Populate the global portion for local node ranks, including our own
    for (size_t local_node_rank = 0; local_node_rank < local_size; local_node_rank++) {
        int lr_global_rank = node_id * local_size + local_node_rank;

        // Copy from local portion to global portion for the local node's ranks
        char* src = shm_base + local_addr_offset + (local_node_rank * total_named_eps * addr_len);
        char* dest = shm_base + global_addr_offset + (lr_global_rank * total_named_eps * addr_len);
        std::memcpy(dest, src, total_named_eps * addr_len);
    }

    return ATL_STATUS_SUCCESS;
}
