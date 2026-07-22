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

#include "common/datatype/datatype.hpp"
#include "common/global/global.hpp"
#include "sched/cache/key.hpp"
#include "common/utils/enums.hpp"

#include <cstring>
#include <numeric>

std::map<ccl_cache_key_type, std::string> ccl_sched_key::key_type_names = {
    std::make_pair(ccl_cache_key_full, "full"),
    std::make_pair(ccl_cache_key_match_id, "match_id")
};

ccl_sched_key::ccl_sched_key(const ccl_coll_param& param, const ccl_coll_attr& attr) {
    set(param, attr);
}

void ccl_sched_key::set(const ccl_coll_param& param, const ccl_coll_attr& attr) {
    if (ccl::global_data::env().cache_key_type == ccl_cache_key_full) {
        /* to zerioize holes in memory layout */
        memset((void*)&f, 0, sizeof(ccl_sched_key_inner_fields));
    }

    f.reduction_fn = attr.reduction_fn;
    match_id = attr.match_id;

    f.ctype = param.ctype;
    f.dtype = param.dtype.idx();
    f.comm = param.comm;

    switch (f.ctype) {
        case ccl_coll_allgather: f.count1 = param.get_send_count(); break;
        case ccl_coll_allgatherv:
            f.count1 = param.get_send_count();
            vec1 = param.recv_counts;
            break;
        case ccl_coll_allreduce:
            f.count1 = param.get_send_count();
            f.reduction = param.reduction;
            break;
        case ccl_coll_alltoall: f.count1 = param.get_send_count(); break;
        case ccl_coll_alltoallv:
            vec1 = param.send_counts;
            vec2 = param.recv_counts;
            break;
        case ccl_coll_barrier: break;
        case ccl_coll_bcast:
        case ccl_coll_broadcast:
            f.count1 = param.get_send_count();
            f.root = param.root;
            break;
        case ccl_coll_reduce:
            f.count1 = param.get_send_count();
            f.reduction = param.reduction;
            f.root = param.root;
            break;
        case ccl_coll_reduce_scatter:
            f.count1 = param.get_send_count();
            f.reduction = param.reduction;
            break;
        case ccl_coll_scatter:
            f.count1 = param.get_send_count();
            f.root = param.root;
            break;
        case ccl_coll_recv:
            f.count1 = param.get_recv_count();
            f.peer_rank = param.peer_rank;
            f.group_id = param.group_id;
            break;
        case ccl_coll_send:
            f.count1 = param.get_send_count();
            f.peer_rank = param.peer_rank;
            f.group_id = param.group_id;
            break;
        default: CCL_THROW("unexpected coll_type ", f.ctype);
    }
}

bool ccl_sched_key::check(const ccl_coll_param& param, const ccl_coll_attr& attr) const {
    bool result = true;

    result &= (attr.reduction_fn == f.reduction_fn || param.ctype == f.ctype ||
               param.dtype == f.dtype || param.comm == f.comm);

    switch (f.ctype) {
        case ccl_coll_allgather: result &= (param.get_send_count() == f.count1); break;
        case ccl_coll_allgatherv:
            result &= (param.get_send_count() == f.count1 && param.recv_counts == vec1);
            break;
        case ccl_coll_allreduce:
            result &= (param.get_send_count() == f.count1 && param.reduction == f.reduction);
            break;
        case ccl_coll_alltoall: result &= (param.get_send_count() == f.count1); break;
        case ccl_coll_alltoallv:
            result &= (param.send_counts == vec1 && param.recv_counts == vec2);
            break;
        case ccl_coll_barrier: break;
        case ccl_coll_bcast:
        case ccl_coll_broadcast:
            result &= (param.get_send_count() == f.count1 && param.root == f.root);
            break;
        case ccl_coll_reduce:
            result &= (param.get_send_count() == f.count1 && param.reduction == f.reduction &&
                       param.root == f.root);
            break;
        case ccl_coll_reduce_scatter:
            result &= (param.get_send_count() == f.count1 && param.reduction == f.reduction);
            break;
        case ccl_coll_scatter:
            result &= (param.get_send_count() == f.count1 && param.root == f.root);
            break;
        case ccl_coll_recv:
            result &= (param.get_recv_count() == f.count1 && param.peer_rank == f.peer_rank &&
                       param.group_id == f.group_id);
            break;
        case ccl_coll_send:
            result &= (param.get_send_count() == f.count1 && param.peer_rank == f.peer_rank &&
                       param.group_id == f.group_id);
            break;
        default: CCL_THROW("unexpected coll_type ", f.ctype);
    }

    return result;
}

bool ccl_sched_key::operator==(const ccl_sched_key& k) const {
    bool are_fields_equal = 1;
    if (ccl::global_data::env().cache_key_type == ccl_cache_key_full) {
        are_fields_equal = !memcmp(&f, &(k.f), sizeof(ccl_sched_key_inner_fields));
        are_fields_equal &= (vec1 == k.vec1) ? 1 : 0;
        are_fields_equal &= (vec2 == k.vec2) ? 1 : 0;
    }

    bool are_keys_equal = are_fields_equal && !match_id.compare(k.match_id);

    LOG_DEBUG("are_keys_equal ", are_keys_equal);

    print();
    k.print();
    return are_keys_equal;
}

void ccl_sched_key::print() const {
    LOG_DEBUG("coll ",
              ccl_coll_type_to_str(f.ctype),
              ", dtype ",
              ccl::global_data::get().dtypes->name(f.dtype),
              ", reduction ",
              ccl_reduction_to_str(f.reduction),
              ", buf1 ",
              f.buf1,
              ", buf2 ",
              f.buf2,
              ", count1 ",
              f.count1,
              ", count2 ",
              f.count2,
              ", root ",
              f.root,
              ", comm ",
              f.comm,
              ", reduction_fn ",
              (void*)f.reduction_fn,
              ", vec1.size ",
              vec1.size(),
              ", vec2.size ",
              vec2.size(),
              ", match_id ",
              match_id);
}

size_t ccl_sched_key_hasher::operator()(const ccl_sched_key& k) const {
    if (k.has_hasher_result)
        return k.get_hasher_result();

    size_t hash_value = string_hasher(k.match_id);
    if (ccl::global_data::env().cache_key_type == ccl_cache_key_full) {
        /* TODO: improve hashing for vec fields to reduce probability of collisions
           e.g. sum(a[idx]*(idx+1)) */
        size_t vec1_sum = std::accumulate(k.vec1.begin(), k.vec1.end(), size_t(0));
        size_t vec2_sum = std::accumulate(k.vec2.begin(), k.vec2.end(), size_t(0));
        hash_value += k.f.ctype + ccl::utils::enum_to_underlying(k.f.dtype) +
                      ccl::utils::enum_to_underlying(k.f.reduction) + k.f.count1 + k.f.count2 +
                      k.f.root + (size_t)k.f.buf1 + (size_t)k.f.buf2 + (size_t)k.f.comm +
                      (size_t)k.f.reduction_fn + vec1_sum + vec2_sum;
    }

    const_cast<ccl_sched_key&>(k).set_hasher_result(hash_value);

    LOG_DEBUG("hash_value ", hash_value);
    k.print();

    return hash_value;
}
