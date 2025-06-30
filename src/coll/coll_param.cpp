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
#include <numeric>

#include "coll/coll_util.hpp"
#include "coll/coll_param.hpp"
#include "common/global/global.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

#define COPY_COMMON_OP_ATTRS(from, to) \
    to->priority = from.get<ccl::operation_attr_id::priority>(); \
    to->synchronous = from.get<ccl::operation_attr_id::synchronous>(); \
    to->to_cache = (from.get<ccl::operation_attr_id::match_id>().length()) \
                       ? from.get<ccl::operation_attr_id::to_cache>() \
                       : false; \
    to->match_id = from.get<ccl::operation_attr_id::match_id>(); \
    if (to->to_cache != from.get<ccl::operation_attr_id::to_cache>()) \
        LOG_INFO("collective caching is requested but no match_id is provided, disable caching");

ccl_coll_attr::ccl_coll_attr(const ccl::allgather_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
}

ccl_coll_attr::ccl_coll_attr(const ccl::allgatherv_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
}

ccl_coll_attr::ccl_coll_attr(const ccl::allreduce_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);

    reduction_fn = attr.get<ccl::allreduce_attr_id::reduction_fn>().get();
}

ccl_coll_attr::ccl_coll_attr(const ccl::alltoall_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
}

ccl_coll_attr::ccl_coll_attr(const ccl::alltoallv_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
}

ccl_coll_attr::ccl_coll_attr(const ccl::barrier_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
}

ccl_coll_attr::ccl_coll_attr(const ccl::broadcast_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
}

ccl_coll_attr::ccl_coll_attr(const ccl::pt2pt_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
    group_id = attr.get<ccl::pt2pt_attr_id::group_id>();
}

ccl_coll_attr::ccl_coll_attr(const ccl::reduce_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
    reduction_fn = attr.get<ccl::reduce_attr_id::reduction_fn>().get();
}

ccl_coll_attr::ccl_coll_attr(const ccl::reduce_scatter_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
    reduction_fn = attr.get<ccl::reduce_scatter_attr_id::reduction_fn>().get();
}

std::string ccl_coll_attr::to_string() const {
    std::stringstream ss;

    ss << "{ "
       << "priority: " << priority << ", sync: " << synchronous << ", to_cache: " << to_cache
       << ", match_id: " << (!match_id.empty() ? match_id : "<empty>");

    if (is_vector_buf) {
        ss << ", vector_buf";
    }

#ifdef CCL_ENABLE_SYCL
    if (is_sycl_buf) {
        ss << ", sycl_buf";
    }
#endif // CCL_ENABLE_SYCL

    ss << " }";

    return ss.str();
}

void ccl_coll_param::copy(const ccl_coll_param& other) {
    ctype = other.ctype;
    hint_algo = other.hint_algo;
    send_buf = other.send_buf;
    recv_buf = other.recv_buf;
    send_bufs = other.send_bufs;
    recv_bufs = other.recv_bufs;
    send_dev_bufs = other.send_dev_bufs;
    recv_dev_bufs = other.recv_dev_bufs;
    send_scale_out_bufs = other.send_scale_out_bufs;
    recv_scale_out_bufs = other.recv_scale_out_bufs;
    send_counts = other.send_counts;
    recv_counts = other.recv_counts;
    send_count = other.send_count;
    count = other.count;
    dtype = other.dtype;
    reduction = other.reduction;
    root = other.root;
    comm = other.comm;
    stream = other.stream;
    peer_rank = other.peer_rank;
    is_scaleout = other.is_scaleout;
    is_hmem_enabled = other.is_hmem_enabled;
    is_validate = other.is_validate;
    is_pt2pt = other.is_pt2pt;
    copy_deps(other.deps);

    if (is_validate) {
        validate();
    }
}

ccl_coll_param::ccl_coll_param(bool in_is_validate) {
    ctype = ccl_coll_last_value;
    send_bufs.reserve(1);
    recv_bufs.reserve(1);
    send_counts.reserve(1);
    recv_counts.reserve(1);
    stream = nullptr;
    comm = nullptr;
    is_scaleout = false;
    is_hmem_enabled = false;
    is_validate = in_is_validate;
    is_pt2pt = false;
}

ccl_coll_param::ccl_coll_param(const ccl_coll_param& other) {
    copy(other);
}

std::string ccl_coll_param::to_string() const {
    std::stringstream ss;

    ss << "{ ";
    ss << "coll: " << ccl_coll_type_to_str(ctype);

    if (!send_bufs.empty()) {
        ss << ", sb: " << get_send_buf() << ", sc: "
           << std::accumulate(
                  send_counts.begin(), send_counts.end(), ccl::utils::initial_count_value);
    }

    if (!recv_bufs.empty()) {
        ss << ", rb: " << get_recv_buf() << ", rc: "
           << std::accumulate(
                  recv_counts.begin(), recv_counts.end(), ccl::utils::initial_count_value);
    }

    if (ctype != ccl_coll_barrier) {
        ss << ", dt: " << ccl::global_data::get().dtypes->name(dtype);
    }

    if (ctype == ccl_coll_allreduce || ctype == ccl_coll_reduce ||
        ctype == ccl_coll_reduce_scatter) {
        ss << ", rt: " << ccl_reduction_to_str(reduction);
    }

    if (ctype == ccl_coll_bcast || ctype == ccl_coll_broadcast || ctype == ccl_coll_reduce) {
        ss << ", root: " << root;
    }

    ss << ", comm: ";
    if (comm)
        ss << "{ rank: " << comm->rank() << ", size: " << comm->size() << " }";
    else
        ss << "null";

#ifdef CCL_ENABLE_SYCL
    if (stream)
        ss << ", stream: " << stream->to_string();
#endif // CCL_ENABLE_SYCL

    if (!deps.empty())
        ss << ", deps: " << deps.size();

    ss << " }";

    return ss.str();
}

void* ccl_coll_param::get_send_buf(size_t idx, ccl_coll_param::buf_type type) const {
    auto& vec = (type == ccl_coll_param::buf_type::regular) ? send_bufs : send_dev_bufs;
    CCL_THROW_IF_NOT(idx < vec.size() || (ctype == ccl_coll_last_value && idx == vec.size()),
                     "coll ",
                     ctype,
                     ", unexpected idx ",
                     idx);
    return vec[idx];
}

void* ccl_coll_param::get_recv_buf(size_t idx, ccl_coll_param::buf_type type) const {
    auto& vec = (type == ccl_coll_param::buf_type::regular) ? recv_bufs : recv_dev_bufs;
    CCL_THROW_IF_NOT(idx < vec.size(), "coll ", ctype, ", unexpected idx ", idx);
    return vec[idx];
}

void* ccl_coll_param::get_send_buf_ptr(size_t idx, ccl_coll_param::buf_type type) const {
    auto& vec = (type == ccl_coll_param::buf_type::regular) ? send_bufs : send_dev_bufs;
    CCL_THROW_IF_NOT(idx < vec.size(), "coll ", ctype, ", unexpected idx ", idx);
    void* res = (void*)(&vec[idx]);
    return res;
}

void* ccl_coll_param::get_recv_buf_ptr(size_t idx, ccl_coll_param::buf_type type) const {
    auto& vec = (type == ccl_coll_param::buf_type::regular) ? recv_bufs : recv_dev_bufs;
    CCL_THROW_IF_NOT(idx < vec.size(), "coll ", ctype, ", unexpected idx ", idx);
    void* res = (void*)(&vec[idx]);
    return res;
}

size_t ccl_coll_param::get_send_count(size_t idx) const {
    CCL_THROW_IF_NOT(
        idx < send_counts.size() || (ctype == ccl_coll_last_value && idx == send_counts.size()),
        "coll ",
        ctype,
        ", unexpected idx ",
        idx);
    return send_counts[idx];
}

size_t ccl_coll_param::get_recv_count(size_t idx) const {
    CCL_THROW_IF_NOT(idx < recv_counts.size(), "coll ", ctype, ", unexpected idx ", idx);
    return recv_counts[idx];
}

bool ccl_coll_param::is_inplace(buf_type type) const {
    if (ctype == ccl_coll_barrier || ctype == ccl_coll_bcast) {
        return true;
    }

    void* send_buf_ptr = nullptr;
    void* recv_buf_ptr = nullptr;

    if ((ctype == ccl_coll_alltoall || ctype == ccl_coll_alltoallv) && (send_bufs.size() > 1)) {
        send_buf_ptr = get_send_buf(comm->rank(), type);
    }
    else {
        send_buf_ptr = get_send_buf(0, type);
    }

    bool scaleout_hmem_enabled = is_scaleout && is_hmem_enabled && ctype == ccl_coll_allgatherv;

    if (scaleout_hmem_enabled && recv_scale_out_bufs.size() > 1) {
        recv_buf_ptr = recv_scale_out_bufs[comm->rank()].get_ptr();
    }
    else if ((ctype == ccl_coll_allgather || ctype == ccl_coll_allgatherv ||
              ctype == ccl_coll_alltoall || ctype == ccl_coll_alltoallv) &&
             (recv_bufs.size() > 1)) {
        recv_buf_ptr = get_recv_buf(comm->rank(), type);
    }
    else {
        recv_buf_ptr = get_recv_buf(0, type);
    }

    if ((scaleout_hmem_enabled && recv_scale_out_bufs.size() == 1) ||
        ((ctype == ccl_coll_allgatherv || ctype == ccl_coll_allgather) && recv_bufs.size() == 1)) {
        return ccl::is_allgatherv_inplace(send_buf_ptr,
                                          get_send_count(),
                                          recv_buf_ptr,
                                          recv_counts.data(),
                                          dtype.size(),
                                          comm->rank(),
                                          comm->size());
    }

    if (ctype == ccl_coll_reduce_scatter) {
        return ccl::is_reduce_scatter_inplace(
            send_buf_ptr, recv_buf_ptr, get_recv_count(), dtype.size(), comm->rank(), comm->size());
    }

    return (send_buf_ptr && (send_buf_ptr == recv_buf_ptr)) ? true : false;
}

std::vector<void*> ccl_coll_param::get_all_non_zero_bufs() const {
    std::vector<void*> bufs;
    switch (ctype) {
        case ccl_coll_alltoallv: {
            /*
                if the sum of the counts is 0 this means that the buf pointer could be anything,
                including nullptr and invalid pointer
                don't validate nor dereference it
            */
            if (std::accumulate(
                    send_counts.begin(), send_counts.end(), ccl::utils::initial_count_value) > 0) {
                bufs.push_back(get_send_buf());
            }

            if (std::accumulate(
                    recv_counts.begin(), recv_counts.end(), ccl::utils::initial_count_value) > 0) {
                bufs.push_back(get_recv_buf());
            }
            break;
        }
        case ccl_coll_allgatherv: {
            if (get_send_count()) {
                bufs.push_back(get_send_buf());
            }

            if (std::accumulate(
                    recv_counts.begin(), recv_counts.end(), ccl::utils::initial_count_value) > 0) {
                if (recv_bufs.size() == 1) {
                    bufs.push_back(get_recv_buf());
                }
                else {
                    for (size_t idx = 0; idx < recv_counts.size(); idx++) {
                        if (recv_counts[idx])
                            bufs.push_back(get_recv_buf(idx));
                    }
                }
            }
            break;
        }
        case ccl_coll_allreduce:
        case ccl_coll_alltoall:
        case ccl_coll_allgather:
        case ccl_coll_bcast:
        case ccl_coll_broadcast:
        case ccl_coll_reduce:
        case ccl_coll_reduce_scatter:
            if (get_send_count()) {
                bufs.push_back(get_send_buf());
            }

            if (get_recv_count()) {
                bufs.push_back(get_recv_buf());
            }
            break;
        default: break;
    }
    return bufs;
}

void ccl_coll_param::validate() const {
    if (ctype > ccl_coll_last_regular) {
        return;
    }

    if (ctype == ccl_coll_barrier) {
        return;
    }

    LOG_TRACE("validate coll_param, coll: ", ccl_coll_type_to_str(ctype));
    CCL_THROW_IF_NOT(
        !send_counts.empty(), "empty send_counts: ctype: ", ccl_coll_type_to_str(ctype));
    CCL_THROW_IF_NOT(
        !recv_counts.empty(), "empty recv_counts ctype: ", ccl_coll_type_to_str(ctype));

    CCL_THROW_IF_NOT(!send_bufs.empty(), "empty send_bufs");
    CCL_THROW_IF_NOT(!recv_bufs.empty(), "empty recv_bufs");

    switch (ctype) {
        case ccl_coll_alltoallv: {
            CCL_THROW_IF_NOT(
                (send_bufs.size() == 1) || (static_cast<int>(send_bufs.size()) == comm->size()),
                "send_bufs size ",
                send_bufs.size(),
                ", comm size ",
                comm->size());

            CCL_THROW_IF_NOT(
                (recv_bufs.size() == 1) || (static_cast<int>(recv_bufs.size()) == comm->size()),
                "recv_bufs size ",
                recv_bufs.size(),
                ", comm size ",
                comm->size());

            CCL_THROW_IF_NOT(send_counts[comm->rank()] == recv_counts[comm->rank()],
                             "send_count[rank] ",
                             send_counts[comm->rank()],
                             ", recv_counts[rank] ",
                             recv_counts[comm->rank()]);

            if (send_counts.size() > 1) {
                CCL_THROW_IF_NOT(static_cast<int>(send_counts.size()) == comm->size(),
                                 "send_counts size ",
                                 send_counts.size(),
                                 ", comm size ",
                                 comm->size());
            }

            if (recv_counts.size() > 1) {
                CCL_THROW_IF_NOT(static_cast<int>(recv_counts.size()) == comm->size(),
                                 "recv_counts size ",
                                 recv_counts.size(),
                                 ", comm size ",
                                 comm->size());
            }
            break;
        }
        case ccl_coll_allgatherv: {
            CCL_THROW_IF_NOT(
                (recv_bufs.size() == 1) || (static_cast<int>(recv_bufs.size()) == comm->size()),
                "recv_bufs size ",
                recv_bufs.size(),
                ", comm size ",
                comm->size());

            CCL_THROW_IF_NOT(
                send_counts.size() == 1, "unexpected send_counts size ", send_counts.size());

            CCL_THROW_IF_NOT(get_send_count() == recv_counts[comm->rank()],
                             "send_count ",
                             get_send_count(),
                             ", recv_counts[rank] ",
                             recv_counts[comm->rank()]);

            if (recv_counts.size() > 1) {
                CCL_THROW_IF_NOT(static_cast<int>(recv_counts.size()) == comm->size(),
                                 "recv_counts size ",
                                 recv_counts.size(),
                                 ", comm size ",
                                 comm->size());
            }
            break;
        }
        case ccl_coll_allreduce:
        case ccl_coll_alltoall:
        case ccl_coll_allgather:
        case ccl_coll_bcast:
        case ccl_coll_broadcast:
        case ccl_coll_reduce:
        case ccl_coll_reduce_scatter:
            CCL_THROW_IF_NOT(send_bufs.size() == send_counts.size(),
                             "send_bufs size ",
                             send_bufs.size(),
                             ", send_counts size ",
                             send_counts.size());

            CCL_THROW_IF_NOT(recv_bufs.size() == recv_counts.size(),
                             "recv_bufs size ",
                             recv_bufs.size(),
                             ", recv_counts size ",
                             recv_counts.size());

            if (ctype == ccl_coll_bcast) {
                CCL_THROW_IF_NOT(get_send_buf() == get_recv_buf(),
                                 "send_buf ",
                                 get_send_buf(),
                                 ", recv_buf ",
                                 get_recv_buf());
            }

            if (ctype == ccl_coll_allreduce || ctype == ccl_coll_reduce_scatter ||
                ctype == ccl_coll_reduce) {
                if (reduction == ccl::reduction::avg) {
                    // CCL_THROW_IF_NOT produce and error message which CI interprets as a failed test,
                    // however in some cases we want to throw exception, catch it and skip the average test.
                    CCL_THROW("average operation is not supported for the scheduler path");
                }
            }

            CCL_THROW_IF_NOT(
                send_counts.size() == 1, "unexpected send_counts size ", send_counts.size());

            if (ctype == ccl_coll_reduce_scatter) {
                CCL_THROW_IF_NOT(get_send_count() == get_recv_count() * comm->size(),
                                 "send_count ",
                                 get_send_count(),
                                 ", recv_count * comm_size ",
                                 get_recv_count() * comm->size());
            }
            else {
                CCL_THROW_IF_NOT(get_send_count() == get_recv_count(),
                                 "send_count ",
                                 get_send_count(),
                                 ", recv_count ",
                                 get_recv_count());
            }
            break;
        default: break;
    }
}

void ccl_coll_param::copy_deps(const std::vector<ccl::event>& d) {
#ifdef CCL_ENABLE_SYCL
    deps.clear();
#endif // CCL_ENABLE_SYCL
    // (modify this comment for other when new operations are integrated) Group api for send/recv ops
    // are not expected to use of vector<ccl::event>& deps, that's why we skip.
    // E.g. one of the issue can be that since deps are const ref, its lifetime is limited,
    // when user passes it to the group api calls, but the group api can be executed later
    // deps can be already destroyed.
    if (!group_impl::is_group_active) {
#ifdef CCL_ENABLE_SYCL
        for (size_t idx = 0; idx < d.size(); idx++) {
            try {
                auto sycl_event = d[idx].get_native();
                deps.push_back(ccl::create_event(sycl_event));
            }
            catch (ccl::exception&) {
            }
        }
#else // CCL_ENABLE_SYCL
        CCL_THROW_IF_NOT(d.size() == 0, "host deps are not supported yet");
#endif // CCL_ENABLE_SYCL
    }
}

void ccl_coll_param::set_common_fields(ccl::datatype d,
                                       ccl_comm* c,
                                       const ccl_stream* s,
                                       const std::vector<ccl::event>& ds) {
    dtype = ccl::global_data::get().dtypes->get(d);
    comm = c;
    stream = (ccl_stream*)s;
    copy_deps(ds);
}

ccl_coll_param ccl_coll_param::create_allgather_param(const void* send_buf,
                                                      void* recv_buf,
                                                      size_t count,
                                                      ccl::datatype dtype,
                                                      const ccl_coll_attr& attr,
                                                      ccl_comm* comm,
                                                      const ccl_stream* stream,
                                                      const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_allgather;
    param.send_bufs.push_back((void*)send_buf);
    param.send_counts.push_back(count);
    param.recv_bufs.push_back(recv_buf);
    param.recv_counts.push_back(count);
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_allgatherv_param(const void* send_buf,
                                                       size_t send_count,
                                                       void* recv_buf,
                                                       const size_t* recv_counts,
                                                       ccl::datatype dtype,
                                                       const ccl_coll_attr& attr,
                                                       ccl_comm* comm,
                                                       const ccl_stream* stream,
                                                       const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_allgatherv;
    param.send_bufs.push_back((void*)send_buf);
    param.send_counts.push_back(send_count);
    if (attr.is_vector_buf) {
        param.recv_bufs.assign((void**)recv_buf, (void**)recv_buf + comm->size());
    }
    else {
        param.recv_bufs.push_back(recv_buf);
    }
    param.recv_counts.assign((size_t*)recv_counts, (size_t*)recv_counts + comm->size());
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_allreduce_param(const void* send_buf,
                                                      void* recv_buf,
                                                      size_t count,
                                                      ccl::datatype dtype,
                                                      ccl::reduction reduction,
                                                      const ccl_coll_attr& attr,
                                                      ccl_comm* comm,
                                                      const ccl_stream* stream,
                                                      const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_allreduce;
    param.count = count;
    param.send_bufs.push_back((void*)send_buf);
    param.send_counts.push_back(count);
    param.recv_bufs.push_back(recv_buf);
    param.recv_counts.push_back(count);
    param.reduction = reduction;
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_alltoall_param(const void* send_buf,
                                                     void* recv_buf,
                                                     size_t count,
                                                     ccl::datatype dtype,
                                                     const ccl_coll_attr& attr,
                                                     ccl_comm* comm,
                                                     const ccl_stream* stream,
                                                     const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_alltoall;
    param.send_bufs.push_back((void*)send_buf);
    param.send_counts.push_back(count);
    param.recv_bufs.push_back(recv_buf);
    param.recv_counts.push_back(count);
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_alltoallv_param(const void* send_buf,
                                                      const size_t* send_counts,
                                                      void* recv_buf,
                                                      const size_t* recv_counts,
                                                      ccl::datatype dtype,
                                                      const ccl_coll_attr& attr,
                                                      ccl_comm* comm,
                                                      const ccl_stream* stream,
                                                      const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_alltoallv;
    //send
    if (attr.is_vector_buf) {
        param.send_bufs.assign((void**)send_buf, (void**)send_buf + comm->size());
    }
    else {
        param.send_bufs.push_back((void*)send_buf);
    }
    param.send_counts.assign((size_t*)send_counts, (size_t*)send_counts + comm->size());

    //recv
    if (attr.is_vector_buf) {
        param.recv_bufs.assign((void**)recv_buf, (void**)recv_buf + comm->size());
    }
    else {
        param.recv_bufs.push_back(recv_buf);
    }
    param.recv_counts.assign((size_t*)recv_counts, (size_t*)recv_counts + comm->size());
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_barrier_param(ccl_comm* comm,
                                                    const ccl_stream* stream,
                                                    const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_barrier;
    param.send_counts.push_back(0);
    param.recv_counts.push_back(0);
    param.set_common_fields(ccl::datatype::int8, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_broadcast_param(void* buf,
                                                      size_t count,
                                                      ccl::datatype dtype,
                                                      int root,
                                                      const ccl_coll_attr& attr,
                                                      ccl_comm* comm,
                                                      const ccl_stream* stream,
                                                      const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_bcast;
    param.send_bufs.push_back(buf);
    param.send_counts.push_back(count);
    param.recv_bufs.push_back(buf);
    param.recv_counts.push_back(count);
    param.root = root;
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_broadcast_param(void* send_buf,
                                                      void* recv_buf,
                                                      size_t count,
                                                      ccl::datatype dtype,
                                                      int root,
                                                      const ccl_coll_attr& attr,
                                                      ccl_comm* comm,
                                                      const ccl_stream* stream,
                                                      const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_broadcast;
    param.send_bufs.push_back(send_buf);
    param.send_counts.push_back(count);
    param.recv_bufs.push_back(recv_buf);
    param.recv_counts.push_back(count);
    param.root = root;
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_reduce_param(const void* send_buf,
                                                   void* recv_buf,
                                                   size_t count,
                                                   ccl::datatype dtype,
                                                   ccl::reduction reduction,
                                                   int root,
                                                   const ccl_coll_attr& attr,
                                                   ccl_comm* comm,
                                                   const ccl_stream* stream,
                                                   const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_reduce;
    param.send_bufs.push_back((void*)send_buf);
    param.send_counts.push_back(count);
    param.recv_bufs.push_back(recv_buf);
    param.recv_counts.push_back(count);
    param.reduction = reduction;
    param.root = root;
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_reduce_scatter_param(const void* send_buf,
                                                           void* recv_buf,
                                                           size_t recv_count,
                                                           ccl::datatype dtype,
                                                           ccl::reduction reduction,
                                                           const ccl_coll_attr& attr,
                                                           ccl_comm* comm,
                                                           const ccl_stream* stream,
                                                           const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_reduce_scatter;
    param.send_bufs.push_back((void*)send_buf);
    param.send_counts.push_back(comm->size() * recv_count);
    param.recv_bufs.push_back(recv_buf);
    param.recv_counts.push_back(recv_count);
    param.reduction = reduction;
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_recv_param(void* recv_buf,
                                                 size_t recv_count,
                                                 ccl::datatype dtype,
                                                 int peer_rank,
                                                 const ccl_coll_attr& attr,
                                                 ccl_comm* comm,
                                                 const ccl_stream* stream,
                                                 const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_recv;
    param.send_bufs.push_back(recv_buf);
    param.send_counts.push_back(recv_count);
    param.recv_bufs.push_back(recv_buf);
    param.recv_counts.push_back(recv_count);
    param.peer_rank = peer_rank;
    param.is_pt2pt = true;
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}

ccl_coll_param ccl_coll_param::create_send_param(const void* send_buf,
                                                 size_t send_count,
                                                 ccl::datatype dtype,
                                                 int peer_rank,
                                                 const ccl_coll_attr& attr,
                                                 ccl_comm* comm,
                                                 const ccl_stream* stream,
                                                 const std::vector<ccl::event>& deps) {
    ccl_coll_param param{};

    param.ctype = ccl_coll_send;
    param.send_bufs.push_back((void*)send_buf);
    param.send_counts.push_back(send_count);
    param.recv_bufs.push_back((void*)send_buf);
    param.recv_counts.push_back(send_count);
    param.peer_rank = peer_rank;
    param.is_pt2pt = true;
    param.set_common_fields(dtype, comm, stream, deps);
    param.validate();

    return param;
}
