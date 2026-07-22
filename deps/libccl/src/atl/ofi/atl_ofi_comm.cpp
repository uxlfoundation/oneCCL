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

#include "atl/ofi/atl_ofi_comm.hpp"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable_simple.h"
#include "atl/util/pm/pmi_rt/pmi_simple.h"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.h"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable_simple_internal.h"
#include "atl/ofi/atl_ofi.hpp"
#include "atl/ofi/atl_comp.hpp"
#include "exec/exec.hpp"

enum ofi_op_tag { allgatherv_tag = 0, barrier_tag, allreduce_tag, reduce_scatter_tag, bcast_tag };

static inline void ofi_send(void* sendbuf,
                            size_t len,
                            int dst,
                            atl_ofi_comm* comm,
                            size_t ep_idx,
                            uint64_t op_tag) {
    atl_status_t ret;
    atl_req send_req;
    do {
        ret = comm->send(ep_idx, sendbuf, len, dst, op_tag, send_req);
        CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "send failed");
        if (ret == ATL_STATUS_AGAIN) {
            ccl_yield(ccl::global_data::env().yield_type);
        }
    } while (ret == ATL_STATUS_AGAIN);
    while (!send_req.is_completed) {
        comm->poll(ep_idx);
        if (!send_req.is_completed) {
            CCL_THROW_IF_NOT(comm->check(ep_idx, send_req) != ATL_STATUS_FAILURE,
                             "check send failed");
        }
    }
}

static inline void ofi_recv(void* recvbuf,
                            size_t len,
                            int dst,
                            atl_ofi_comm* comm,
                            size_t ep_idx,
                            uint64_t op_tag) {
    atl_status_t ret;
    atl_req recv_req;
    do {
        ret = comm->recv(ep_idx, recvbuf, len, dst, op_tag, recv_req);
        CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "recv failed");
        if (ret == ATL_STATUS_AGAIN) {
            ccl_yield(ccl::global_data::env().yield_type);
        }
    } while (ret == ATL_STATUS_AGAIN);
    while (!recv_req.is_completed) {
        comm->poll(ep_idx);
        if (!recv_req.is_completed) {
            CCL_THROW_IF_NOT(comm->check(ep_idx, recv_req) != ATL_STATUS_FAILURE,
                             "check recv failed");
        }
    }
}

atl_ofi_comm::atl_ofi_comm() {
    pmi = std::shared_ptr<ipmi>(new pmi_simple());
    CCL_THROW_IF_NOT(init_transport(true) == ATL_STATUS_SUCCESS, "init transport failed");
}

atl_ofi_comm::atl_ofi_comm(std::shared_ptr<ikvs_wrapper> k) {
    pmi = std::shared_ptr<ipmi>(new pmi_simple());
    CCL_THROW_IF_NOT(init_transport(true) == ATL_STATUS_SUCCESS, "init transport failed");
}

atl_ofi_comm::atl_ofi_comm(int comm_size,
                           const std::vector<int>& ranks,
                           std::shared_ptr<ikvs_wrapper> k) {
    std::shared_ptr<internal_kvs> kvs;
    if ((kvs = std::dynamic_pointer_cast<internal_kvs>(k)) != nullptr) {
        pmi = std::shared_ptr<ipmi>(new pmi_resizable_simple_internal(comm_size, ranks, kvs));
    }
    else {
        pmi = std::shared_ptr<ipmi>(new pmi_resizable_simple(comm_size, ranks, k));
    }

    CCL_THROW_IF_NOT(init_transport(true) == ATL_STATUS_SUCCESS, "init transport failed");
}

atl_status_t atl_ofi_comm::barrier(size_t ep_idx, atl_req_t& req) {
    ssize_t ret = ATL_STATUS_SUCCESS;

    req.is_completed = false;
    atl_ofi_req_t* ofi_req = ((atl_ofi_req_t*)req.internal);

    if (size == 1) {
        ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;
        return ATL_STATUS_SUCCESS;
    }

    int tag_comm_id = (comm_id != atl_comm_id_storage::invalid_comm_id)
                          ? comm_id
                          : atl_comm_id_storage::max_comm_id;
    int tagc = tag_counter_barrier++;

    LOG_DEBUG("ofi_barrier: comm_rank: ",
              rank,
              ", comm_size: ",
              size,
              ", comm_id: ",
              comm_id,
              ", tag_comm_id: ",
              tag_comm_id,
              ", tag_counter: ",
              tagc);

    int src, dst;
    const int len = 1;
    char sendbuf[len], recvbuf[len];
    int mask = 0x1;
    while (mask < size) {
        dst = (rank + mask) % size;
        src = (rank + size - mask) % size;
        atl_req send_req, recv_req;
        uint64_t op_tag = tag_creator->create(rank, tag_comm_id, tagc, ofi_op_tag::barrier_tag);
        do {
            ret = send(ep_idx, sendbuf, len, dst, op_tag, send_req);
            CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "send failed");
            if (ret == ATL_STATUS_AGAIN) {
                ccl_yield(ccl::global_data::env().yield_type);
            }
        } while (ret == ATL_STATUS_AGAIN);
        op_tag = tag_creator->create(src, tag_comm_id, tagc, ofi_op_tag::barrier_tag);
        do {
            ret = recv(ep_idx, recvbuf, len, src, op_tag, recv_req);
            CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "recv failed");
            if (ret == ATL_STATUS_AGAIN) {
                ccl_yield(ccl::global_data::env().yield_type);
            }
        } while (ret == ATL_STATUS_AGAIN);
        while (!send_req.is_completed || !recv_req.is_completed) {
            poll(ep_idx);
            if (!send_req.is_completed) {
                CCL_THROW_IF_NOT(check(ep_idx, send_req) != ATL_STATUS_FAILURE,
                                 "check send failed");
            }
            if (!recv_req.is_completed) {
                CCL_THROW_IF_NOT(check(ep_idx, recv_req) != ATL_STATUS_FAILURE,
                                 "check recv failed");
            }
        }
        mask <<= 1;
    }

    LOG_DEBUG("ofi_barrier done: comm_rank: ",
              rank,
              ", comm_size: ",
              size,
              ", comm_id: ",
              comm_id,
              ", tag_comm_id: ",
              tag_comm_id,
              ", tag_counter: ",
              tagc);

    ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;
    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi_comm::bcast(size_t ep_idx, void* buf, size_t len, int root, atl_req_t& req) {
    int src, dst;

    req.is_completed = false;
    atl_ofi_req_t* ofi_req = ((atl_ofi_req_t*)req.internal);

    if (size == 1) {
        ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;
        return ATL_STATUS_SUCCESS;
    }

    int tag_comm_id = (comm_id != atl_comm_id_storage::invalid_comm_id)
                          ? comm_id
                          : atl_comm_id_storage::max_comm_id;
    int tagc = tag_counter_bcast++;

    LOG_DEBUG("ofi_bcast: comm_rank: ",
              rank,
              ", comm_size: ",
              size,
              ", len: ",
              len,
              ", comm_id: ",
              comm_id,
              ", tag_comm_id: ",
              tag_comm_id,
              ", tag_counter: ",
              tagc);

    // binomial tree, good for small message sizes
    int relative_rank = (rank >= root) ? rank - root : rank - root + size;
    int mask = 0x1;
    while (mask < size) {
        if (relative_rank & mask) {
            src = rank - mask;
            if (src < 0)
                src += size;
            uint64_t op_tag = tag_creator->create(src, tag_comm_id, tagc, ofi_op_tag::bcast_tag);
            ofi_recv(buf, len, src, this, ep_idx, op_tag);
            break;
        }
        mask <<= 1;
    }

    mask >>= 1;
    while (mask > 0) {
        if (relative_rank + mask < size) {
            dst = rank + mask;
            if (dst >= size)
                dst -= size;
            uint64_t op_tag = tag_creator->create(rank, tag_comm_id, tagc, ofi_op_tag::bcast_tag);
            ofi_send(buf, len, dst, this, ep_idx, op_tag);
        }
        mask >>= 1;
    }

    // to let user complete this operation through wait(req)
    req.is_completed = false;

    ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi_comm::allgatherv(size_t ep_idx,
                                      const void* send_buf,
                                      size_t send_len,
                                      void* recv_buf,
                                      const size_t* recv_lens,
                                      const size_t* offsets,
                                      atl_req_t& req) {
    std::vector<atl_req> send_reqs(size - 1);
    std::vector<atl_req> recv_reqs(size - 1);

    int tag_comm_id = (comm_id != atl_comm_id_storage::invalid_comm_id)
                          ? comm_id
                          : atl_comm_id_storage::max_comm_id;

    LOG_DEBUG("ofi_allgatherv: comm_rank: ",
              rank,
              ", comm_size: ",
              size,
              ", send_len: ",
              send_len,
              ", comm_id: ",
              comm_id,
              ", tag_comm_id: ",
              tag_comm_id,
              ", tag_counter: ",
              tag_counter);

    for (int peer = 0, req_idx = 0; peer < size; peer++) {
        if (peer == rank)
            continue;

        uint64_t op_tag = tag_creator->create(rank, tag_comm_id, tag_counter);
        // LOG_DEBUG("ofi_allgatherv: send: rank: ", rank,
        //     ", peer: ", peer,
        //     ", comm_id: ", comm_id,
        //     ", tag_comm_id: ", tag_comm_id,
        //     ", tag_counter: ", tag_counter,
        //     ", op_tag: ", op_tag);

        atl_status_t ret;

        do {
            ret = send(ep_idx, send_buf, send_len, peer, op_tag, send_reqs[req_idx]);
            CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "send failed");
            if (ret == ATL_STATUS_AGAIN) {
                ccl_yield(ccl::global_data::env().yield_type);
            }
        } while (ret == ATL_STATUS_AGAIN);

        op_tag = tag_creator->create(peer, tag_comm_id, tag_counter);
        // LOG_DEBUG("ofi_allgatherv: recv: rank: ", rank,
        //     ", peer: ", peer,
        //     ", comm_id: ", comm_id,
        //     ", tag_comm_id: ", tag_comm_id,
        //     ", tag_counter: ", tag_counter,
        //     ", op_tag: ", op_tag);

        do {
            ret = recv(ep_idx,
                       (char*)recv_buf + offsets[peer],
                       recv_lens[peer],
                       peer,
                       op_tag,
                       recv_reqs[req_idx]);
            CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "recv failed");
            if (ret == ATL_STATUS_AGAIN) {
                ccl_yield(ccl::global_data::env().yield_type);
            }
        } while (ret == ATL_STATUS_AGAIN);

        req_idx++;
    }

    if ((char*)recv_buf + offsets[rank] != send_buf) {
        memcpy((char*)recv_buf + offsets[rank], send_buf, recv_lens[rank]);
    }

    bool is_completed = false;
    while (!is_completed) {
        is_completed = true;
        poll(ep_idx);
        for (size_t i = 0; i < send_reqs.size(); i++) {
            if (!send_reqs[i].is_completed) {
                CCL_THROW_IF_NOT(check(ep_idx, send_reqs[i]) != ATL_STATUS_FAILURE,
                                 "check send failed");
                is_completed = false;
                break;
            }
            if (!recv_reqs[i].is_completed) {
                CCL_THROW_IF_NOT(check(ep_idx, recv_reqs[i]) != ATL_STATUS_FAILURE,
                                 "check recv failed");
                is_completed = false;
                break;
            }
        }
    }

    // to let user complete this operation through wait(req)
    req.is_completed = false;

    atl_ofi_req_t* ofi_req = ((atl_ofi_req_t*)req.internal);
    ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;

    LOG_DEBUG("ofi_allgatherv done: comm_rank: ",
              rank,
              ", comm_size: ",
              size,
              ", send_len: ",
              send_len,
              ", comm_id: ",
              comm_id,
              ", tag_comm_id: ",
              tag_comm_id,
              ", tag_counter: ",
              tag_counter);

    tag_counter++;

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi_comm::allreduce(size_t ep_idx,
                                     const void* send_buf,
                                     void* recv_buf,
                                     size_t len,
                                     atl_datatype_t dtype,
                                     atl_reduction_t op,
                                     atl_req_t& req) {
    // simple recursive double algorithm
    atl_status_t ret;
    int newrank, mask;
    int dsize = get_atl_datatype_size(dtype);
    size_t total_size = len * dsize;

    int tag_comm_id = (comm_id != atl_comm_id_storage::invalid_comm_id)
                          ? comm_id
                          : atl_comm_id_storage::max_comm_id;

    int tagc = tag_counter_allreduce++;

    LOG_DEBUG("ofi_allreduce: comm_rank: ",
              rank,
              ", comm_size: ",
              size,
              ", len: ",
              len,
              ", comm_id: ",
              comm_id,
              ", tag_comm_id: ",
              tag_comm_id,
              ", tag_counter: ",
              tagc);

    if (send_buf != recv_buf && send_buf != ATL_IN_PLACE) {
        memcpy(recv_buf, send_buf, total_size);
    }

    void* tmp_buf = malloc(total_size);
    CCL_THROW_IF_NOT(tmp_buf, "malloc failed");

    size_t pof2_size_t = ccl::utils::pof2(size);
    // coverity fix: overflowed constant
    CCL_THROW_IF_NOT(pof2_size_t <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     "pof2 value exceeds int range");
    int pof2 = static_cast<int>(pof2_size_t);
    int rem = size - pof2;

    if (rank < 2 * rem) {
        if (rank % 2 == 0) { /* even */
            atl_req send_req;
            uint64_t op_tag =
                tag_creator->create(rank, tag_comm_id, tagc, ofi_op_tag::allreduce_tag);
            do {
                ret = send(ep_idx, recv_buf, total_size, rank + 1, op_tag, send_req);
                CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "send failed");
                if (ret == ATL_STATUS_AGAIN) {
                    ccl_yield(ccl::global_data::env().yield_type);
                }
            } while (ret == ATL_STATUS_AGAIN);
            while (!send_req.is_completed) {
                poll(ep_idx);
                if (!send_req.is_completed) {
                    CCL_THROW_IF_NOT(check(ep_idx, send_req) != ATL_STATUS_FAILURE,
                                     "check send failed");
                }
            }

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        }
        else { /* odd */
            atl_req recv_req;
            uint64_t op_tag =
                tag_creator->create(rank - 1, tag_comm_id, tagc, ofi_op_tag::allreduce_tag);
            do {
                ret = recv(ep_idx, tmp_buf, total_size, rank - 1, op_tag, recv_req);
                CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "recv failed");
                if (ret == ATL_STATUS_AGAIN) {
                    ccl_yield(ccl::global_data::env().yield_type);
                }
            } while (ret == ATL_STATUS_AGAIN);
            while (!recv_req.is_completed) {
                poll(ep_idx);
                if (!recv_req.is_completed) {
                    CCL_THROW_IF_NOT(check(ep_idx, recv_req) != ATL_STATUS_FAILURE,
                                     "check recv failed");
                }
            }

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            size_t out_count;
            ret = atl_comp_reduce_regular(tmp_buf, len, recv_buf, &out_count, dtype, op);
            CCL_THROW_IF_NOT(ret == ATL_STATUS_SUCCESS, "atl_comp_reduce_regular failed");

            /* change the rank */
            newrank = rank / 2;
        }
    }
    else {
        newrank = rank - rem;
    }

    if (newrank != -1) {
        mask = 0x1;
        while (mask < pof2) {
            int newdst = newrank ^ mask;
            /* find real rank of dest */
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            /* Send the most current data, which is in recvbuf. Recv
             * into tmp_buf */
            atl_req send_req, recv_req;
            uint64_t op_tag =
                tag_creator->create(rank, tag_comm_id, tagc, ofi_op_tag::allreduce_tag);
            do {
                ret = send(ep_idx, recv_buf, total_size, dst, op_tag, send_req);
                CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "send failed");
                if (ret == ATL_STATUS_AGAIN) {
                    ccl_yield(ccl::global_data::env().yield_type);
                }
            } while (ret == ATL_STATUS_AGAIN);
            op_tag = tag_creator->create(dst, tag_comm_id, tagc, ofi_op_tag::allreduce_tag);
            do {
                ret = recv(ep_idx, tmp_buf, total_size, dst, op_tag, recv_req);
                CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "recv failed");
                if (ret == ATL_STATUS_AGAIN) {
                    ccl_yield(ccl::global_data::env().yield_type);
                }
            } while (ret == ATL_STATUS_AGAIN);
            while (!send_req.is_completed || !recv_req.is_completed) {
                poll(ep_idx);
                if (!send_req.is_completed) {
                    CCL_THROW_IF_NOT(check(ep_idx, send_req) != ATL_STATUS_FAILURE,
                                     "check send failed");
                }
                if (!recv_req.is_completed) {
                    CCL_THROW_IF_NOT(check(ep_idx, recv_req) != ATL_STATUS_FAILURE,
                                     "check recv failed");
                }
            }

            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            size_t out_count;
            // assume the op is always commutative
            ret = atl_comp_reduce_regular(tmp_buf, len, recv_buf, &out_count, dtype, op);
            CCL_THROW_IF_NOT(ret == ATL_STATUS_SUCCESS, "atl_comp_reduce_regular failed");
            mask <<= 1;
        }
    }

    if (rank < 2 * rem) {
        if (rank % 2) { /* odd */
            atl_req send_req;
            uint64_t op_tag =
                tag_creator->create(rank, tag_comm_id, tagc, ofi_op_tag::allreduce_tag);
            do {
                ret = send(ep_idx, recv_buf, total_size, rank - 1, op_tag, send_req);
                CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "send failed");
                if (ret == ATL_STATUS_AGAIN) {
                    ccl_yield(ccl::global_data::env().yield_type);
                }
            } while (ret == ATL_STATUS_AGAIN);
            while (!send_req.is_completed) {
                poll(ep_idx);
                if (!send_req.is_completed) {
                    CCL_THROW_IF_NOT(check(ep_idx, send_req) != ATL_STATUS_FAILURE,
                                     "check send failed");
                }
            }
        }
        else { /* even */
            atl_req recv_req;
            uint64_t op_tag =
                tag_creator->create(rank + 1, tag_comm_id, tagc, ofi_op_tag::allreduce_tag);
            do {
                ret = recv(ep_idx, recv_buf, total_size, rank + 1, op_tag, recv_req);
                CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "recv failed");
                if (ret == ATL_STATUS_AGAIN) {
                    ccl_yield(ccl::global_data::env().yield_type);
                }
            } while (ret == ATL_STATUS_AGAIN);
            while (!recv_req.is_completed) {
                poll(ep_idx);
                if (!recv_req.is_completed) {
                    CCL_THROW_IF_NOT(check(ep_idx, recv_req) != ATL_STATUS_FAILURE,
                                     "check recv failed");
                }
            }
        }
    }

    // to let user complete this operation through wait(req)
    req.is_completed = false;

    atl_ofi_req_t* ofi_req = ((atl_ofi_req_t*)req.internal);
    ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;

    free(tmp_buf);
    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi_comm::reduce_scatter(size_t ep_idx,
                                          const void* send_buf,
                                          void* recv_buf,
                                          size_t recv_len,
                                          atl_datatype_t dtype,
                                          atl_reduction_t op,
                                          atl_req_t& req) {
    // Recursive halving algorithm, suitable for small and medium size
    // messages
    atl_status_t ret;
    int i;
    size_t* disps = NULL;
    size_t *newcnts = NULL, *newdisps = NULL;
    void *tmp_recvbuf = NULL, *tmp_results = NULL;
    int mask, dst, pof2, rem;
    size_t pof2_size_t;
    int newdst, send_idx, recv_idx, last_idx;
    int newrank;
    int dsize = get_atl_datatype_size(dtype);
    size_t total_count = size * recv_len;
    size_t total_size = total_count * dsize;

    int tag_comm_id = (comm_id != atl_comm_id_storage::invalid_comm_id)
                          ? comm_id
                          : atl_comm_id_storage::max_comm_id;

    int tagc = tag_counter_reduce_scatter++;

    LOG_DEBUG("ofi_reduce_scatter: comm_rank: ",
              rank,
              ", comm_size: ",
              size,
              ", len: ",
              recv_len,
              ", comm_id: ",
              comm_id,
              ", tag_comm_id: ",
              tag_comm_id,
              ", tag_counter: ",
              tagc);

    if (recv_len == 0)
        goto fn_exit;

    disps = new size_t[size];
    total_count = 0;
    for (i = 0; i < size; i++) {
        disps[i] = total_count;
        total_count += recv_len;
    }

    tmp_recvbuf = malloc(total_size);
    CCL_THROW_IF_NOT(tmp_recvbuf, "malloc failed");

    tmp_results = malloc(total_size);
    CCL_THROW_IF_NOT(tmp_results, "malloc tmp_results failed");

    if (send_buf != recv_buf && send_buf != ATL_IN_PLACE) {
        memcpy(tmp_results, send_buf, total_size);
    }
    else {
        memcpy(tmp_results, recv_buf, total_size);
    }

    pof2_size_t = ccl::utils::pof2(size);
    // coverity fix: overflowed constant
    CCL_THROW_IF_NOT(pof2_size_t <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     "pof2 value exceeds int range");
    pof2 = static_cast<int>(pof2_size_t);
    rem = size - pof2;

    if (rank < 2 * rem) {
        if (rank % 2 == 0) { /* even */
            uint64_t op_tag =
                tag_creator->create(rank, tag_comm_id, tagc, ofi_op_tag::reduce_scatter_tag);
            ofi_send(tmp_results, total_size, rank + 1, this, ep_idx, op_tag);

            /* set the newrank to -1 to not to participate
             * in the main loop */
            newrank = -1;
        }
        else { /* odd */
            uint64_t op_tag =
                tag_creator->create(rank - 1, tag_comm_id, tagc, ofi_op_tag::reduce_scatter_tag);
            ofi_recv(tmp_recvbuf, total_size, rank - 1, this, ep_idx, op_tag);

            size_t out_count;
            ret = atl_comp_reduce_regular(
                tmp_recvbuf, total_count, tmp_results, &out_count, dtype, op);
            CCL_THROW_IF_NOT(ret == ATL_STATUS_SUCCESS, "atl_comp_reduce_regular failed");

            /* change the rank */
            newrank = rank / 2;
        }
    }
    else /* rank >= 2*rem */
        newrank = rank - rem;

    if (newrank != -1) {
        /* update recvcounts and disps arrays */
        newcnts = new size_t[pof2];
        newdisps = new size_t[pof2];

        for (i = 0; i < pof2; i++) {
            int old_i = (i < rem) ? i * 2 + 1 : i + rem;
            if (old_i < 2 * rem) {
                /* This process will do its left neighbor's work */
                newcnts[i] = recv_len * 2;
            }
            else
                newcnts[i] = recv_len;
        }

        newdisps[0] = 0;
        for (i = 1; i < pof2; i++)
            newdisps[i] = newdisps[i - 1] + newcnts[i - 1];

        mask = pof2 >> 1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + mask;
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
            }
            else {
                recv_idx = send_idx + mask;
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }

            /* Send data from tmp_results. Recv into tmp_recvbuf */
            if ((send_cnt != 0) && (recv_cnt != 0)) {
                atl_req send_req, recv_req;
                uint64_t op_tag =
                    tag_creator->create(rank, tag_comm_id, tagc, ofi_op_tag::reduce_scatter_tag);
                do {
                    ret = send(ep_idx,
                               (char*)tmp_results + newdisps[send_idx] * dsize,
                               send_cnt * dsize,
                               dst,
                               op_tag,
                               send_req);
                    CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "send failed");
                    if (ret == ATL_STATUS_AGAIN) {
                        ccl_yield(ccl::global_data::env().yield_type);
                    }
                } while (ret == ATL_STATUS_AGAIN);
                op_tag =
                    tag_creator->create(dst, tag_comm_id, tagc, ofi_op_tag::reduce_scatter_tag);
                do {
                    ret = recv(ep_idx,
                               (char*)tmp_recvbuf + newdisps[recv_idx] * dsize,
                               recv_cnt * dsize,
                               dst,
                               op_tag,
                               recv_req);
                    CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "recv failed");
                    if (ret == ATL_STATUS_AGAIN) {
                        ccl_yield(ccl::global_data::env().yield_type);
                    }
                } while (ret == ATL_STATUS_AGAIN);
                while (!send_req.is_completed || !recv_req.is_completed) {
                    poll(ep_idx);
                    if (!send_req.is_completed) {
                        CCL_THROW_IF_NOT(check(ep_idx, send_req) != ATL_STATUS_FAILURE,
                                         "check send failed");
                    }
                    if (!recv_req.is_completed) {
                        CCL_THROW_IF_NOT(check(ep_idx, recv_req) != ATL_STATUS_FAILURE,
                                         "check recv failed");
                    }
                }
            }
            else if ((send_cnt == 0) && (recv_cnt != 0)) {
                uint64_t op_tag =
                    tag_creator->create(dst, tag_comm_id, tagc, ofi_op_tag::reduce_scatter_tag);
                ofi_recv((char*)tmp_recvbuf + newdisps[recv_idx] * dsize,
                         recv_cnt * dsize,
                         dst,
                         this,
                         ep_idx,
                         op_tag);
            }
            else if ((recv_cnt == 0) && (send_cnt != 0)) {
                uint64_t op_tag =
                    tag_creator->create(rank, tag_comm_id, tagc, ofi_op_tag::reduce_scatter_tag);
                ofi_send((char*)tmp_results + newdisps[send_idx] * dsize,
                         send_cnt * dsize,
                         dst,
                         this,
                         ep_idx,
                         op_tag);
            }

            /* tmp_recvbuf contains data received in this step.
             * tmp_results contains data accumulated so far */

            if (recv_cnt) {
                size_t out_count;
                ret = atl_comp_reduce_regular((char*)tmp_recvbuf + newdisps[recv_idx] * dsize,
                                              recv_cnt,
                                              (char*)tmp_results + newdisps[recv_idx] * dsize,
                                              &out_count,
                                              dtype,
                                              op);
                CCL_THROW_IF_NOT(ret == ATL_STATUS_SUCCESS, "atl_comp_reduce_regular failed");
            }

            send_idx = recv_idx;
            last_idx = recv_idx + mask;
            mask >>= 1;
        }

        /* copy this process's result from tmp_results to recvbuf */
        memcpy(recv_buf, (char*)tmp_results + disps[rank] * dsize, recv_len * dsize);
    }

    /* In the non-power-of-two case, all odd-numbered processes 
     * which did extra work sends results to left neighbor */
    if (rank < 2 * rem) {
        if (rank % 2) { /* odd */
            if (recv_len) {
                uint64_t op_tag = tag_creator->create(rank, tag_comm_id, tagc, 3);
                ofi_send((char*)tmp_results + disps[rank - 1] * dsize,
                         recv_len * dsize,
                         rank - 1,
                         this,
                         ep_idx,
                         op_tag);
            }
        }
        else { /* even */
            if (recv_len) {
                uint64_t op_tag = tag_creator->create(
                    rank + 1, tag_comm_id, tagc, ofi_op_tag::reduce_scatter_tag);
                ofi_recv(recv_buf, recv_len * dsize, rank + 1, this, ep_idx, op_tag);
            }
        }
    }

fn_exit:
    // to let user complete this operation through wait(req)
    req.is_completed = false;

    atl_ofi_req_t* ofi_req = ((atl_ofi_req_t*)req.internal);
    ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;

    delete[] newcnts;
    delete[] newdisps;
    delete[] disps;
    free(tmp_recvbuf);
    free(tmp_results);
    return ATL_STATUS_SUCCESS;
}

std::shared_ptr<atl_base_comm> atl_ofi_comm::comm_split(int color, int key) {
    return std::shared_ptr<atl_base_comm>(new atl_ofi_comm(this, color));
}

atl_ofi_comm::atl_ofi_comm(atl_ofi_comm* parent, int color) {
    eps = parent->eps;
    parent_size = parent->size;
    parent_rank = parent->rank;
    pmi = parent->pmi;

    coord.hostname_hash = transport->get_proc_coord().hostname_hash;
    coord.local_idx = 0;
    coord.local_count = 0;

    std::vector<rank_info_t> ranks_info(parent_size);
    rank_info_t rank_info{ color, parent_rank, coord.hostname_hash };
    std::vector<size_t> recv_lens(parent_size, sizeof(rank_info));
    std::vector<size_t> offsets(parent_size);
    offsets[0] = 0;
    for (size_t i = 1; i < offsets.size(); i++) {
        offsets[i] = offsets[i - 1] + recv_lens[i];
    }

    atl_req req{};
    atl_status_t status = parent->allgatherv(0 /* ep_idx */,
                                             &rank_info,
                                             sizeof(rank_info),
                                             ranks_info.data(),
                                             recv_lens.data(),
                                             offsets.data(),
                                             req);
    CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "allgatherv failed with status: ", status);

    status = wait(0, req);
    CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "wait failed with status: ", status);

    CCL_THROW_IF_NOT(rank2proc_map.empty());
    CCL_THROW_IF_NOT(rank2rank_map.empty());

    size = 0;

    for (auto& it : ranks_info) {
        int recv_color;
        int recv_rank;
        size_t recv_hash;
        std::tie(recv_color, recv_rank, recv_hash) = it;
        if (recv_color == color) {
            rank2proc_map.push_back(parent->rank2proc_map[recv_rank]);
            rank2rank_map.push_back(recv_rank);

            if (recv_hash == coord.hostname_hash) {
                coord.local_count++;
            }

            if (recv_rank == parent_rank) {
                coord.global_idx = rank = rank2proc_map.size() - 1;
                coord.local_idx = (coord.local_count - 1);
            }
            size++;
        }
    }
    coord.global_count = size;

    LOG_DEBUG("color: ",
              color,
              ", ",
              to_string(coord),
              ", rank2proc_map: ",
              ccl::utils::vec_to_string(rank2proc_map),
              ", parent rank2proc_map: ",
              ccl::utils::vec_to_string(parent->rank2proc_map));

    coord.validate(rank, size);

    CCL_THROW_IF_NOT(init_transport(false) == ATL_STATUS_SUCCESS, "init transport failed");
}

atl_status_t atl_ofi_comm::init_transport(bool is_new) {
    LOG_DEBUG("init atl, requested ep_count ", attr.in.ep_count);

    if (is_new) {
        ATL_CHECK_STATUS(pmi->pmrt_init(), "pmi init failed");
        static std::mutex memory_mutex;
        {
            std::lock_guard<std::mutex> lock(memory_mutex);
            if (!transport) {
                transport = new atl_ofi();
            }
            if (!transport->is_inited()) {
                CCL_THROW_IF_NOT(
                    transport->init(nullptr, nullptr, &attr, nullptr, pmi) == ATL_STATUS_SUCCESS,
                    "failed to initialize ATL");

                if (pmi->get_rank() == 0) {
                    LOG_INFO(transport->to_string());
                    LOG_INFO(to_string(attr));
                }
            }
        }
        eps = transport->get_eps();

        parent_rank = rank = pmi->get_rank();
        parent_size = size = pmi->get_size();

        coord = transport->get_proc_coord();
        coord.validate(rank, size);

        transport->get_rank2proc_map(pmi, rank2proc_map, coord);
        rank2rank_map.resize(size);
        for (int i = 0; i < size; i++) {
            rank2rank_map[i] = i;
        }
    }

    init_tag();

    comm_id = create_comm_id();
    comm_count++;

    update_executor();

    return ATL_STATUS_SUCCESS;
}
