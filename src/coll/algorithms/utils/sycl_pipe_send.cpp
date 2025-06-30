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
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"

//namespace ccl {
//namespace v1 {

const int num_chunk_buffs = ccl_scaleout_pipeline_bufs::num_chunk_buffs;

// calculate the total number of chunks for pipeline
void pipe_prep(size_t min_msg_count,
               size_t max_msg_count,
               int dsize,
               size_t pipeline_chunk_size,
               size_t& nchunks) {
    int align = std::max(4, dsize);
    min_msg_count = (min_msg_count * dsize + align - 1) / align;
    max_msg_count = (max_msg_count * dsize + align - 1) / align;
    size_t max_pipeline_chunk_count = pipeline_chunk_size / align;
    size_t min_nchunks = (min_msg_count + max_pipeline_chunk_count - 1) / max_pipeline_chunk_count;
    size_t max_nchunks = (max_msg_count + max_pipeline_chunk_count - 1) / max_pipeline_chunk_count;
    nchunks = std::max(min_nchunks, max_nchunks);
}

static void calculate_chunk_sizes(size_t total_send_size,
                                  size_t total_recv_size,
                                  int dsize,
                                  size_t nchunks,
                                  size_t& send_chunk_size,
                                  size_t& recv_chunk_size) {
    send_chunk_size = (total_send_size + nchunks - 1) / nchunks;
    recv_chunk_size = (total_recv_size + nchunks - 1) / nchunks;
    int align = std::max(4, dsize);
    send_chunk_size = (send_chunk_size + align - 1) / align * align;
    recv_chunk_size = (recv_chunk_size + align - 1) / align * align;
    // validation
    size_t max_pipeline_chunk_size = ccl::global_data::env().sycl_max_pipeline_chunk_size;
    CCL_ASSERT(send_chunk_size <= max_pipeline_chunk_size &&
               recv_chunk_size <= max_pipeline_chunk_size);
}

// one-way RDMA
sycl::event pipe_sendrecv_rdma(sycl::queue& q,
                               const void* send_buf,
                               size_t send_count,
                               int dest,
                               int sendtag,
                               void* recv_buf,
                               size_t recv_count,
                               int src,
                               int recvtag,
                               ccl::datatype dtype,
                               size_t nchunks,
                               ccl_comm* comm,
                               const ccl::vector_class<sycl::event>& deps) {
    sycl::event e;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t total_send_size = send_count * ccl_dtype.size();
    size_t total_recv_size = recv_count * ccl_dtype.size();
    size_t send_chunk_size, recv_chunk_size;

    calculate_chunk_sizes(total_send_size,
                          total_recv_size,
                          ccl_dtype.size(),
                          nchunks,
                          send_chunk_size,
                          recv_chunk_size);

    // get pipe_chunks
    void** send_pipe_chunks = comm->get_scaleout_send_pipeline_bufs(num_chunk_buffs);
    void** recv_pipe_chunks = comm->get_scaleout_recv_pipeline_bufs(num_chunk_buffs);

    sycl::queue q_scopy = q; //get_lce_queue(q, 2);
    sycl::queue q_rcopy = q; //get_lce_queue(q, 3);

    int ep_idx = 0;
    sycl::event send_e, recv_e, send_done_e, recv_done_e, recv_copy_e;

    // start pipeline
    int idx = 0;
    int chunk_index = 0, prev_chunk_index = 0;
    int next_chunk_index = 1;
    int send_size = (1 == nchunks ? total_send_size : send_chunk_size);
    int recv_size = (1 == nchunks ? total_recv_size : recv_chunk_size);

    // post recv (pipe)
    recv_e = q.submit([=](sycl::handler& h) {
        h.depends_on(deps);
        h.host_task([=]() {
            atl_req_t& recv_req = comm->get_pipeline_recv_req();
            ATL_CALL_THROW_IF_ERROR(atl_comm->recv(
                ep_idx, recv_pipe_chunks[chunk_index], recv_size, src, recvtag, recv_req));
        });
    });
    // post send (rdma)
    send_e = q.submit([=](sycl::handler& h) {
        h.depends_on(deps);
        h.host_task([=]() {
            atl_req_t& send_req = comm->get_pipeline_send_req();
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, send_buf, send_size, dest, sendtag, send_req));
        });
    });
    recv_done_e = q.submit([=](sycl::handler& h) {
        h.depends_on({ recv_e, send_e });
        h.host_task([=]() {
            atl_req_t& send_req = comm->get_pipeline_send_req();
            atl_req_t& recv_req = comm->get_pipeline_recv_req();
            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, recv_req));
            if (!recv_req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            }
            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, send_req));
            if (!send_req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            }
        });
    });
    for (idx = 1; idx < nchunks; idx++) {
        send_size =
            (idx == nchunks - 1 ? total_send_size - send_chunk_size * idx : send_chunk_size);
        recv_size =
            (idx == nchunks - 1 ? total_recv_size - recv_chunk_size * idx : recv_chunk_size);

        chunk_index = idx % num_chunk_buffs;
        prev_chunk_index = (chunk_index + num_chunk_buffs - 1) % num_chunk_buffs;

        // post send and recv
        recv_e = q.submit([=](sycl::handler& h) {
            h.depends_on(recv_done_e);
            h.host_task([=]() {
                atl_req_t& send_req = comm->get_pipeline_send_req();
                atl_req_t& recv_req = comm->get_pipeline_recv_req();
                ATL_CALL_THROW_IF_ERROR(atl_comm->send(ep_idx,
                                                       (char*)send_buf + idx * send_chunk_size,
                                                       send_size,
                                                       dest,
                                                       sendtag,
                                                       send_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->recv(
                    ep_idx, recv_pipe_chunks[chunk_index], recv_size, src, recvtag, recv_req));
            });
        });
        // copy for receive for prev chunk
        recv_copy_e = q_rcopy.submit([=](sycl::handler& h) {
            h.depends_on(recv_done_e);
            h.memcpy((char*)recv_buf + (idx - 1) * recv_chunk_size,
                     recv_pipe_chunks[prev_chunk_index],
                     recv_chunk_size);
        });
        // wait for send/recv completion
        recv_done_e = q.submit([=](sycl::handler& h) {
            h.depends_on({ recv_e, recv_copy_e });
            h.host_task([=]() {
                atl_req_t& send_req = comm->get_pipeline_send_req();
                atl_req_t& recv_req = comm->get_pipeline_recv_req();
                ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, recv_req));
                if (!recv_req.is_completed) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
                }
                ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, send_req));
                if (!send_req.is_completed) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
                }
            });
        });
    }
    // copy for recv for last chunk (recv_size is already calculated)
    recv_copy_e = q_rcopy.submit([=](sycl::handler& h) {
        h.depends_on(recv_done_e);
        h.memcpy((char*)recv_buf + (nchunks - 1) * recv_chunk_size,
                 recv_pipe_chunks[chunk_index],
                 recv_size);
    });

    return recv_copy_e;
}

sycl::event pipe_sendrecv_plain(sycl::queue& q,
                                const void* send_buf,
                                size_t send_count,
                                int dest,
                                int sendtag,
                                void* recv_buf,
                                size_t recv_count,
                                int src,
                                int recvtag,
                                ccl::datatype dtype,
                                size_t nchunks,
                                ccl_comm* comm,
                                const ccl::vector_class<sycl::event>& deps) {
    sycl::event e;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t total_send_size = send_count * ccl_dtype.size();
    size_t total_recv_size = recv_count * ccl_dtype.size();
    size_t send_chunk_size, recv_chunk_size;

    calculate_chunk_sizes(total_send_size,
                          total_recv_size,
                          ccl_dtype.size(),
                          nchunks,
                          send_chunk_size,
                          recv_chunk_size);

    // get pipe_chunks
    void** send_pipe_chunks = comm->get_scaleout_send_pipeline_bufs(num_chunk_buffs);
    void** recv_pipe_chunks = comm->get_scaleout_recv_pipeline_bufs(num_chunk_buffs);

    sycl::queue q_scopy = q; // get_mce_queue(q);
    sycl::queue q_rcopy = q; // get_mce_queue(q);

    int ep_idx = 0;
    sycl::event send_e, recv_e, recv_done_e, send_copy_e, recv_copy_e;

    // start pipeline
    int idx = 0;
    int chunk_index = 0, prev_chunk_index = 0;
    int next_chunk_index = 1;
    int send_size = (1 == nchunks ? total_send_size : send_chunk_size);
    int recv_size = (1 == nchunks ? total_recv_size : recv_chunk_size);

    // recv0   send0   |  send1     recv1   |  send2     recv2 | send3   recv3
    // s_copy0 s_copy1 |  s_copy2   r_copy0 |  s_copy3 r_copy1 | r_copy2 r_copy3
    //
    // copy for send
    send_copy_e = q_scopy.submit([=](sycl::handler& h) {
        h.depends_on(deps);
        h.memcpy(send_pipe_chunks[chunk_index], send_buf, send_size);
    });
    // post recv
    recv_e = q.submit([=](sycl::handler& h) {
        h.depends_on(deps);
        h.host_task([=]() {
            atl_req_t& recv_req = comm->get_pipeline_recv_req();
            ATL_CALL_THROW_IF_ERROR(atl_comm->recv(
                ep_idx, recv_pipe_chunks[chunk_index], recv_size, src, recvtag, recv_req));
        });
    });
    // post send after copy is done
    send_e = q.submit([=](sycl::handler& h) {
        h.depends_on(send_copy_e);
        h.host_task([=]() {
            atl_req_t& send_req = comm->get_pipeline_send_req();
            ATL_CALL_THROW_IF_ERROR(atl_comm->send(
                ep_idx, send_pipe_chunks[chunk_index], send_size, dest, sendtag, send_req));
        });
    });
    // start copy next chunk for send
    if (nchunks > 1) {
        int next_send_size =
            (1 == nchunks - 1 ? total_send_size - send_chunk_size : send_chunk_size);
        send_copy_e = q_scopy.submit([=](sycl::handler& h) {
            h.depends_on(send_copy_e);
            h.memcpy(send_pipe_chunks[next_chunk_index],
                     (char*)send_buf + send_chunk_size,
                     next_send_size);
        });
    }
    recv_done_e = q.submit([=](sycl::handler& h) {
        h.depends_on({ send_copy_e, recv_e, send_e });
        h.host_task([=]() {
            atl_req_t& send_req = comm->get_pipeline_send_req();
            atl_req_t& recv_req = comm->get_pipeline_recv_req();
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            //if (!send_req.is_completed) {
            //    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            //}
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            //if (!recv_req.is_completed) {
            //    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            //}
        });
    });

    for (idx = 1; idx < nchunks - 1; idx++) {
        send_size = send_chunk_size;
        int next_send_size = (idx + 1 == nchunks - 1 ? total_send_size - send_chunk_size * (idx + 1)
                                                     : send_chunk_size);
        recv_size = recv_chunk_size;
        int next_recv_size = (idx + 1 == nchunks - 1 ? total_recv_size - recv_chunk_size * (idx + 1)
                                                     : recv_chunk_size);
        chunk_index = idx % num_chunk_buffs;
        next_chunk_index = (chunk_index + 1) % num_chunk_buffs;
        prev_chunk_index = (chunk_index + num_chunk_buffs - 1) % num_chunk_buffs;
        // post send and recv
        recv_e = q.submit([=](sycl::handler& h) {
            h.depends_on(recv_done_e);
            h.host_task([=]() {
                atl_req_t& send_req = comm->get_pipeline_send_req();
                atl_req_t& recv_req = comm->get_pipeline_recv_req();
                ATL_CALL_THROW_IF_ERROR(atl_comm->recv(
                    ep_idx, recv_pipe_chunks[chunk_index], recv_size, src, recvtag, recv_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->send(
                    ep_idx, send_pipe_chunks[chunk_index], send_size, dest, sendtag, send_req));
            });
        });
        // copy for receive for prev chunk
        recv_copy_e = q_rcopy.submit([=](sycl::handler& h) {
            h.depends_on(recv_done_e);
            h.memcpy((char*)recv_buf + (idx - 1) * recv_chunk_size,
                     recv_pipe_chunks[prev_chunk_index],
                     recv_chunk_size);
        });
        // copy for send for next chunk
        send_copy_e = q_scopy.submit([=](sycl::handler& h) {
            h.depends_on(recv_done_e);
            h.memcpy(send_pipe_chunks[next_chunk_index],
                     (char*)send_buf + (idx + 1) * send_chunk_size,
                     next_send_size);
        });
        // wait for send/recv completion
        recv_done_e = q.submit([=](sycl::handler& h) {
            h.depends_on({ recv_e, recv_copy_e, send_copy_e });
            h.host_task([=]() {
                atl_req_t& send_req = comm->get_pipeline_send_req();
                atl_req_t& recv_req = comm->get_pipeline_recv_req();
                ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, recv_req));
                if (!recv_req.is_completed) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
                }
                ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, send_req));
                if (!send_req.is_completed) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
                }
            });
        });
    }

    //  send last chunk
    if (nchunks > 1) {
        send_size = total_send_size - send_chunk_size * (nchunks - 1);
        recv_size = total_recv_size - recv_chunk_size * (nchunks - 1);
        chunk_index = idx % num_chunk_buffs;
        // post send and recv
        recv_e = q.submit([=](sycl::handler& h) {
            h.depends_on(recv_done_e);
            h.host_task([=]() {
                atl_req_t& send_req = comm->get_pipeline_send_req();
                atl_req_t& recv_req = comm->get_pipeline_recv_req();
                ATL_CALL_THROW_IF_ERROR(atl_comm->recv(
                    ep_idx, recv_pipe_chunks[chunk_index], recv_size, src, recvtag, recv_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->send(
                    ep_idx, send_pipe_chunks[chunk_index], send_size, dest, sendtag, send_req));
            });
        });
    }
    // copy for recv for previous chunk
    int prev_recv_size = total_recv_size > recv_chunk_size ? recv_chunk_size : total_recv_size;
    prev_chunk_index = (idx - 1) % num_chunk_buffs;
    recv_copy_e = q_rcopy.submit([=](sycl::handler& h) {
        h.depends_on(recv_done_e);
        h.memcpy((char*)recv_buf + (idx - 1) * recv_chunk_size,
                 recv_pipe_chunks[prev_chunk_index],
                 prev_recv_size);
    });
    if (nchunks > 1) {
        recv_done_e = q.submit([=](sycl::handler& h) {
            h.depends_on({ recv_e, recv_copy_e });
            h.host_task([=]() {
                atl_req_t& send_req = comm->get_pipeline_send_req();
                atl_req_t& recv_req = comm->get_pipeline_recv_req();
                ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, send_req));
                if (!send_req.is_completed) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
                }
                ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, recv_req));
                if (!recv_req.is_completed) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
                }
            });
        });
        // copy for recv for current chunk
        recv_copy_e = q_rcopy.submit([=](sycl::handler& h) {
            h.depends_on(recv_done_e);
            h.memcpy((char*)recv_buf + (nchunks - 1) * recv_chunk_size,
                     recv_pipe_chunks[chunk_index],
                     recv_size);
        });
    }

    return recv_copy_e;
}

sycl::event pipe_sendrecv(sycl::queue& q,
                          const void* send_buf,
                          size_t send_count,
                          int dest,
                          int sendtag,
                          void* recv_buf,
                          size_t recv_count,
                          int src,
                          int recvtag,
                          ccl::datatype dtype,
                          size_t nchunks,
                          ccl_comm* comm,
                          const ccl::vector_class<sycl::event>& deps,
                          bool use_rdma) {
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    if (should_disable_rdma(ze_dev)) {
        use_rdma = false;
    }
    if (!use_rdma) {
        return pipe_sendrecv_plain(q,
                                   send_buf,
                                   send_count,
                                   dest,
                                   sendtag,
                                   recv_buf,
                                   recv_count,
                                   src,
                                   recvtag,
                                   dtype,
                                   nchunks,
                                   comm,
                                   deps);
    }
    else {
        return pipe_sendrecv_rdma(q,
                                  send_buf,
                                  send_count,
                                  dest,
                                  sendtag,
                                  recv_buf,
                                  recv_count,
                                  src,
                                  recvtag,
                                  dtype,
                                  nchunks,
                                  comm,
                                  deps);
    }
}

// two-way RDMA, no pipeline
sycl::event sendrecv_rdma(sycl::queue& q,
                          const void* send_buf,
                          size_t send_count,
                          int dest,
                          int sendtag,
                          void* recv_buf,
                          size_t recv_count,
                          int src,
                          int recvtag,
                          ccl::datatype dtype,
                          ccl_comm* comm,
                          const ccl::vector_class<sycl::event>& deps) {
    sycl::event e;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t send_size = send_count * ccl_dtype.size();
    size_t recv_size = recv_count * ccl_dtype.size();
    CCL_ASSERT(send_size <= comm->get_scaleout_host_buf_size());
    CCL_ASSERT(recv_size <= comm->get_scaleout_host_buf_size());

    int ep_idx = 0;
    e = q.submit([=](sycl::handler& h) {
        h.depends_on(deps);
        h.host_task([=]() {
            atl_req_t send_req, recv_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, send_buf, send_size, dest, sendtag, send_req));
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->recv(ep_idx, recv_buf, recv_size, src, recvtag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, send_req));
            if (!send_req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            }
            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, recv_req));
            if (!recv_req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            }
        });
    });
    return e;
}

// no pipeline
sycl::event gpu_send_plain(sycl::queue& q,
                           const void* send_buf,
                           size_t send_count,
                           int dest,
                           int sendtag,
                           ccl::datatype dtype,
                           ccl_comm* comm,
                           const ccl::vector_class<sycl::event>& deps) {
    sycl::event e;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t send_size = send_count * ccl_dtype.size();
    CCL_ASSERT(send_size <= comm->get_scaleout_host_buf_size());

    void* host_buf = comm->get_scaleout_host_buf();
    e = q.submit([=](sycl::handler& h) {
        h.depends_on(deps);
        h.memcpy(host_buf, send_buf, send_size);
    });

    int ep_idx = 0;
    e = q.submit([=](sycl::handler& h) {
        h.depends_on(e);
        h.host_task([=]() {
            atl_req_t send_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, host_buf, send_size, dest, sendtag, send_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, send_req));
            if (!send_req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            }
        });
    });
    comm->put_scaleout_host_buf(host_buf);
    return e;
}

sycl::event gpu_recv_plain(sycl::queue& q,
                           void* recv_buf,
                           size_t recv_count,
                           int src,
                           int recvtag,
                           ccl::datatype dtype,
                           ccl_comm* comm,
                           const ccl::vector_class<sycl::event>& deps) {
    sycl::event e;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t recv_size = recv_count * ccl_dtype.size();
    CCL_ASSERT(recv_size <= comm->get_scaleout_host_buf_size());

    void* host_buf = comm->get_scaleout_host_buf();
    int ep_idx = 0;
    e = q.submit([=](sycl::handler& h) {
        h.depends_on(deps);
        h.host_task([=]() {
            atl_req_t recv_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->recv(ep_idx, host_buf, recv_size, src, recvtag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, recv_req));
            if (!recv_req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            }
        });
    });

    e = q.submit([=](sycl::handler& h) {
        h.depends_on(e);
        h.memcpy(recv_buf, host_buf, recv_size);
    });

    comm->put_scaleout_host_buf(host_buf);
    return e;
}

//} // namespace v1
//} // namespace ccl
