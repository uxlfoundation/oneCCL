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
#include "common/log/log.hpp"
#include "common/utils/exchange_utils.hpp"
#include "common/utils/fd_info.hpp"
#include "coll/coll_util.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"

namespace ccl {
namespace utils {

bool allgather(const std::shared_ptr<atl_base_comm>& comm,
               const void* send_buf,
               void* recv_buf,
               size_t bytes,
               bool sync) {
    std::vector<size_t> recv_bytes(comm->get_size(), bytes);
    return allgatherv(comm, send_buf, recv_buf, recv_bytes, sync);
}

bool allgatherv(const std::shared_ptr<atl_base_comm>& comm,
                const void* send_buf,
                void* recv_buf,
                const std::vector<size_t>& recv_bytes,
                bool sync) {
    atl_req_t req{};
    bool ret = true;
    int comm_rank = comm->get_rank();
    int comm_size = comm->get_size();

    CCL_THROW_IF_NOT((int)recv_bytes.size() == comm->get_size(),
                     "unexpected recv_bytes size ",
                     recv_bytes.size(),
                     ", comm_size ",
                     comm_size);

    std::vector<size_t> offsets(comm_size, 0);
    for (int i = 1; i < comm_size; i++) {
        offsets[i] = offsets[i - 1] + recv_bytes[i - 1];
    }

    // MPI_Iallgatherv used in ipc exchange is interfering with
    // other MPI collectives from other threads when both sycl
    // and scheduler paths are running together. MPI_Allgatherv
    // seems to not have the issue, and so using it here.
    if (ccl::global_data::env().atl_transport == ccl_atl_mpi &&
        ccl::global_data::env().ipc_allgatherv_wa) {
        const size_t send_len = recv_bytes[comm_rank];
        MPI_Comm mpi_comm = comm->get_mpi_comm();

        std::vector<size_t> recv_lens_size_t(comm_size, 0);
        std::vector<Compat_MPI_Count_t> recv_conv_lens(comm_size);
        std::vector<Compat_MPI_Aint_t> recv_conv_offsets(comm_size);

        for (int i = 0; i < comm_size; ++i) {
            recv_lens_size_t[i] = recv_bytes[i];
            recv_conv_lens[i] = static_cast<Compat_MPI_Aint_t>(recv_bytes[i]);
            recv_conv_offsets[i] = static_cast<Compat_MPI_Aint_t>(offsets[i]);
        }

        bool inplace = ccl::is_allgatherv_inplace(send_buf,
                                                  send_len,
                                                  recv_buf,
                                                  recv_lens_size_t.data(),
                                                  offsets.data(),
                                                  1 /*dtype_size*/,
                                                  comm_rank,
                                                  comm_size);

        int mpi_ret = MPI_Allgatherv_c(inplace ? MPI_IN_PLACE : send_buf,
                                       send_len,
                                       MPI_CHAR,
                                       recv_buf,
                                       recv_conv_lens.data(),
                                       recv_conv_offsets.data(),
                                       MPI_CHAR,
                                       mpi_comm);
        ret = mpi_ret == MPI_SUCCESS;
    }
    else {
        atl_status_t status = comm->allgatherv(0 /* ep_idx */,
                                               send_buf,
                                               recv_bytes[comm_rank],
                                               recv_buf,
                                               recv_bytes.data(),
                                               offsets.data(),
                                               req);
        CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "allgatherv failed with status: ", status);

        if (sync) {
            status = comm->wait(0 /* ep_idx */, req);
            CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "wait failed with status: ", status);
        }
        else {
            CCL_THROW("unexpected sync parameter");
        }
    }
    return ret;
}

void check(const std::shared_ptr<atl_base_comm>& comm, atl_req_t& req) {
    atl_status_t atl_status = comm->check(0, req);

    if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
        CCL_THROW("check failed: atl_status: ", atl_status_to_str(atl_status));
    }

    while (!req.is_completed) {
        atl_status_t status = comm->check(0, req);
        if (unlikely(status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("check failed: atl_status: ", atl_status_to_str(status));
        }
        if (req.is_completed) {
            break;
        }
    }
}

bool check_async(const std::shared_ptr<atl_base_comm>& comm, atl_req_t& req) {
    atl_status_t atl_status = comm->check(0, req);

    if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
        CCL_THROW("check failed: atl_status: ", atl_status_to_str(atl_status));
    }

    return req.is_completed;
}

atl_req_t recv(const std::shared_ptr<atl_base_comm>& comm,
               void* buf,
               int count,
               int peer_rank,
               uint64_t tag,
               bool sync) {
    atl_req_t req{};
    atl_status_t status = comm->recv(0 /* ep_idx */, buf, count, peer_rank /*src rank*/, tag, req);
    CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "recv failed with status: ", status);

    if (sync) {
        check(comm, req);
    }

    return req;
}

atl_req_t send(const std::shared_ptr<atl_base_comm>& comm,
               void* buf,
               int count,
               int peer_rank,
               uint64_t tag,
               bool sync) {
    atl_req_t req{};
    atl_status_t status = comm->send(0 /* ep_idx */, buf, count, peer_rank /*dst rank*/, tag, req);
    CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "send failed with status: ", status);

    if (sync) {
        check(comm, req);
    }

    return req;
}

atl_req_t send_ack_to_peer(const std::shared_ptr<atl_base_comm>& comm,
                           uint64_t tag,
                           int peer_rank,
                           bool sync) {
    auto req = ccl::utils::send(comm, nullptr, 0, peer_rank, tag, sync);
    LOG_DEBUG("send ack msg with tag: ", tag);
    return req;
}

atl_req_t recv_ack_from_peer(const std::shared_ptr<atl_base_comm>& comm,
                             uint64_t tag,
                             int peer_rank,
                             bool sync) {
    char ack[1];
    auto req = ccl::utils::recv(comm, ack, 0, peer_rank, tag, sync);
    LOG_DEBUG("recv ack msg with tag: ", tag);
    return req;
}

int check_msg_retval(std::string operation_name,
                     ssize_t bytes,
                     struct iovec iov,
                     struct msghdr msg,
                     size_t union_size,
                     int sock,
                     int fd) {
    LOG_DEBUG(operation_name,
              ": ",
              bytes,
              ", expected_bytes:",
              iov.iov_len,
              ", expected size of cntr_buf: ",
              union_size,
              " -> gotten cntr_buf: ",
              msg.msg_controllen,
              ", socket: ",
              sock,
              ", fd: ",
              fd);
    int ret = -1;
    if (bytes == static_cast<ssize_t>(iov.iov_len)) {
        ret = 0;
    }
    else if (bytes < 0) {
        ret = -errno;
    }
    else {
        ret = -EIO;
    }
    return ret;
}

void sendmsg_fd(int sock, int fd, void* payload, int payload_len) {
    CCL_THROW_IF_NOT(fd >= 0, "unexpected fd value");
    char empty_buf;
    struct iovec iov;
    memset(&iov, 0, sizeof(iov));
    if (!payload) {
        iov.iov_base = &empty_buf;
        iov.iov_len = 1;
    }
    else {
        iov.iov_base = payload;
        iov.iov_len = payload_len;
    }

    union {
        struct cmsghdr align;
        char cntr_buf[CMSG_SPACE(sizeof(int))]{};
    } u;

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_control = u.cntr_buf;
    msg.msg_controllen = sizeof(u.cntr_buf);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    auto cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_len = CMSG_LEN(sizeof(fd));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    *(int*)CMSG_DATA(cmsg) = fd;

    ssize_t send_bytes = sendmsg(sock, &msg, 0);
    CCL_THROW_IF_NOT(
        !check_msg_retval("sendmsg", send_bytes, iov, msg, sizeof(u.cntr_buf), sock, fd),
        " errno: ",
        strerror(errno));
}

void recvmsg_fd(int sock, int* fd, void* payload, int payload_len) {
    CCL_THROW_IF_NOT(fd != nullptr, "unexpected fd value");
    char empty_buf;
    struct iovec iov;
    memset(&iov, 0, sizeof(iov));
    if (!payload) {
        iov.iov_base = &empty_buf;
        iov.iov_len = 1;
    }
    else {
        iov.iov_base = payload;
        iov.iov_len = payload_len;
    }

    union {
        struct cmsghdr align;
        char cntr_buf[CMSG_SPACE(sizeof(int))]{};
    } u;

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_control = u.cntr_buf;
    msg.msg_controllen = sizeof(u.cntr_buf);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    ssize_t recv_bytes = recvmsg(sock, &msg, 0);
    CCL_THROW_IF_NOT(
        !check_msg_retval("recvmsg", recv_bytes, iov, msg, sizeof(u.cntr_buf), sock, *fd),
        " errno: ",
        strerror(errno));

    if (msg.msg_flags & (MSG_CTRUNC | MSG_TRUNC)) {
        std::string flag_str = "";
        if (msg.msg_flags & MSG_CTRUNC) {
            flag_str += " MSG_CTRUNC";
        }
        if (msg.msg_flags & MSG_TRUNC) {
            flag_str += " MSG_TRUNC";
        }

        /** MSG_CTRUNC message can be in case of:
         * - remote peer send invalid fd, so msg_controllen == 0
         * - limit of fds reached in the current process, so msg_controllen == 0
         * - the remote peer control message > msg_control buffer size
         */
        CCL_THROW("control or usual message is truncated:",
                  flag_str,
                  " control message size: ",
                  msg.msg_controllen,
                  ", ",
                  to_string(ccl::utils::get_fd_info()));
    }

    for (auto cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_len == CMSG_LEN(sizeof(int)) && cmsg->cmsg_level == SOL_SOCKET &&
            cmsg->cmsg_type == SCM_RIGHTS) {
            memcpy(fd, CMSG_DATA(cmsg), sizeof(int));
            break;
        }
    }

    // we assume that the message has a strict format and size, if not this means that something
    // is wrong.
    size_t expected_len = 1;
    if (payload) {
        expected_len = payload_len;
    }
    if (msg.msg_iov[0].iov_len != expected_len) {
        CCL_THROW("received data in unexpected format");
    }
}

void sendmsg_call(int sock, int fd, void* payload, int payload_len, const int rank) {
    sendmsg_fd(sock, fd, payload, payload_len);
    LOG_DEBUG("send: rank[", rank, "], send fd: ", fd, ", sock: ", sock);
}

void recvmsg_call(int sock, int* fd, void* payload, int payload_len, const int rank) {
    recvmsg_fd(sock, fd, payload, payload_len);
    LOG_DEBUG("recv: rank[", rank, "], got fd: ", fd, ", sock: ", sock);
}
} // namespace utils
} // namespace ccl
