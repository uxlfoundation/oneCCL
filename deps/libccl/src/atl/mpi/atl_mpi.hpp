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

#ifdef CCL_ENABLE_MPI

#include "atl/atl_base_transport.hpp"
#include "atl/mpi/atl_mpi_ctx.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"

#define ATL_MPI_RET(ret) (ret != MPI_SUCCESS) ? ATL_STATUS_FAILURE : ATL_STATUS_SUCCESS

#define ATL_MPI_BASE_PM_KEY      "atl-mpi"
#define ATL_MPI_RANK_INFO_PM_KEY ATL_MPI_BASE_PM_KEY "-rank_info"

#define ATL_MPI_RANK_STR_SIZE 8

#define MPI_BFLOAT16 \
    ({ \
        CCL_THROW_IF_NOT(ctx.bf16.dtype != MPI_DATATYPE_NULL, \
                         "unsupported datatype: ATL_DTYPE_BF16"); \
        ctx.bf16.dtype; \
    })

#define MPI_FLOAT16 \
    ({ \
        CCL_THROW_IF_NOT(ctx.fp16.dtype != MPI_DATATYPE_NULL, \
                         "unsupported datatype: ATL_DTYPE_FP16"); \
        ctx.fp16.dtype; \
    })

// MLSL-3103
#define KVS_MPI_TAG_TO_ROOT   222
#define KVS_MPI_TAG_FROM_ROOT 223

typedef enum { ATL_MPI_COMP_POSTED, ATL_MPI_COMP_COMPLETED } atl_mpi_comp_state_t;

typedef struct {
    MPI_Request native_req;
    atl_mpi_comp_state_t comp_state;
} atl_mpi_req_t;

typedef struct {
    MPI_Comm mpi_comm;
    /* dummy recv operation to ensure progress in atl_poll */
    atl_mpi_req_t dummy_req;
    MPI_Comm dummy_comm;
} atl_mpi_ep_t;

typedef struct atl_mpi_env_info {
    int found;
    char key[MPI_MAX_INFO_KEY];
    char value[MPI_MAX_INFO_VAL];

    atl_mpi_env_info() {
        found = 0;
        memset(key, 0, MPI_MAX_INFO_KEY);
        memset(value, 0, MPI_MAX_INFO_VAL);
    }
} atl_mpi_env_info_t;

typedef struct atl_mpi_comm_info : atl_mpi_env_info_t {
    MPI_Comm comm;

    atl_mpi_comm_info() {
        comm = MPI_COMM_WORLD;
    }
} atl_mpi_comm_info_t;

class atl_mpi : public atl_base_transport {
public:
    atl_mpi() = default;
    atl_mpi(const atl_mpi& other) = delete;
    atl_mpi& operator=(const atl_mpi& other) = delete;
    ~atl_mpi();

    atl_status_t init(int* argc,
                      char*** argv,
                      atl_attr_t* attr,
                      const char* main_addr,
                      std::shared_ptr<ipmi> pmi) override;

    atl_status_t update(std::shared_ptr<ipmi> pmi) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t mr_reg(const void* buf, size_t len, atl_mr_t** mr) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t mr_dereg(atl_mr_t* mr) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t send(atl_ep_t& ep,
                      const void* buf,
                      size_t len,
                      int dst_proc_idx,
                      uint64_t tag,
                      atl_req_t& req) override;

    atl_status_t recv(atl_ep_t& ep,
                      void* buf,
                      size_t len,
                      int src_proc_idx,
                      uint64_t tag,
                      atl_req_t& req) override;

    atl_status_t probe(atl_ep_t& ep,
                       int src_proc_idx,
                       uint64_t tag,
                       int* found,
                       size_t* recv_len) override;

    atl_status_t allgather(atl_ep_t& ep,
                           const void* send_buf,
                           void* recv_buf,
                           size_t len,
                           atl_req_t& req) override;

    atl_status_t allgatherv(atl_ep_t& ep,
                            const void* send_buf,
                            size_t send_len,
                            void* recv_buf,
                            const size_t* recv_lens,
                            const size_t* offsets,
                            atl_req_t& req) override;

    atl_status_t allreduce(atl_ep_t& ep,
                           const void* send_buf,
                           void* recv_buf,
                           size_t len,
                           atl_datatype_t dtype,
                           atl_reduction_t op,
                           atl_req_t& req) override;

    atl_status_t alltoall(atl_ep_t& ep,
                          const void* send_buf,
                          void* recv_buf,
                          int len,
                          atl_req_t& req) override;

    atl_status_t alltoallv(atl_ep_t& ep,
                           const void* send_buf,
                           const size_t* send_lens,
                           const size_t* send_offsets,
                           void* recv_buf,
                           const size_t* recv_lens,
                           const size_t* recv_offsets,
                           atl_req_t& req) override;

    atl_status_t barrier(atl_ep_t& ep, atl_req_t& req) override;

    atl_status_t bcast(atl_ep_t& ep, void* buf, size_t len, int root, atl_req_t& req) override;
    atl_status_t broadcast(atl_ep_t& ep,
                           void* send_buf,
                           void* recv_buf,
                           size_t len,
                           int root,
                           atl_req_t& req) override;

    atl_status_t reduce(atl_ep_t& ep,
                        const void* send_buf,
                        void* recv_buf,
                        size_t len,
                        int root,
                        atl_datatype_t dtype,
                        atl_reduction_t op,
                        atl_req_t& req) override;

    atl_status_t reduce_scatter(atl_ep_t& ep,
                                const void* send_buf,
                                void* recv_buf,
                                size_t recv_len,
                                atl_datatype_t dtype,
                                atl_reduction_t op,
                                atl_req_t& req) override;

    atl_status_t read(atl_ep_t& ep,
                      void* buf,
                      size_t len,
                      atl_mr_t* mr,
                      uint64_t addr,
                      uintptr_t remote_key,
                      int dst_proc_idx,
                      atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t write(atl_ep_t& ep,
                       const void* buf,
                       size_t len,
                       atl_mr_t* mr,
                       uint64_t addr,
                       uintptr_t remote_key,
                       int dst_proc_idx,
                       atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t wait(atl_ep_t& ep, atl_req_t& req) override;

    atl_status_t wait_all(atl_ep_t& ep, std::vector<atl_req_t>& reqs, size_t count) override;

    atl_status_t cancel(atl_ep_t& ep, atl_req_t& req) override;

    atl_status_t poll(atl_ep_t& ep) override;

    atl_status_t check(atl_ep_t& ep, atl_req_t& req) override;

    atl_proc_coord_t create_proc_coord(atl_ep_t& ep) override;
    atl_proc_coord_t create_proc_coord(MPI_Comm comm);

    void comms_free(std::vector<atl_ep_t>& eps) override;

    atl_status_t comm_split(const std::vector<atl_ep_t>& base_eps,
                            std::vector<atl_ep_t>& eps,
                            size_t color,
                            int key,
                            int local_idx,
                            int local_count) override;

    atl_status_t get_rank2proc_map(std::shared_ptr<ipmi> pmi,
                                   std::vector<int>& rank2proc_map,
                                   atl_proc_coord_t coord) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    std::string to_string() override;

    atl_status_t finalize(int global_idx = 0) override;

    static void get_mpi_ranks(const int rank,
                              const int size,
                              const int root_rank,
                              MPI_Comm comm,
                              std::vector<int>& mpi_ranks);

    atl_status_t comm_create(int comm_size,
                             const std::vector<int>& comm_ranks,
                             std::shared_ptr<ipmi> pmi,
                             MPI_Comm* new_comm);

    atl_status_t ep_init(std::vector<atl_ep_t>& eps,
                         MPI_Comm global_comm,
                         int local_idx,
                         int local_count);

    static void set_env(const atl_attr_t& attr);
    static atl_mpi_env_info_t get_env_info(const char* key);
    static atl_mpi_comm_info_t get_comm_info(MPI_Comm comm, const char* key);

private:
    MPI_Datatype atl2mpi_dtype(atl_datatype_t dtype);
    MPI_Op atl2mpi_op(atl_reduction_t rtype, MPI_Datatype dtype);

    void init_req(atl_req_t& req);
    inline atl_status_t progress_ep(atl_ep_t& ep, atl_mpi_req_t* req);

    void check_comm_nic_idx(MPI_Comm comm, size_t expected_idx);
    void check_comm_ep_idx(MPI_Comm comm, size_t expected_idx);
    void check_comm_info(MPI_Comm comm, const char* key, const char* expected_value);
    void check_ep(atl_ep_t& ep);

    size_t get_ep_idx(size_t ep_idx);

    atl_mpi_ctx ctx;
};
#endif // CCL_ENABLE_MPI
