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

#include <cerrno>
#include <cmath>
#include <sycl/sycl.hpp>

#include "coll/algorithms/utils/check_zero_buffers.hpp"
#include "coll/algorithms/utils/tvisa/include/gen_visa_templates.hpp"
#include "coll/algorithms/utils/transmit/transmit.hpp"

#include "coll/algorithms/utils/transmit/work_size_data.hpp"

template <typename T,
          template <typename, int>
          class Proto,
          bool LoadSeqNoInRuntime,
          int SubGroupSize,
          template <typename, template <typename, int> class, bool, int>
          class Transmit>
class oneccl_reduce_pcie {};

template <typename T,
          template <typename, int>
          class Proto,
          bool LoadSeqNoInRuntime,
          int SubGroupSize,
          template <typename, template <typename, int> class, bool, int>
          class Transmit>
class oneccl_reduce_counter_pcie {};

template <typename T,
          template <typename, int>
          class Proto,
          template <typename, template <typename, int> class, bool, int>
          class Transmit,
          bool LoadSeqNoInRuntime,
          int SubGroupSize = 16>
struct Reduce {
    using TransmitT = Transmit<T, Proto, LoadSeqNoInRuntime, SubGroupSize>;
    using ProtoT = Proto<T, SubGroupSize>;
    using message_t = typename ProtoT::message_t;
    using WorkSizeDataT = WorkSizeData<message_t>;

    Reduce(int nranks,
              T* input,
              T* output,
              size_t nelems,
              int root,
              int rank,
              uint32_t* seqNoPtr,
              uint32_t seqNo,
              T* scatterBuf,
              T* gatherBuf,
              size_t ipcbuf_size,
              T* const peerBuf0[],
              T* const peerBuf1[],
              ccl::reduction reduction,
              bool p2p);


    std::pair<sycl::nd_range<1>, uint64_t> getLaunchParam(sycl::queue q,
                                                          const std::shared_ptr<ccl_comm> comm,
                                                          T* ipcbuf0,
                                                          T* ipcbuf1) const;

    static sycl::event launch(LLPatternData ll_pattern_data,
                              int nranks,
                              T* input,
                              T* output,
                              T* ipcbuf0,
                              T* ipcbuf1,
                              size_t ipcbuf_size,
                              size_t tmp_buf_size,
                              T* const peerbuf0[],
                              T* const peerbuf1[],
                              size_t nelems,
                              ccl::reduction reduction,
                              int root,
                              int rank,
                              int peer_rank,
                              int global_current_id,
                              sycl::queue queue,
                              pattern_type type,
                              const std::shared_ptr<ccl_comm> comm,
                              bool p2p,
                              bool& done);
    //
    //
    // Found this analogy fascinating:
    //
    // Let correspond sub-group to wire, sequential guaranteed.
    // Bundle sub-groups(wires) into group(cable).
    //
    // Total cables will deliver the full capacity of single loop.
    //
    void operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (sycl::nd_item<1> pos) const;
private:
    static AccessGranularity calcAccessGranularity(T* input,
                                                   T* output,
                                                   size_t count,
                                                   size_t nranks);

    static WorkSizeDataT calcWorkSize(T* input, T* output, size_t size, int NRanks);
    
    // do not reorder items
    size_t rank;
    size_t nRanks;
    size_t ipcbuf_size;
    AccessGranularity gr;
    // remainder data
    WorkSizeDataT workSizeData;
    ccl::reduction reduction;
    int root;
    TransmitT transmit;
};
