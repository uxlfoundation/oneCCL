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

#include "common/log/log.hpp"
#include "common/utils/utils.hpp"

#include <tuple>

using ccl_op_id_t = uint8_t;
using ccl_sched_id_t = uint16_t;
using ccl_comm_id_t = uint16_t;

// A list of observed distinct provider specific number of bits used for the tag
enum tag_layout : unsigned int { mpich = 17, impi = 24, cxi = 48, common = 64 };

// Point-to-point tag configuration.
// Adjusted to the maximum ccl_sched_id_t value
// that can be encoded into the tag
template <unsigned int N = 16>
struct pt2pt_tag_layout {
    static_assert(N > 0 && N <= sizeof(ccl_sched_id_t) * 8,
                  "the number of bits should not exceed the size of ccl_sched_id_t type");
    // We have declared the tag using the data type ccl_sched_id_t,
    // however, it can be equivalent to a N-bit unsigned integer. For pt2pt,
    // we use the maximum value that can be represented with these N bits as the tag.
    static constexpr ccl_sched_id_t pt2pt_sched_id = (1 << N) - 1;
    static_assert(pt2pt_sched_id <= std::numeric_limits<ccl_sched_id_t>::max(),
                  "pt2pt_sched_id should not exceed the max value of ccl_sched_id_t");
    // these tags are reserved for pt2pt ack messages to align topo pt2pt operations
    static constexpr ccl_sched_id_t pt2pt_ack_tag = pt2pt_sched_id - 1;
    static constexpr ccl_sched_id_t pt2pt_ack_first = pt2pt_sched_id - 2;
    static constexpr ccl_sched_id_t pt2pt_ack_second = pt2pt_sched_id - 3;
    // maximum value of schedule id in scope of the current communicator
    static constexpr ccl_sched_id_t max_sched_count = pt2pt_sched_id - 4;
};

/*
 * Common 64-bit layout suitable for the most cases
 */
/**********************************************************************************
 * common tag layout                                                              *
 * ********************************************************************************
 * 01234567 01234567 01234567 | 01234567 01234567 | 01234567 01234567  | 01234567 |
 *                            |                   |                    |          |
 *           rank             |      comm_id      |       sched_id     |   op_id  |
 *********************************************************************************/
struct common_tag_layout : pt2pt_tag_layout<16> {
    static constexpr int op_id_shift = 0;
    static constexpr int sched_id_shift = 8;
    static constexpr int comm_id_shift = 24;
    static constexpr int rank_shift = 40;

    static constexpr uint64_t op_id_mask = 0x00000000000000FF;
    static constexpr uint64_t sched_id_mask = 0x0000000000FFFF00;
    static constexpr uint64_t comm_id_mask = 0x000000FFFF000000;
    static constexpr uint64_t rank_mask = 0xFFFFFF0000000000;
};

/*
 * CXI provider accepts 48-bits tag.
 * With the common layout it means, that we have to cut the most
 * significant bits in our tag for correctness. However, rank identifier
 * could then hold only 8-bits, therefore 256 ranks could be encoded in the tag.
 * If there is more then 256 ranks communicating with each other, tags
 * may coalesce, leading to the wrong messages accepted in a place,
 * which will be hard to track. 2^20 number of ranks should be sufficient,
 * other bit fields were cut for reasonable values.
 */
/******************************************************************
 * cxi tag layout                                                 *
 * ****************************************************************
 * 0123 01234567 01234567 | 01234567 0123 | 01234567 0123 | 0123  |
 *                        |               |               |       |
 *         rank           |    comm_id    |    sched_id   | op_id |
 ******************************************************************/
struct ofi_cxi_tag_layout : pt2pt_tag_layout<12> {
    static constexpr int op_id_shift = 0;
    static constexpr int sched_id_shift = 4;
    static constexpr int comm_id_shift = 16;
    static constexpr int rank_shift = 28;

    static constexpr uint64_t op_id_mask = 0x000000000000000F;
    static constexpr uint64_t sched_id_mask = 0x000000000000FFF0;
    static constexpr uint64_t comm_id_mask = 0x000000000FFF0000;
    static constexpr uint64_t rank_mask = 0x0000FFFFF0000000;
};

/*
 * MPI standart requires the tag to be at most 32-bit integer number.
 * However, also the tag should be no less then 16-bit value.
 * However, since the rank and the comm and
 * TODO: support MPICH and I_MPI layout separately.
 */
/************************************************
 * mpi tag layout                               *
 * **********************************************
 * 01234567 | 01234567 | 01234567 0123 | 0123  |
 *          |          |               |       |
 *   rank   | comm_id  |   sched_id    | op_id |
 ************************************************/
struct mpi_tag_layout : pt2pt_tag_layout<12> {
    static constexpr int op_id_shift = 0;
    static constexpr int sched_id_shift = 4;
    static constexpr int comm_id_shift = 16;
    static constexpr int rank_shift = 24;

    static constexpr uint64_t op_id_mask = 0x000000000000000F;
    static constexpr uint64_t sched_id_mask = 0x000000000000FFF0;
    static constexpr uint64_t comm_id_mask = 0x0000000000FF0000;
    static constexpr uint64_t rank_mask = 0x00000000FF000000;
};

class ccl_atl_tag {
public:
    ccl_atl_tag(size_t tag_bits, size_t max_tag) : tag_bits{ tag_bits }, max_tag{ max_tag } {
        CCL_THROW_IF_NOT(tag_bits >= 32, "unexpected tag_bits ", tag_bits);
        CCL_ASSERT(sizeof(ccl_op_id_t) == 1);
        CCL_ASSERT(sizeof(ccl_sched_id_t) <= 2);
        CCL_ASSERT(sizeof(ccl_comm_id_t) <= 2);

        if (max_tag == ccl::utils::pof2(max_tag) * 2 - 1)
            max_tag_mask = max_tag;
        else
            max_tag_mask = ccl::utils::pof2(max_tag) - 1;
    }

    ccl_atl_tag(const ccl_atl_tag& other) = delete;
    ccl_atl_tag(ccl_atl_tag&& other) = delete;

    ccl_atl_tag& operator=(const ccl_atl_tag& other) = delete;
    ccl_atl_tag& operator=(ccl_atl_tag&& other) = delete;

    virtual ~ccl_atl_tag() = default;

    std::string to_string() const;

    /**
     * Generates the tag to be used by ATL communication operations
     * @param rank identifier of the rank within communicator
     * @param comm_id identifier of the communicator
     * @param sched_id identifier of the schedule within communicator
     * @param op_id local operation id, used as sub-schedule identifier
     * @return ATL communication tag
     */
    virtual uint64_t create(int rank,
                            ccl_comm_id_t comm_id,
                            ccl_sched_id_t sched_id,
                            ccl_op_id_t op_id = 0) = 0;

    // Point-to-point config data accessors
    virtual ccl_sched_id_t get_max_sched_count() = 0;
    virtual ccl_sched_id_t get_pt2pt_sched_id() = 0;
    virtual ccl_sched_id_t get_pt2pt_ack_tag() = 0;
    virtual std::tuple<ccl_sched_id_t, ccl_sched_id_t> get_pt2pt_sync_tags() = 0;

protected:
    size_t tag_bits;
    size_t max_tag;
    size_t max_tag_mask;
};

template <typename Layout>
class ccl_atl_tag_impl : public ccl_atl_tag {
public:
    ccl_atl_tag_impl(size_t tag_bits, size_t max_tag) : ccl_atl_tag{ tag_bits, max_tag } {}

    uint64_t create(int rank,
                    ccl_comm_id_t comm_id,
                    ccl_sched_id_t sched_id,
                    ccl_op_id_t op_id = 0);

    ccl_sched_id_t get_max_sched_count();
    ccl_sched_id_t get_pt2pt_sched_id();
    ccl_sched_id_t get_pt2pt_ack_tag();
    std::tuple<ccl_sched_id_t, ccl_sched_id_t> get_pt2pt_sync_tags();
};
