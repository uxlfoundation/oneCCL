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

#include "common/utils/tree.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/sched.hpp"

static void bcast_tree(const ccl_bin_tree& tree,
                       ccl_sched* sched,
                       ccl_buffer buffer,
                       size_t count,
                       const ccl_datatype& dtype,
                       ccl_comm* comm) {
    if (tree.parent() != -1) {
        LOG_DEBUG("recv from parent ", tree.parent());
        entry_factory::create<recv_entry>(
            sched, buffer, count, dtype, static_cast<size_t>(tree.parent()), comm);
        sched->add_barrier();
    }
    if (tree.left() != -1) {
        LOG_DEBUG("send to left ", tree.left());
        entry_factory::create<send_entry>(
            sched, buffer, count, dtype, static_cast<size_t>(tree.left()), comm);
    }
    if (tree.right() != -1) {
        LOG_DEBUG("send to right ", tree.right());
        entry_factory::create<send_entry>(
            sched, buffer, count, dtype, static_cast<size_t>(tree.right()), comm);
    }
}

static void reduce_tree(const ccl_bin_tree& tree,
                        ccl_sched* sched,
                        ccl_buffer buffer,
                        size_t count,
                        const ccl_datatype& dtype,
                        ccl::reduction reduction,
                        ccl_comm* comm) {
    if (tree.left() != -1) {
        LOG_DEBUG("recv_reduce left ", tree.left());
        entry_factory::create<recv_reduce_entry>(
            sched, buffer, count, dtype, reduction, static_cast<size_t>(tree.left()), comm);
    }
    if (tree.right() != -1) {
        LOG_DEBUG("recv_reduce right ", tree.right());
        entry_factory::create<recv_reduce_entry>(
            sched, buffer, count, dtype, reduction, static_cast<size_t>(tree.right()), comm);
    }
    if (tree.parent() != -1) {
        if (tree.left() != -1 || tree.right() != -1) {
            sched->add_barrier();
        }
        LOG_DEBUG("send to parent ", tree.parent());
        entry_factory::create<send_entry>(
            sched, buffer, count, dtype, static_cast<size_t>(tree.parent()), comm);
    }
}

static void reduce_bcast_tree(const ccl_bin_tree& tree,
                              ccl_sched* sched,
                              ccl_buffer buffer,
                              size_t count,
                              const ccl_datatype& dtype,
                              ccl::reduction reduction,
                              ccl_comm* comm) {
    if (tree.left() != -1) {
        LOG_DEBUG("recv_reduce left ", tree.left());
        entry_factory::create<recv_reduce_entry>(
            sched, buffer, count, dtype, reduction, static_cast<size_t>(tree.left()), comm);
    }
    if (tree.right() != -1) {
        LOG_DEBUG("recv_reduce right ", tree.right());
        entry_factory::create<recv_reduce_entry>(
            sched, buffer, count, dtype, reduction, static_cast<size_t>(tree.right()), comm);
    }
    if (tree.parent() != -1) {
        if (tree.left() != -1 || tree.right() != -1) {
            sched->add_barrier();
        }

        LOG_DEBUG("send to parent ", tree.parent());
        entry_factory::create<send_entry>(
            sched, buffer, count, dtype, static_cast<size_t>(tree.parent()), comm);

        LOG_DEBUG("recv from parent ", tree.parent());
        entry_factory::create<recv_entry>(
            sched, buffer, count, dtype, static_cast<size_t>(tree.parent()), comm);
    }

    if (tree.left() != -1 || tree.right() != -1) {
        sched->add_barrier();
    }

    if (tree.left() != -1) {
        LOG_DEBUG("send to left ", tree.left());
        entry_factory::create<send_entry>(
            sched, buffer, count, dtype, static_cast<size_t>(tree.left()), comm);
    }
    if (tree.right() != -1) {
        LOG_DEBUG("send to right ", tree.right());
        entry_factory::create<send_entry>(
            sched, buffer, count, dtype, static_cast<size_t>(tree.right()), comm);
    }
}

ccl::status ccl_coll_build_double_tree_op(ccl_sched* sched,
                                          ccl_coll_type coll_type,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction op,
                                          const ccl_double_tree& dtree,
                                          ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    LOG_DEBUG("build double tree ", ccl_coll_type_to_str(coll_type));

    if (coll_type != ccl_coll_bcast && send_buf != recv_buf) {
        LOG_DEBUG("out of place op");
        entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        sched->add_barrier();
    }

    size_t t1_count = count / 2;
    size_t t2_count = count - t1_count;

    ccl_buffer t1_start = recv_buf;
    ccl_buffer t1_end = t1_start + t1_count * dtype.size();

    ccl_buffer t2_start = t1_end;
    ccl_buffer t2_end = t2_start + t2_count * dtype.size();

    //todo: evaluate/configure k param;
    size_t parts = 1;
    if (ccl::global_data::env().dtree_partition_count != CCL_ENV_SIZET_NOT_SPECIFIED) {
        parts = ccl::global_data::env().dtree_partition_count;
    }
    LOG_DEBUG("double tree partition (k) count: ", parts);

    size_t t1_part_count = t1_count / parts;
    size_t t2_part_count = t2_count / parts;

    size_t t1_part_remainder = t1_count % parts;
    size_t t2_part_remainder = t2_count % parts;

    LOG_DEBUG("t1: start ",
              t1_start,
              ", end ",
              t1_end,
              ", count ",
              t1_count,
              ", part size ",
              t1_part_count,
              ", part remainder ",
              t1_part_remainder);

    LOG_DEBUG("t2: start ",
              t2_start,
              ", end ",
              t2_end,
              ", count ",
              t2_count,
              ", part size ",
              t2_part_count,
              ", part remainder ",
              t2_part_remainder);

    ccl_op_id_t t1_op_id = 0;
    ccl_op_id_t t2_op_id = t1_op_id + static_cast<ccl_op_id_t>(parts);

    ccl_buffer t1_work_buf = t1_start;
    ccl_buffer t2_work_buf = t2_start;

    for (size_t iter = 0; iter < parts; ++iter) {
        /* first t1_part_remainder partitions are slightly larger */
        size_t t1_work_count = t1_part_count;
        if (iter < t1_part_remainder) {
            t1_work_count++;
        }

        /* first t2_part_remainder partitions are slightly larger */
        size_t t2_work_count = t2_part_count;
        if (iter < t2_part_remainder) {
            t2_work_count++;
        }

        const auto& t1 = dtree.T1();
        const auto& t2 = dtree.T2();

        switch (coll_type) {
            case ccl_coll_bcast:
                entry_factory::create<subsched_entry>(
                    sched,
                    t1_op_id,
                    [t1_work_buf, t1_work_count, dtype, t1, comm](ccl_sched* s) {
                        bcast_tree(t1, s, t1_work_buf, t1_work_count, dtype, comm);
                    },
                    "bcast_t1");

                entry_factory::create<subsched_entry>(
                    sched,
                    t2_op_id,
                    [t2_work_buf, t2_work_count, dtype, t2, comm](ccl_sched* s) {
                        bcast_tree(t2, s, t2_work_buf, t2_work_count, dtype, comm);
                    },
                    "bcast_t2");

                break;
            case ccl_coll_broadcast:
                entry_factory::create<subsched_entry>(
                    sched,
                    t1_op_id,
                    [t1_work_buf, t1_work_count, dtype, t1, comm](ccl_sched* s) {
                        bcast_tree(t1, s, t1_work_buf, t1_work_count, dtype, comm);
                    },
                    "bcast_t1");

                entry_factory::create<subsched_entry>(
                    sched,
                    t2_op_id,
                    [t2_work_buf, t2_work_count, dtype, t2, comm](ccl_sched* s) {
                        bcast_tree(t2, s, t2_work_buf, t2_work_count, dtype, comm);
                    },
                    "bcast_t2");

                break;
            case ccl_coll_reduce: {
                if (comm->rank() % 2 == 0) {
                    //even ranks are leaves in T2, start schedule with T2
                    entry_factory::create<subsched_entry>(
                        sched,
                        t2_op_id,
                        [t2_work_buf, t2_work_count, dtype, op, t2, comm](ccl_sched* s) {
                            reduce_tree(t2, s, t2_work_buf, t2_work_count, dtype, op, comm);
                        },
                        "reduce_t2");

                    entry_factory::create<subsched_entry>(
                        sched,
                        t1_op_id,
                        [t1_work_buf, t1_work_count, dtype, op, t1, comm](ccl_sched* s) {
                            reduce_tree(t1, s, t1_work_buf, t1_work_count, dtype, op, comm);
                        },
                        "reduce_t1");
                }
                else {
                    entry_factory::create<subsched_entry>(
                        sched,
                        t2_op_id,
                        [t2_work_buf, t2_work_count, dtype, op, t2, comm](ccl_sched* s) {
                            reduce_tree(t2, s, t2_work_buf, t2_work_count, dtype, op, comm);
                        },
                        "reduce_t2");

                    entry_factory::create<subsched_entry>(
                        sched,
                        t1_op_id,
                        [t1_work_buf, t1_work_count, dtype, op, t1, comm](ccl_sched* s) {
                            reduce_tree(t1, s, t1_work_buf, t1_work_count, dtype, op, comm);
                        },
                        "reduce_t1");
                }

                break;
            }
            case ccl_coll_allreduce: {
                if (comm->rank() % 2 == 0) {
                    //even ranks are leaves in T2, start schedule with T2
                    entry_factory::create<subsched_entry>(
                        sched,
                        t2_op_id,
                        [t2_work_buf, t2_work_count, dtype, op, t2, comm](ccl_sched* s) {
                            reduce_bcast_tree(t2, s, t2_work_buf, t2_work_count, dtype, op, comm);
                        },
                        "reduce_bcast_t2");

                    entry_factory::create<subsched_entry>(
                        sched,
                        t1_op_id,
                        [t1_work_buf, t1_work_count, dtype, op, t1, comm](ccl_sched* s) {
                            reduce_bcast_tree(t1, s, t1_work_buf, t1_work_count, dtype, op, comm);
                        },
                        "reduce_bcast_t1");
                }
                else {
                    entry_factory::create<subsched_entry>(
                        sched,
                        t1_op_id,
                        [t1_work_buf, t1_work_count, dtype, op, t1, comm](ccl_sched* s) {
                            reduce_bcast_tree(t1, s, t1_work_buf, t1_work_count, dtype, op, comm);
                        },
                        "reduce_bcast_t1");

                    entry_factory::create<subsched_entry>(
                        sched,
                        t2_op_id,
                        [t2_work_buf, t2_work_count, dtype, op, t2, comm](ccl_sched* s) {
                            reduce_bcast_tree(t2, s, t2_work_buf, t2_work_count, dtype, op, comm);
                        },
                        "reduce_bcast_t2");
                }
                break;
            }
            default: CCL_FATAL("unsupported double tree op ", ccl_coll_type_to_str(coll_type));
        }

        /* calculate buffer for next iteration */
        t1_work_buf += t1_work_count * dtype.size();
        t2_work_buf += t2_work_count * dtype.size();

        ++t1_op_id;
        ++t2_op_id;
    }

    sched->add_barrier();

    return status;
}
