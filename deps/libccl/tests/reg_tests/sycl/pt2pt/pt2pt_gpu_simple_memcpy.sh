#!/bin/bash
#
# Re-runs the pt2pt_gpu test under both CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=0 and
# CCL_SYCL_PT2PT_COPY_ENGINE=1 so the send_memcpy/recv_memcpy dispatch
# branch in send_sycl.cpp / recv_sycl.cpp is exercised through the public
# ccl::send/recv API. This is the only public test that drives the inner
# memcpy branch of the simple-protocol gate; without it, a regression in
# the env-var dispatch logic would slip past CI.
#
# On ARC this hits the q.memcpy + 2x invoke_p2p_barrier_pair path; on
# non-ARC HW is_arc_card() is false, the simple gate is skipped, and the
# existing LL256 path runs (env knobs harmlessly read).

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
# Re-use the already-built pt2pt_gpu binary; we only ship a different launcher.
BINFILE="pt2pt_gpu"
PATH_TO_BINFILE="${SCRIPT_DIR}/${BINFILE}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

export CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=0
export CCL_SYCL_PT2PT_COPY_ENGINE=1

# Minimal subset of pt2pt_gpu.sh that hits the topo single-node send path —
# the dispatch site we care about is gated on single-node + non-group + ARC.
cmd="CCL_ATL_TRANSPORT=mpi"
cmd+=" CCL_SEND=topo CCL_RECV=topo"
cmd+=" mpiexec -l -n 2 ${PATH_TO_BINFILE} --test_case 1 --queue 0"
cmd+=" --cache 0 --wait 1 --peers 1,0 --iter 4 > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_log ${TEST_LOG}

rm ${TEST_LOG}
echo "Pass"
