#!/bin/bash
#
# Re-runs the pt2pt_gpu test under CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=0 to exercise the
# send_simple/recv_simple dispatch path through the public ccl::send/recv
# API. On ARC this hits the new fused single-kernel protocol; on non-ARC HW
# is_arc_card() is false so the existing path runs and the test still
# passes (the env knob is harmlessly read).

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
