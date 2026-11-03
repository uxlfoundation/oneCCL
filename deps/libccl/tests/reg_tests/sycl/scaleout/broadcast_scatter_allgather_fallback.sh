#!/bin/bash

# Exercise the SAG fallback path: ENABLE=1 but message is below the threshold,
# so SAG returns done=false and the legacy broadcast path handles it.
# Verifies that the fallback produces correct data (check=last).
#
# Shape: 2 nodes x 2 ranks/node = 4 ranks total, -ppn 2.
#   p2p_comm  = node comm, size 2  (satisfies p2p_comm->size() >= 2)
#   r2r_comm  = rank-to-rank comm, size 2  (satisfies r2r_comm->size() >= 2)
# A threshold of 512MB with a 4096-element int32 payload (16KB) ensures the
# message is always below threshold and SAG always falls back to legacy.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export I_MPI_JOB_TIMEOUT=120

cmd="CCL_SYCL_FORCE_PCIE=1"
cmd+=" CCL_SYCL_P2P_LEVEL=node"
cmd+=" CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_ENABLE=1"
cmd+=" CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_FORCE=0"
cmd+=" CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_THRESHOLD=536870912"
cmd+=" CCL_LOG_LEVEL=warn"
cmd+=" mpiexec -l -n 4 -ppn 2 ${SCRIPT_DIR}/benchmark"
cmd+=" -a gpu -m usm -u device -l broadcast"
cmd+=" -i 2 -w 1 -y 1,17,4096"
cmd+=" -j off --check last -d int32 -e in_order"
cmd+=" > ${TEST_LOG} 2>&1"

echo "  |TEST CMD| ${cmd}"
run_cmd "${cmd}"
check_log ${TEST_LOG}

rm ${TEST_LOG}
echo "Pass"
