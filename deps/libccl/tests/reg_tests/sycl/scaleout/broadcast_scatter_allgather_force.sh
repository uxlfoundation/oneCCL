#!/bin/bash

# Force mode is a truth-mode benchmark knob: if Scatter-AllGather (SAG) cannot
# run, the benchmark must fail loudly instead of silently measuring fallback.
#
# This test uses p2p_level=none to make p2p_comm->size()=1, which fails the SAG
# precondition check in broadcast_sycl_multi_domain. With FORCE=1 that silent
# fallback becomes a hard throw. The test verifies the process exits non-zero
# and the expected error message appears in the log.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export I_MPI_JOB_TIMEOUT=120

cmd="CCL_SYCL_FORCE_PCIE=1"
cmd+=" CCL_SYCL_P2P_LEVEL=none"
cmd+=" CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_ENABLE=1"
cmd+=" CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_FORCE=1"
cmd+=" CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_THRESHOLD=0"
cmd+=" CCL_LOG_LEVEL=warn"
cmd+=" mpiexec -l -n 2 -ppn 1 ${SCRIPT_DIR}/benchmark"
cmd+=" -a gpu -m usm -u device -l broadcast"
cmd+=" -i 1 -w 0 -y 4096"
cmd+=" -j off --check last -d int32 -e in_order"
cmd+=" > ${TEST_LOG} 2>&1"

echo "  |TEST CMD| ${cmd}"
run_negative_cmd "${cmd}"

expected_msg="forced broadcast scatter-allgather cannot run"
if ! grep -q "${expected_msg}" ${TEST_LOG}; then
    echo "Error: did not find expected force-mode failure in ${TEST_LOG}"
    echo "Expected pattern: ${expected_msg}"
    cat ${TEST_LOG}
    exit 1
fi

rm ${TEST_LOG}
echo "Pass"
