#!/bin/bash

# Verifies send_simple()/recv_simple() round-trip with iterated staging-slot
# reuse. The .cpp calls the simple kernels directly through internal headers,
# bypassing the is_arc_card() dispatch gate so it runs on any SYCL GPU.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname $(dirname "${SCRIPT_DIR}"))"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

source ${ROOT_DIR}/utils.sh

# Preamble equivalent to utils.sh's make_common_actions, minus the
# get_bench step. The internal-tests gtest binaries are built by the
# main libccl tree; we don't need the shared examples/benchmark/benchmark
# binary that get_bench would otherwise try to (re)build.
export NO_COLOR=1
check_impi
check_ccl
cd ${SCRIPT_DIR}

# Test only uses 2 participants (sender + receiver). Allow override for
# larger jobs (extra ranks just sit at MPI_Barrier, harmless smoke test).
NUM_RANKS="${1:-2}"

echo "Running p2p_simple_pair with ${NUM_RANKS} ranks"

cmd=" mpiexec -l -n ${NUM_RANKS} ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_log ${TEST_LOG}

rm ${TEST_LOG}
echo "Pass"
