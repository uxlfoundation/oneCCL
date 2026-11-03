#!/bin/bash

# Verifies invoke_p2p_barrier_pair() barrier between any 2 arbitrary ranks.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
# utils.sh lives at tests/reg_tests/utils.sh; internal_tests is two levels deep.
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

# Number of ranks: use first script argument if given, otherwise auto-detect
# the number of Level Zero GPUs reported by sycl-ls (one rank per GPU).
if [[ -n "$1" ]]; then
    NUM_RANKS="$1"
else
    if command -v sycl-ls >/dev/null 2>&1; then
        NUM_RANKS=$(sycl-ls 2>/dev/null | grep -c '^\[level_zero:gpu\]')
    else
        NUM_RANKS=0
    fi
    if [[ "${NUM_RANKS}" -lt 2 ]]; then
        echo "Could not detect >= 2 Level Zero GPUs via sycl-ls; falling back to 2 ranks."
        NUM_RANKS=2
    fi
fi

echo "Running p2p_barrier_pair with ${NUM_RANKS} ranks"

cmd=" mpiexec -l -n ${NUM_RANKS} ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_log ${TEST_LOG}

rm ${TEST_LOG}
echo "Pass"
