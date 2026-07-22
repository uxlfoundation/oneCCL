#!/bin/bash

# This test is checking if oneCCL is able to run without 
# CCL_ROOT variable set.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="allreduce_events_inorder"

source "${ROOT_DIR}/utils.sh"

make_common_actions "${SCRIPT_DIR}" "${TEST_LOG}" "sycl"

function run_cmd() {
    if ! "$@"; then
        echo "run_cmd: $*"
        echo "Fail"
        exit 1
    fi
}

ccl_root_copy=${CCL_ROOT}

echo "Removing CCL_ROOT=${CCL_ROOT}" > "${TEST_LOG}"
unset CCL_ROOT

run_cmd env CCL_LOG_LEVEL=debug mpiexec -n 2 "${SCRIPT_DIR}/${BINFILE}" --count=1024 --random >> "${TEST_LOG}"

export CCL_ROOT=$ccl_root_copy
echo "Reverting CCL_ROOT=${CCL_ROOT}" >> "${TEST_LOG}"

check_log "${TEST_LOG}"
rm "${TEST_LOG}"
echo "Pass"
