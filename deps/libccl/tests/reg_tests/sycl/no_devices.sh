#!/bin/bash

# This test is checking if oneCCL is able to run without 
# any sycl devices available on the system to run cpu
# comunicators.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${SCRIPT_DIR}/../../../build/_install/examples/cpu/cpu_allreduce_test"

source "${ROOT_DIR}/utils.sh"

make_common_actions "${SCRIPT_DIR}" "${TEST_LOG}" "sycl"

function run_cmd() {
    if ! "$@"; then
        echo "run_cmd: $*"
        echo "Fail"
        exit 1
    fi
}

true > "${TEST_LOG}"

# We are using ONEAPI_DEVICE_SELECTOR="cuda:*", because our systems do not have 
# GPU's supporting cuda. This way we can ensure that no device is detected by
# sycl runtime.
run_cmd env ONEAPI_DEVICE_SELECTOR="cuda:*" mpiexec -n 2 "${BINFILE}"  >> "${TEST_LOG}"

check_log "${TEST_LOG}"
rm "${TEST_LOG}"
echo "Pass"
