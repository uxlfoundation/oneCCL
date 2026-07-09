#!/bin/bash

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

rm "${TEST_LOG}" 2>/dev/null

for count in 1 2 4 5 6 15 17 25 30 37 67 78 90 123 200 253 300 512 567 800 1024 1062; do
    for pc in 2 4; do
        run_cmd mpiexec -n "$pc" ${SCRIPT_DIR}/${BINFILE} --count "$count" --random >> "${TEST_LOG}"
    done
done

check_log "${TEST_LOG}"
echo "Pass"
