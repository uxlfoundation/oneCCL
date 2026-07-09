#!/bin/bash

# This test is aimed to verify that 8 GB allreduce doesn't hang up: MLSL-2340

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

cmd=" mpiexec -l -n 2 ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_log ${TEST_LOG}

rm ${TEST_LOG}
echo "Pass"
