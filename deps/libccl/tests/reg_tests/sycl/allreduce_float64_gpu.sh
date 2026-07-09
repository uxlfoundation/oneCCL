#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

# single node test
cmd=" mpiexec -l -n 2 ${SCRIPT_DIR}/benchmark"
cmd+=" -a gpu -m usm -u device -l allreduce"
cmd+=" -i 1 -w 0 -t 67108864 -f 1 -j off -p 0"
cmd+=" -d float64 -e in_order --check last"
cmd+=" > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_log ${TEST_LOG}

rm ${TEST_LOG}
echo "Pass"

