#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
BINFILE="${BASENAME}"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"
ranks_list="2 4"

export ZE_AFFINITY_MASK=0,1,2,3
for ranks in ${ranks_list}
do
    TEST_LOG="${BASENAME}_${ranks}_ranks.log"
    cmd="mpiexec -n ${ranks} -ppn 1 ${SCRIPT_DIR}/${BINFILE} gpu > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"
    check_log ${TEST_LOG}
done

rm -f ${BINFILE} ${TEST_LOG}
echo "Pass"
