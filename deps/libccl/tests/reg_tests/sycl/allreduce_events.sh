#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
BINFILE="${BASENAME}"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_SYCL_OUTPUT_EVENT=1
export CCL_ZE_CLOSE_IPC_WA=1
export EnableDirectSubmission=1
export CCL_ZE_QUEUE_INDEX_OFFSET=1
export ZEX_NUMBER_OF_CCS=0:4,1:4

sync="0 1"

for op_sync in $sync
do
    export CCL_OP_SYNC=${op_sync}
    TEST_LOG="${BASENAME}_op_sync_${op_sync}.log"
    mpiexec -n 2 ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1
    rc=$?
    if [[ ${rc} -ne 0 ]]
    then
        echo "Fail"
        exit 1
    fi
    
    check_log ${TEST_LOG}
done

unset ZEX_NUMBER_OF_CCS
export CCL_ZE_QUEUE_INDEX_OFFSET=0
export CCL_OP_SYNC=1

mpiexec -n 2 ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1
rc=$?
if [[ ${rc} -ne 0 ]]
then
    echo "Fail"
    exit 1
fi
check_log ${TEST_LOG}

rm -f ${BINFILE} ${TEST_LOG}
echo "Pass"
