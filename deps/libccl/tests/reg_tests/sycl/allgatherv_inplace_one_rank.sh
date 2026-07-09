#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
BINFILE="${BASENAME}"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_SYCL_OUTPUT_EVENT=1

op_sync_modes="0 1"
in_order="0 1"

for op_sync in $op_sync_modes
do
    for order in $in_order
    do
       cmd="CCL_OP_SYNC=${op_sync}"
       TEST_LOG="${BASENAME}_op_sync_${op_sync}_inorder_${order}.log"
       cmd+=" mpiexec -n 1 ${SCRIPT_DIR}/${BINFILE} ${order} > ${TEST_LOG} 2>&1"
       run_cmd "${cmd}"
       check_log ${TEST_LOG}
    done
done

rm -f ${BINFILE} ${TEST_LOG}
echo "Pass"
