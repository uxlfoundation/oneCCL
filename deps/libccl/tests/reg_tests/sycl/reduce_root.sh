#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

proc_counts=(4)

for proc_count in ${proc_counts[@]}
do
    ppn=$((${proc_count}/2))

    cmd="CCL_REDUCE=topo mpiexec -l -n ${proc_count} -ppn ${ppn} ${SCRIPT_DIR}/${BINFILE} gpu device 0 > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"
    check_log ${TEST_LOG}

    cmd="CCL_REDUCE=topo mpiexec -l -n ${proc_count} -ppn ${ppn} ${SCRIPT_DIR}/${BINFILE} gpu device ${ppn} > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"
    check_log ${TEST_LOG}
done

rm ${TEST_LOG}
echo "Pass"
