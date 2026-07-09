#!/bin/bash

# TODO: make this test run on cpu as well and move
# it to common directory

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

bench_options="-b sycl -w 2 -i 4 -c all -l allreduce -y 1000 $(get_default_bench_dtype)"

export CCL_ITT_LEVEL=1

for worker_offload in 0 1; do
    export CCL_WORKER_OFFLOAD=${worker_offload}
    mpiexec -l -n 2 ${SCRIPT_DIR}/benchmark ${bench_options} > ${TEST_LOG} 2>&1
    ret_val=$?

    if [ $ret_val -ne 0 ]
    then
        echo "Fail"
        exit -1
    fi

    check_log ${TEST_LOG}
done

rm ${TEST_LOG}
echo "Pass"
