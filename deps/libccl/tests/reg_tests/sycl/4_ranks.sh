#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

proc_counts="1 2 4"
transports="ofi mpi"

bench_options="-i 2 -w 0 -c all -l all -b sycl -t 131072 $(get_default_bench_dtype)"

for proc_count in ${proc_counts}
do
    for transport in ${transports}
    do
        export CCL_ATL_TRANSPORT=${transport}
        mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark ${bench_options} > ${TEST_LOG} 2>&1
        rc=$?
        if [ ${rc} -ne 0 ]
        then
            echo "Fail: '${transport}'"
            exit 1
        fi
        check_log ${TEST_LOG}
    done
done

echo "Pass"
