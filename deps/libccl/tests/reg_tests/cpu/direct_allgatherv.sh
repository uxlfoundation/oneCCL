#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE=${BASENAME}

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

transports="ofi"
proc_counts="2 4"

bench_options="-w 1 -i 4 -l allgatherv -c all -b host -t 1048576 $(get_default_bench_dtype)"

for transport in ${transports}
do
    for proc_count in ${proc_counts}
    do
        cmd="CCL_ATL_TRANSPORT=${transport}"
        cmd+=" CCL_ALLGATHERV=direct"
        cmd+=" mpiexec -l -n ${proc_count}"
        cmd+=" ${SCRIPT_DIR}/benchmark ${bench_options} > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm ${TEST_LOG}
echo "Pass"
