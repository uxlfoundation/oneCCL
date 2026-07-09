#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

proc_counts="2 4"
inplace_modes="0 1"

bench_options="-d bfloat16 -b sycl -w 1 -i 2 -c all"
bench_options+=" -l allreduce -y 1,7,8,16,17,64,133,1077,16384,65539,131072,133073"

for proc_count in ${proc_counts}
do
    cmd="mpiexec -l -n ${proc_count}"
    cmd+=" ${SCRIPT_DIR}/${BINFILE}"
    cmd+=" > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"
    check_log ${TEST_LOG}

    for inplace_mode in ${inplace_modes}
    do
        cmd="mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
        cmd+=" ${bench_options} -q ${inplace_mode}"
        cmd+=" > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm -f ${BINFILE} ${TEST_LOG}
echo "Pass"
