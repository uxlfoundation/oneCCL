#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_ALLTOALLV=topo
export CCL_ZE_AUTO_TUNE_PORTS=0

bench_options="-w 0 -i 1 -j off -b sycl -y 1,7,8,9,10,15,16,17,31,32,33,63,64,65,127,128,129,254,255,256,511,512,513,1000,1023,1024,1025,2000,10000,33554431,33554432,33554433 -d all "

proc_counts="4"

for proc_count in ${proc_counts}
do

    # basic tests
    cmd=" CCL_ENABLE_SYCL_KERNELS=1"
    cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
    cmd+=" ${bench_options} -l all"
    cmd+=" > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"
    check_log ${TEST_LOG}

    alltoall_rw_protocols="read write"

    for alltoall_rw_protocol in ${alltoall_rw_protocols}
    do
        cmd="CCL_SYCL_ALLTOALL_PROTOCOL=${alltoall_rw_protocol}"
        cmd+=" CCL_ENABLE_SYCL_KERNELS=1"
        cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
        cmd+=" ${bench_options} -l alltoall"
        cmd+=" > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm ${TEST_LOG}
echo "Pass"
