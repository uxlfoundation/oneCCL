#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

# This test disables the check of oversubscription case
# topo_manager detects oversubscription case on jfsdp host,
# where 2 devices are only. And run copy in/out algos
export CCL_ZE_ENABLE_OVERSUBSCRIPTION_THROW=0

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"
# need to use proc_count > device count
# in CI we have lrz - 8 devices and sdp - 4 devices
proc_counts="10"
transports="ofi mpi"

bench_options="-b sycl -w 1 -i 2 -y 1,7,8,16,17,64,133,1077,16384,65539,131072,133073"
bench_options+=" -c all -l all -d int8,int32,float16,float32,bfloat16"

for proc_count in ${proc_counts}
do
    for transport in ${transports}
    do
        cmd="CCL_ATL_TRANSPORT=${transport}"
        cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
        cmd+=" ${bench_options} > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm ${TEST_LOG}
echo "Pass"
