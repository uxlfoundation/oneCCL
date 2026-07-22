#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_ZE_AUTO_TUNE_PORTS=0
export CCL_YIELD=sched_yield

inplace_modes="0 1"
reduce_scatter_modes="0 1"
reduce_scatter_pipeline_modes="0 1"
allgatherv_monolithic_modes="0 1"
allgatherv_read_modes="0 1"
allgatherv_pipeline_modes="0 1"
reduce_scatter_read_modes="0 1"

bench_options="-b sycl -w 1 -i 2 -y 1,7,8,16,17,64,133,1077,16384,65539,131072,133073"
bench_options+=" -c all -l allreduce -d int8,int32,float16,float32,bfloat16"

for inplace_mode in ${inplace_modes}
do
    for reduce_scatter_mode in ${reduce_scatter_modes}
    do
        for allgatherv_monolithic_mode in ${allgatherv_monolithic_modes}
        do
            cmd=" CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=0"
            cmd+=" CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL=0"
            cmd+=" CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL=${reduce_scatter_mode}"
            cmd+=" CCL_ALLGATHERV_MONOLITHIC_KERNEL=${allgatherv_monolithic_mode}"
            cmd+=" mpiexec -l -n 4 ${SCRIPT_DIR}/benchmark ${bench_options} -q ${inplace_mode}"
            cmd+=" > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done


for inplace_mode in ${inplace_modes}
do
    for allgatherv_read_mode in ${allgatherv_read_modes}
    do
        for reduce_scatter_read_mode in ${reduce_scatter_read_modes}
        do
            cmd="CCL_ALLGATHERV_TOPO_READ=${allgatherv_read_mode}"
            cmd+=" CCL_REDUCE_SCATTER_TOPO_READ=${reduce_scatter_read_mode}"
            cmd+=" mpiexec -l -n 4 ${SCRIPT_DIR}/benchmark ${bench_options} -q ${inplace_mode}"
            cmd+=" > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done

for inplace_mode in ${inplace_modes}
do
    #default run means when: CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=1 CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL=1
    cmd="CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=1 CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL=1"
    cmd+=" mpiexec -l -n 4 ${SCRIPT_DIR}/benchmark  ${bench_options} -q ${inplace_mode}"
    cmd+=" > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"
    check_log ${TEST_LOG}
done

rm ${TEST_LOG}
echo "Pass"
