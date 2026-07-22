#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_REDUCE_SCATTER=topo

proc_counts="4"
 #TODO: fix CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=0
reduce_scatter_pipeline_modes="1"
reduce_scatter_fallback_algos="0 1"
reduce_scatter_monolithic_modes="0 1"
reduce_scatter_read_modes="0 1"

bench_options="-b sycl -w 1 -i 2 -y 1,7,8,16,17,64,133,1077,16384,65539,131072,133073"
bench_options+=" -c all -l reduce_scatter -d int8,int32,float16,float32,bfloat16"

for proc_count in ${proc_counts}
do
    for reduce_scatter_pipeline_mode in ${reduce_scatter_pipeline_modes}
    do
        for reduce_scatter_fallback_algo in ${reduce_scatter_fallback_algos}
        do
            for reduce_scatter_monolithic_mode in ${reduce_scatter_monolithic_modes}
            do
                for reduce_scatter_read_mode in ${reduce_scatter_read_modes}
                do
                    cmd="CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=${reduce_scatter_pipeline_mode}"
                    cmd+=" CCL_REDUCE_SCATTER_FALLBACK_ALGO=${reduce_scatter_fallback_algo}"
                    cmd+=" CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL=${reduce_scatter_monolithic_mode}"
                    cmd+=" CCL_REDUCE_SCATTER_TOPO_READ=${reduce_scatter_read_mode}"
                    cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
                    cmd+=" ${bench_options}"
                    cmd+=" > ${TEST_LOG} 2>&1"
                    run_cmd "${cmd}"
                    check_log ${TEST_LOG}
                done
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
