#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

proc_counts="2 4"
reduce_scatter_pipeline_modes="0 1"
# TODO: enable 1 case when MLSL-3066
reduce_scatter_fallback_algos="0"
reduce_scatter_monolithic_modes="0 1"

for proc_count in ${proc_counts}
do
    for reduce_scatter_pipeline_mode in ${reduce_scatter_pipeline_modes}
    do
        for reduce_scatter_fallback_algo in ${reduce_scatter_fallback_algos}
        do
            for reduce_scatter_monolithic_mode in ${reduce_scatter_monolithic_modes}
            do
                cmd="CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=${reduce_scatter_pipeline_mode}"
                cmd+=" CCL_REDUCE_SCATTER_FALLBACK_ALGO=${reduce_scatter_fallback_algo}"
                cmd+=" CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL=${reduce_scatter_monolithic_mode}"
                cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/${BINFILE} gpu device"
                cmd+=" > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_log ${TEST_LOG}
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
