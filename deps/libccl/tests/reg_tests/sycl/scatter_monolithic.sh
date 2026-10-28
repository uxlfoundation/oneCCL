#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_SCATTER=topo

proc_counts="4"
# 0 = IPC-direct large-kernel path, 1 = tmp-buf staged.
large_tmp_buf_modes="0 1"

bench_options="-b sycl -w 1 -i 2 -y 1,7,8,16,17,64,133,1077,16384,65539,131072,133073"
bench_options+=" -c all -l scatter -d int8,int32,float16,float32,bfloat16"

for proc_count in ${proc_counts}
do
    for large_tmp_buf_mode in ${large_tmp_buf_modes}
    do
        cmd="CCL_SYCL_SCATTER_LARGE_TMP_BUF=${large_tmp_buf_mode}"
        cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
        cmd+=" ${bench_options}"
        cmd+=" > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm ${TEST_LOG}
echo "Pass"
