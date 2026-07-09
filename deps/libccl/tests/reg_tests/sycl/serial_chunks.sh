#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_ALLREDUCE=topo
export CCL_REDUCE=topo
export CCL_REDUCE_SCATTER=topo
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# Intentionally use small chunk size to expose possible races
export CCL_ZE_TMP_BUF_SIZE=$((64*1024*1024))

proc_count="4"

colls="allreduce,reduce,reduce_scatter"
bench_options="-w 2 -i 2 -j off -c all -b sycl -y 10,15,53,126,632,5216,200000000"

cmd+="mpiexec -l -n ${proc_count} -ppn 2 ${SCRIPT_DIR}/benchmark"
cmd+=" ${bench_options} -l ${colls}"
cmd+=" > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_log ${TEST_LOG}

rm ${TEST_LOG}
echo "Pass"
