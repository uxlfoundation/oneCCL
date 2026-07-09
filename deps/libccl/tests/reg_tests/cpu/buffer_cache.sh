#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

export CCL_ATL_TRANSPORT=ofi

proc_counts="2 4"
buffer_cache_modes="0 1"

bench_options="-w 4 -i 8 -l all -c all -b host -t 1048576 $(get_default_bench_dtype)"

for proc_count in ${proc_counts}
do
    for buffer_cache_mode in ${buffer_cache_modes}
    do
        cmd="CCL_BUFFER_CACHE=${buffer_cache_mode}"
        cmd+=" mpiexec -l -n ${proc_count} -ppn 1 ${SCRIPT_DIR}/benchmark"
        cmd+=" ${bench_options} > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm ${TEST_LOG}
echo "Pass"
