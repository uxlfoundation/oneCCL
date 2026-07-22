#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

colls="allreduce reduce reduce_scatter"
bench_options="-b sycl -w 2 -i 15 -j off -c all"
bench_options+=" -y 544323,7,8,16,335544,17,64,22222222,133,133,1077,16384,65539,131072,133073,1,2"


#plain
for coll in ${colls}
do
    cmd="CCL_ZE_DEVICE_CACHE_POLICY=plain"
    cmd+=" mpiexec -l -n 4 ${SCRIPT_DIR}/benchmark ${bench_options} -l ${coll}"
    cmd+=" > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"
    check_log ${TEST_LOG}
done

#chunk
num_blocks_in_chunks="1 4 12 16 32"
evict_smallest_chunk="0 1"

for coll in ${colls}
do
    for smallest in ${evict_smallest_chunk}
    do
        for allgatherv_read_mode in ${allgatherv_read_modes}
        do
            for reduce_scatter_read_mode in ${reduce_scatter_read_modes}
            do
                for num_blocks_in_chunk in ${num_blocks_in_chunks}
                do
                    cmd=" CCL_ZE_DEVICE_CACHE_POLICY=chunk CCL_ZE_DEVICE_CACHE_UPPER_LIMIT=12"
                    cmd+=" CCL_ZE_DEVICE_CACHE_EVICT_SMALLEST=${smallest}"
                    cmd+=" CCL_ZE_DEVICE_CACHE_NUM_BLOCKS_IN_CHUNK=${num_blocks_in_chunk}"
                    cmd+=" CCL_REDUCE_SCATTER_TOPO_READ=${reduce_scatter_read_mode}"
                    cmd+=" CCL_ALLGATHERV_TOPO_READ=${allgatherv_read_mode}"
                    cmd+=" mpiexec -l -n 4 ${SCRIPT_DIR}/benchmark ${bench_options} -l ${coll}"
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
