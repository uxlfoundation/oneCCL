#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

transports="mpi"
proc_counts="4"

colls="allgatherv,allreduce,reduce,reduce_scatter"

chunk_counts="1 2 3 4"
cache_modes="0 1"
worker_counts="1 2"

bench_options="-w 2 -i 2 -j off -b sycl -y 0,1,2,3,7,8,16,17,64,133,1077,16384,65539,131072,133073"
bench_options+=" -c all -d int8,int32,float16,float32,bfloat16 -r all"

function run_pipe_test() {
    chunks_allgatherv="${1}"
    chunks_allreduce="${2}"
    chunks_reduce_scatter="${3}"
    chunks_reduce="${4}"

    cmd="CCL_ALLGATHERV_PIPE_CHUNK_COUNT=${chunks_allgatherv}"
    cmd+=" CCL_ALLREDUCE_PIPE_CHUNK_COUNT=${chunks_allreduce}"
    cmd+=" CCL_REDUCE_SCATTER_PIPE_CHUNK_COUNT=${chunks_reduce_scatter}"
    cmd+=" CCL_REDUCE_PIPE_CHUNK_COUNT=${chunks_reduce}"
    cmd+=" CCL_ATL_TRANSPORT=${transport}"
    cmd+=" CCL_WORKER_COUNT=${worker_count}"
    cmd+=" CCL_YIELD=sched_yield"
    cmd+=" CCL_ENABLE_SYCL_KERNELS=0"
    cmd+=" CCL_ALLGATHERV=topo"
    cmd+=" CCL_ALLREDUCE=topo"
    cmd+=" CCL_REDUCE=topo"
    cmd+=" CCL_REDUCE_SCATTER=topo"
    cmd+=" mpiexec -l -n ${proc_count} -ppn 2 ${SCRIPT_DIR}/benchmark"
    cmd+=" ${bench_options} -l ${colls} -p ${cache_mode}"
    cmd+=" > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"

    check_log ${TEST_LOG}
}


for chunk_count in ${chunk_counts}
do
    for transport in ${transports}
    do
        for proc_count in ${proc_counts}
        do
            for cache_mode in ${cache_modes}
            do
            	for worker_count in ${worker_counts}
	            do
                    run_pipe_test ${chunk_count} ${chunk_count} ${chunk_count} ${chunk_count}
                done
            done
        done
    done
done


transport="mpi"
proc_count="4"
cache_mode="0"
worker_count="1"

run_pipe_test 0 0 0 0
run_pipe_test 0 0 0 4
run_pipe_test 0 0 4 0
run_pipe_test 0 0 4 4

run_pipe_test 0 4 0 0
run_pipe_test 0 4 0 4
run_pipe_test 0 4 4 0
run_pipe_test 0 4 4 4

run_pipe_test 4 0 0 0
run_pipe_test 4 0 0 4
run_pipe_test 4 0 4 0
run_pipe_test 4 0 4 4

run_pipe_test 4 4 0 0
run_pipe_test 4 4 0 4
run_pipe_test 4 4 4 0
run_pipe_test 4 4 4 4


rm ${TEST_LOG}
echo "Pass"
