#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

function check_algo_count_not_zero() {
    algo_count=$1

    if [[ "${algo_count}" == "0" ]]
    then
        echo "Failed algo count: ${algo_count}"
        exit 1
    fi
}

check_omp_log () {
    log_path=$1
    coll_name=$2

    count=`grep -i "invoke omp ${coll_name}" ${log_path} | wc -l`
    check_algo_count_not_zero ${count}
}

# check if omp collectives are invoked
omp_threads="2"
transports="mpi"
proc_counts="2"
is_syncs="1"

bench_options="-w 0 -i 1 -l allreduce -c all -b host --elem_counts 1024 -d bfloat16,float16"
for omp_thread in ${omp_threads}
do
    for transport in ${transports}
    do
        for proc_count in ${proc_counts}
        do
            for is_sync in ${is_syncs}
            do
                cmd="OMP_NUM_THREADS=${omp_thread}"
                cmd+=" CCL_ATL_TRANSPORT=${transport}"
                cmd+=" CCL_ALLREDUCE=direct"
                cmd+=" CCL_ATL_SYNC_COLL=${is_sync}"
                cmd+=" CCL_LOG_LEVEL=debug"
                cmd+=" mpiexec -l -n ${proc_count} -ppn ${proc_count}"
                cmd+=" ${SCRIPT_DIR}/benchmark ${bench_options} > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_omp_log ${TEST_LOG} "allreduce"
            done
        done
    done
done

bench_options="-w 0 -i 1 -l allgatherv -c all -b host --elem_counts 1024 $(get_default_bench_dtype)"
for omp_thread in ${omp_threads}
do
    for transport in ${transports}
    do
        for proc_count in ${proc_counts}
        do
            for is_sync in ${is_syncs}
            do
                cmd="OMP_NUM_THREADS=${omp_thread}"
                cmd+=" CCL_ATL_TRANSPORT=${transport}"
                cmd+=" CCL_ALLGATHERV=direct"
                cmd+=" CCL_ATL_SYNC_COLL=${is_sync}"
                cmd+=" CCL_LOG_LEVEL=debug"
                cmd+=" mpiexec -l -n ${proc_count} -ppn 2"
                cmd+=" ${SCRIPT_DIR}/benchmark ${bench_options} > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_omp_log ${TEST_LOG} "allgatherv"
            done
        done
    done
done

# check if omp collectives run successfully
omp_threads="0 2"
transports="mpi"
proc_counts="2"
is_syncs="0 1"

bench_options="-w 0 -i 4 -l allreduce -c all -b host -t 1048576 -d bfloat16,float16"
for omp_thread in ${omp_threads}
do
    for transport in ${transports}
    do
        for proc_count in ${proc_counts}
        do
            for is_sync in ${is_syncs}
            do
                cmd="OMP_NUM_THREADS=${omp_thread}"
                cmd+=" CCL_ATL_TRANSPORT=${transport}"
                cmd+=" CCL_ALLREDUCE=direct"
                cmd+=" CCL_ATL_SYNC_COLL=${is_sync}"
                cmd+=" mpiexec -l -n ${proc_count} -ppn ${proc_count}"
                cmd+=" ${SCRIPT_DIR}/benchmark ${bench_options} > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_log ${TEST_LOG}
            done
        done
    done
done

bench_options="-w 0 -i 4 -l allgatherv -c all -b host -t 1048576 $(get_default_bench_dtype)"
for omp_thread in ${omp_threads}
do
    for transport in ${transports}
    do
        for proc_count in ${proc_counts}
        do
            for is_sync in ${is_syncs}
            do
                cmd="OMP_NUM_THREADS=${omp_thread}"
                cmd+=" CCL_ATL_TRANSPORT=${transport}"
                cmd+=" CCL_ALLGATHERV=direct"
                cmd+=" CCL_ATL_SYNC_COLL=${is_sync}"
                cmd+=" mpiexec -l -n ${proc_count} -ppn 2"
                cmd+=" ${SCRIPT_DIR}/benchmark ${bench_options} > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_log ${TEST_LOG}
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
