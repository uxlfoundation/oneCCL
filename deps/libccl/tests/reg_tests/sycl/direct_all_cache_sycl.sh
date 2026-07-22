#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

transports="mpi"
proc_counts="4"
cache_modes="0 1"
backends="host sycl"
bench_options="-w 1 -i 2 -l all -c all -y 17,1024,65536,1048576"

for transport in ${transports}
do
    for proc_count in ${proc_counts}
    do
        for cache_mode in ${cache_modes}
        do
            for backend in ${backends}
            do
                cmd="CCL_ATL_TRANSPORT=${transport}"
                cmd+=" CCL_ATL_SYNC_COLL=1"
                cmd+=" CCL_ALLGATHER=direct"
                cmd+=" CCL_ALLGATHERV=direct"
                cmd+=" CCL_ALLREDUCE=direct"
                cmd+=" CCL_ALLTOALL=direct"
                cmd+=" CCL_BARRIER=direct"
                cmd+=" CCL_BCAST=direct"
                cmd+=" CCL_BCASTEXT=direct"
                cmd+=" CCL_REDUCE=direct"
                cmd+=" CCL_REDUCE_SCATTER=direct"
                cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
                cmd+=" ${bench_options} -p ${cache_mode} -b ${backend} > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_log ${TEST_LOG}
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
