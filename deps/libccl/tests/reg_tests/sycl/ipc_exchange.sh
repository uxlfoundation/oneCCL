#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export I_MPI_JOB_TIMEOUT=360
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=200
export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=200

ipc_exchange_modes="sockets drmfd pidfd"
transports="ofi mpi"
sched_cache_modes="0 1"
ze_cache_modes="0 1"

bench_options="-w 16 -i 1024 -j off -c all -b sycl -y 32,64,1024 -l all $(get_default_bench_dtype)"

for ze_cache_mode in ${ze_cache_modes}
do
    for sched_cache_mode in ${sched_cache_modes}
    do
        for ipc_exchange_mode in ${ipc_exchange_modes}
        do
            for transport in ${transports}
            do
                cmd=" CCL_ATL_TRANSPORT=${transport}"
                cmd+=" CCL_ZE_IPC_EXCHANGE=${ipc_exchange_mode} CCL_ZE_CACHE=${ze_cache_mode}"
                cmd+=" mpiexec -l -n 4 ${SCRIPT_DIR}/benchmark"
                cmd+=" ${bench_options} -p ${sched_cache_mode} > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_log ${TEST_LOG}
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
