#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_ATL_SYNC_COLL=1
export CCL_ATL_TRANSPORT=mpi

export CCL_ALLREDUCE=topo
export CCL_BARRIER=direct

export I_MPI_FABRICS=shm

bench_options="-b sycl -w 4 -i 4 -c all -l allreduce -y 1000 $(get_default_bench_dtype)"
lock_levels="global vci"

for lock_level in ${lock_levels}
do
    export I_MPI_THREAD_LOCK_LEVEL=${lock_level}
    mpiexec -l -n 2 ${SCRIPT_DIR}/benchmark ${bench_options} > ${TEST_LOG} 2>&1
    ret_val=$?
    if [ $ret_val -ne 0 ]
    then
        echo "Fail: ${lock_level}"
        exit -1
    fi
    check_log ${TEST_LOG}
done

rm ${TEST_LOG}
echo "Pass"
