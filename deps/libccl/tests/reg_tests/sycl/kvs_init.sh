#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

kvs_modes="pmi mpi"
backends="sycl host"

bench_options="-w 0 -i 1 -d int32 -l allreduce -y 1024"

for kvs_mode in ${kvs_modes}
do
    for backend in ${backends}
    do
        cmd="CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=${kvs_mode}"
        cmd+=" mpiexec -l -n 2 ${SCRIPT_DIR}/benchmark"
        cmd+=" ${bench_options} -b ${backend} > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm ${TEST_LOG}
echo "Pass"
