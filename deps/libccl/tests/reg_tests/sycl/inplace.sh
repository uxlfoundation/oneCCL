#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

check_impi
check_ccl
get_bench ${SCRIPT_DIR} ${TEST_LOG} "sycl"

cd ${SCRIPT_DIR}

export CCL_ALLREDUCE=topo
export CCL_ALLTOALL=topo
export CCL_ALLTOALLV=topo

transports="ofi mpi"
proc_counts="4"

colls="allreduce alltoall alltoallv"

bench_options="-w 0 -i 2 -j off -c all -b sycl -y 131072 -q 1"

for transport in ${transports}
do
    for proc_count in ${proc_counts}
    do
        for coll in ${colls}
        do
            cmd=" CCL_ATL_TRANSPORT=${transport}"
            cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
            cmd+=" ${bench_options} -l ${coll}  > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
