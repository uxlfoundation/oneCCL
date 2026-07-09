#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

proc_counts="4"
transports="ofi mpi"
provs="$(get_default_and_native_provs)"
algos="ring topo"

for proc_count in ${proc_counts}
do
    for transport in ${transports}
    do
        for prov in ${provs}
        do
            for algo in ${algos}
            do
                cmd="CCL_ATL_TRANSPORT=${transport}"
                cmd+=" FI_PROVIDER=${prov}"
                cmd+=" CCL_ALLREDUCE=${algo}"
                cmd+=" mpiexec -l -n ${proc_count} -ppn 2"
                cmd+=" ${SCRIPT_DIR}/../recreate_sycl_comm > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_log ${TEST_LOG}
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
