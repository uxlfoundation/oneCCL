#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
BINFILE="${BASENAME}"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

ppns="4"
transports="mpi ofi"
ccl_worker_count=2
sync_colls="0 1"

for ppn in ${ppns}
do
    for transport in ${transports}
    do
        for sync_coll in ${sync_colls}
        do
            nproc=$((${ppn}*2))
            cmd="CCL_ATL_TRANSPORT=${transport}"
            cmd+=" CCL_ATL_SYNC_COLL=${sync_coll}"
            cmd+=" CCL_WORKER_COUNT=${ccl_worker_count}"
            cmd+=" CCL_WORKER_AFFINITY=5-$((5-1+${nproc}*${ccl_worker_count}))"
            cmd+=" I_MPI_PIN_PROCESSOR_LIST=1,2,3,4"
            cmd+=" CCL_LOG_LEVEL=debug"
            cmd+=" mpiexec -l -n ${nproc} -ppn ${ppn}"
            cmd+=" ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done

rm ${BINFILE} ${TEST_LOG}

echo "Pass"
