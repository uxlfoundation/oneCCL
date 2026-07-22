#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

proc_counts="4 8"
transports="ofi mpi"
ofi_provs="$(get_default_and_native_provs)"

for proc_count in ${proc_counts}
do
    for transport in ${transports}
    do
        for ofi_prov in ${ofi_provs}
        do
            cmd="CCL_WORKER_COUNT=2"
            cmd+=" CCL_ATL_TRANSPORT=${transport}"
            cmd+=" FI_PROVIDER=${ofi_prov}"
            cmd+=" mpiexec -l -n ${proc_count} -ppn 2"
            cmd+=" ${SCRIPT_DIR}/${BASENAME} > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done

rm ${BINFILE} ${TEST_LOG}
echo "Pass"
