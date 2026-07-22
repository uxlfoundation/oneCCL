#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"
PATH_TO_BINFILE="${SCRIPT_DIR}/${BINFILE}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

sizes="2 4"
transports="mpi ofi"
algos="direct topo offload"
# currently CCL_ZE_PT2PT_READ=0 is not supported for group API

# case with 4 ranks
for algo in ${algos}
do
    for transport in ${transports}
    do
        for size in ${sizes}
        do
            cmd="CCL_ATL_TRANSPORT=${transport} CCL_SEND=${algo} CCL_RECV=${algo}"
            cmd+=" mpiexec -l -n ${size} ${PATH_TO_BINFILE} gpu > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done

rm -f ${TEST_LOG}
echo "PASSED"
