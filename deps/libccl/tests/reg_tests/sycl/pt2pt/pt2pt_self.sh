#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"
PATH_TO_BINFILE="${SCRIPT_DIR}/${BINFILE}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

sizes="1 2"
transports="mpi ofi"

for transport in ${transports}
do
    for size in ${sizes}
    do
        cmd="CCL_ATL_TRANSPORT=${transport}"
        cmd+=" mpiexec -l -n ${size} ${PATH_TO_BINFILE} gpu > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm -f ${TEST_LOG}
echo "PASSED"
