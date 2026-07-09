#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

check_run() {
    log=$1
    grep 'PASSED' ${log} > /dev/null 2>&1
    rc=$?
    if [[ ${rc} -ne 0 ]] ; then
        echo "Fail"
        exit 1
    fi
}

CCL_ZE_LIBRARY_PATH=/lib/fake-path/libze_loader.so \
mpiexec -n 2 ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1
rc=$?
if [[ ${rc} -ne 0 ]] ; then
    echo "Fail"
    echo "DEBUG: Fake Path failure for mpiexec is good news"
    exit 1
fi
check_run ${TEST_LOG}
rm -f ${TEST_LOG}

CCL_ZE_ENABLE=0 mpiexec -n 2 ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1
rc=$?
if [[ ${rc} -ne 0 ]] ; then
    echo "Fail"
    exit 1
fi
check_run ${TEST_LOG}
rm -f ${BINFILE} ${TEST_LOG}

echo "Pass"
