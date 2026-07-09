#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

function check_stub_backend_log() {
    COMM_ID="$1"
    TEST_LOG="$2"
    NUM_RANKS="$3"

    if [[ $(grep -o "running stub communicator id: ${COMM_ID}" ${TEST_LOG} | wc -l) != "${NUM_RANKS}" ]]; then
        echo "failed to find a correct comm id, expected: ${COMM_ID}"
        exit 1
    fi
}

function run_test() {
    COMM_ID=$1

    export CCL_BACKEND=stub

    if [[ "$COMM_ID" != "0" ]]; then
        export CCL_STUB_BACKEND_COMM_ID=${COMM_ID}
    else
        unset CCL_STUB_BACKEND_COMM_ID
    fi

    NUM_RANKS=2

    mpiexec -l -n ${NUM_RANKS} -ppn 1 ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1
    rc=$?
    if [[ ${rc} -ne 0 ]]
    then
        echo "Fail"
        exit 1
    fi

    check_log ${TEST_LOG}
    check_stub_backend_log ${COMM_ID} ${TEST_LOG} ${NUM_RANKS}
    rm -rf ${TEST_LOG}
}

# just some arbitrary number
run_test 123
run_test 0

rm -f ${BINFILE}
echo "Pass"
