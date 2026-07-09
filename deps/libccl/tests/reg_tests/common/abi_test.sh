#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="`basename $0 .sh`"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1
rc=$?

if [[ $rc -ne 0 ]]; then
    echo "Fail"
    exit 1
fi

check_log ${TEST_LOG}

rm -rf ${BINFILE} ${TEST_LOG}
echo "Pass"

