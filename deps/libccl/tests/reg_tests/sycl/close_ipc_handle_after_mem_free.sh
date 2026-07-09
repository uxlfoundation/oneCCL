#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

check_ccl

export CCL_ZE_CLOSE_IPC_WA=0

cache_modes="0 1"
allocator_modes="0 1"

for cache_mode in ${cache_modes}
do
    for allocator_mode in ${allocator_modes}
    do
        export CCL_ZE_CACHE=${cache_mode}
        export CCL_ZE_CACHE_OPEN_IPC_HANDLES=${cache_mode}
        export SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR=${allocator_mode}
        mpiexec -n 2 ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1
        rc=$?
        if [[ ${rc} -ne 0 ]] ; then
            echo "Fail:  allocator_mode: ${allocator_mode}"
            exit 1
        fi
        check_log ${TEST_LOG}
    done
done

rm -f ${BINFILE} ${TEST_LOG}
echo "Pass"
