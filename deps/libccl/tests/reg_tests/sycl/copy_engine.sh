#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

copy_engines="auto main link"
h2d_copy_engines="main auto"
d2d_copy_engines="none main"

for copy_engine in ${copy_engines}
do
    for h2d_copy_engine in ${h2d_copy_engines}
    do
        for d2d_copy_engine in ${d2d_copy_engines}
        do
                cmd="CCL_ZE_COPY_ENGINE=${copy_engine}"
                cmd+=" CCL_ZE_H2D_COPY_ENGINE=${h2d_copy_engine}"
                cmd+=" CCL_ZE_D2D_COPY_ENGINE=${d2d_copy_engine}"
                cmd+=" mpiexec -l -n 4 ${SCRIPT_DIR}/benchmark -l allreduce -y 131072 > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_log ${TEST_LOG}
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
