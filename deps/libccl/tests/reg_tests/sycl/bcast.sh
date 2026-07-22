#!/bin/bash

# This test is aimed to verify that multiple collective invocation (bcast as a reference)
# does not cause the hang up (originating from sync event lifecycle). Related issues:
# MLSL-2229, MLSL-2316

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

queue_modes="in_order out_of_order"
group_api_modes="0 1"

for group_api_mode in ${group_api_modes}
do
    for queue_mode in ${queue_modes}
    do
        cmd=" CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 mpiexec -l -n 2 ${SCRIPT_DIR}/${BINFILE}"
        cmd+=" --group_api ${group_api_mode}"
        cmd+=" --queue_type ${queue_mode} > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm ${TEST_LOG}
echo "Pass"
