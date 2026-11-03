#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

# 4 ranks over 2 per node: the test needs >= 3 ranks, and CCL requires the same
# number of ranks on every node (create_topo_subcomms asserts
# node_comm->size() * r2r_comm->size() == comm size). 4x2 keeps that true both
# on a single node, where mpiexec puts all 4 ranks, and on a 2-node allocation,
# where it puts 2 on each.
mpiexec -l -n 4 -ppn 2 ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1
rc=$?
if [[ ${rc} -ne 0 ]]
then
    echo "Fail"
    echo "DEBUG: alltoallv_empty_count.sh fails for mpiexec"
    exit 1
fi

check_log ${TEST_LOG}

rm -f ${BINFILE} ${TEST_LOG}
echo "Pass"
