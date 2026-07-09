#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "cpu"

transports=("ofi" "mpi")

run_allreduce_test() {
    local transport=$1
    local scenario=$2

    cmd="CCL_ATL_TRANSPORT=${transport}"
    cmd+=" mpiexec -l -n 4 ${SCRIPT_DIR}/allreduce_comm_split ${scenario}"
    cmd+=" > ${TEST_LOG} 2>&1"

    run_cmd "${cmd}"
    check_log ${TEST_LOG}
}

for transport in "${transports[@]}"; do
    for scenario in {0..4}; do
        run_allreduce_test ${transport} ${scenario}
    done
done

rm -f ${TEST_LOG}
echo "All tests passed!"
