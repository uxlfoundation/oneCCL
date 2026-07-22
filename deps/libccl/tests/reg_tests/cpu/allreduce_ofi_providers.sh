#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_SHM=1

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "cpu"

provs="shm" #TODO: add more provs (see TODO-1 below)

#TODO-1: also test using different ofi provider
#        e.g., FI_PROVIDER=cxi, while CCL_ATL_SHM=1
#TODO-2: add a multi-node test with TODO-1 config

# single node test
for prov in ${provs}
do
    cmd="FI_PROVIDER=${prov}"
    cmd+=" mpiexec -l -n 2 ${SCRIPT_DIR}/benchmark -l allreduce"
    cmd+=" > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"
    check_log ${TEST_LOG}
done

rm ${TEST_LOG}
echo "Pass"
