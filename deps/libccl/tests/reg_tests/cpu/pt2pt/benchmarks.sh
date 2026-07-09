#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "cpu" "pt2pt"
algos="direct"
transports="mpi ofi"

# latency
for algo in ${algos}
do
    for transport in ${transports}
    do
        cmd="CCL_ATL_TRANSPORT=${transport}"
        cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
	cmd+=" mpiexec -l -n 2 "
	cmd+=" ${SCRIPT_DIR}/ccl_latency -b cpu -t 2097152 -w 5 -i 5 -c all"
	cmd+=" > ${TEST_LOG} 2>&1"
	run_cmd "${cmd}"
	check_log ${TEST_LOG}
    done
done

# bandwidth
window_sizes="32 64"

for algo in ${algos}
do
    for window_size in ${window_sizes}
    do
	for transport in ${transports}
	do
	    cmd="CCL_ATL_TRANSPORT=${transport}"
	    cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
	    cmd+=" mpiexec -l -n 2 "
	    cmd+=" ${SCRIPT_DIR}/ccl_bw -b cpu -t 2097152 -w 5 -i 5 -c all"
	    cmd+=" -m ${window_size}"
	    cmd+=" > ${TEST_LOG} 2>&1"
	    run_cmd "${cmd}"
	    check_log ${TEST_LOG}
	done
    done
done

rm ${TEST_LOG}
echo "Pass"
