#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl" "pt2pt"
algos="direct topo"
transports="mpi ofi"
cache_modes="0"
queue_modes="0 1"

# latency
for algo in ${algos}
do
	for queue_mode in ${queue_modes}
	do
		for cache_mode in ${cache_modes}
		do
			for transport in ${transports}
		    do
			    cmd="CCL_ATL_TRANSPORT=${transport}"
			    cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
			    cmd+=" mpiexec -l -n 2 "
			    cmd+=" ${SCRIPT_DIR}/ccl_latency -b gpu -t 2097152 -w 2 -i 2 -c all"
			    cmd+=" -p ${cache_mode} -e ${queue_mode} > ${TEST_LOG} 2>&1"
			    run_cmd "${cmd}"
			    check_log ${TEST_LOG}
			done
		done
	done
done

# bandwidth
window_sizes="32 64"

for algo in ${algos}
do
	for window_size in ${window_sizes}
	do
		for queue_mode in ${queue_modes}
		do
			for cache_mode in ${cache_modes}
			do
				for transport in ${transports}
			    do
				    cmd="CCL_ATL_TRANSPORT=${transport}"
				    cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
				    cmd+=" mpiexec -l -n 2 "
				    cmd+=" ${SCRIPT_DIR}/ccl_bw -b gpu -t 2097152 -w 2 -i 2 -c all"
				    cmd+=" -p ${cache_mode} -e ${queue_mode} -m ${window_size}"
				    cmd+=" > ${TEST_LOG} 2>&1"
				    run_cmd "${cmd}"
				    check_log ${TEST_LOG}
				done
			done
		done
	done
done

rm ${TEST_LOG}
echo "Pass"
