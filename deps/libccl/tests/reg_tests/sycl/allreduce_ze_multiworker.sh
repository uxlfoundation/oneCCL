#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

ppn="2"
worker_counts="1 2"
mnics="local global none"

for mnic in ${mnics}
do
	for worker_count in ${worker_counts}
	do
		cmd="CCL_LOG_LEVEL=info"
		cmd+=" CCL_ATL_TRANSPORT=mpi"
		cmd+=" FI_PROVIDER=tcp"
		cmd+=" CCL_WORKER_COUNT=${worker_count}"
		cmd+=" CCL_ZE_MULTI_WORKERS=1"
		cmd+=" CCL_MNIC=${mnic}"
		# each rank gets 6 cores group, the core groups are spread across the sockets
		cmd+=" I_MPI_PIN_CELL=core I_MPI_PIN_ORDER=bunch I_MPI_PIN_DOMAIN=6"
		cmd+=" mpiexec -l -n $((2*ppn)) -ppn ${ppn}"
		cmd+=" ${SCRIPT_DIR}/${BINFILE}"
		cmd+=" > ${TEST_LOG} 2>&1"
		run_cmd "${cmd}"
		check_log ${TEST_LOG}
	done
done

rm ${TEST_LOG}
echo "Pass"
