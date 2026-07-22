#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename "$0" .sh)"
TEST_LOG="${BASENAME}.log"
TIMESTAMPS_LOG="${BASENAME}_timestamps.log"

source "${ROOT_DIR}/utils.sh"

check_impi
check_ccl
get_bench "${SCRIPT_DIR}" "${TEST_LOG}" "sycl"

cd "${SCRIPT_DIR}" || exit 1

export CCL_ALLREDUCE=topo
export CCL_REDUCE=topo


colls="allreduce,reduce"

bench_options="-w 0 -i 2 -j off -c all -b sycl -y 131072"

cmd=" CCL_LOG_LEVEL=debug CCL_DEBUG_TIMESTAMPS_LEVEL=1 CCL_ZE_TMP_BUF_SIZE=131072 CCL_ALLREDUCE_PIPE_CHUNK_COUNT=2 CCL_REDUCE_PIPE_CHUNK_COUNT=2"
cmd+=" mpiexec -outfile-pattern \"${TIMESTAMPS_LOG}.%r\" -l -n 4 ${SCRIPT_DIR}/benchmark"
cmd+=" ${bench_options} -l ${colls} 2>&1"
run_cmd "${cmd}"

cat "${TIMESTAMPS_LOG}".* >> "${TEST_LOG}"

check_log "${TEST_LOG}"

run_cmd "python3 ${SCRIPT_DIR}/groups.py ${TIMESTAMPS_LOG}.0 >> ${TEST_LOG} 2>&1"

rm "${TEST_LOG}"
rm "${TIMESTAMPS_LOG}".*
echo "Pass"
