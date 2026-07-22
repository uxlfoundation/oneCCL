#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

check_single_rank_case() {
    local log_file="$1"
    local missing_strings=()

    if [ ! -f "$log_file" ]; then
        echo "log file '$log_file' not found."
        return 1
    fi

    check_string() {
        local string="$1"
        if ! grep -q "$string" "$log_file"; then
            missing_strings+=("$string")
        fi
    }

    check_string "single rank: out-of-place case, coll: allgather"
    check_string "single rank: out-of-place case, coll: allgatherv"
    check_string "single rank: out-of-place case, coll: allreduce"
    check_string "single rank: out-of-place case, coll: alltoall"
    check_string "single rank: out-of-place case, coll: alltoallv"
    check_string "single rank: inplace case, coll: bcast"
    check_string "single rank: out-of-place case, coll: reduce"
    check_string "single rank: out-of-place case, coll: reduce_scatter"

    if [ ${#missing_strings[@]} -eq 0 ]; then
        return 0
    else
        echo "Error: The following required strings were not found in the log file:" >> "${log_file}"
        for missing_string in "${missing_strings[@]}"; do
            echo "  $missing_string" >> "${log_file}"
        done
        echo "Error: not all required strings found in the log file" >> "${log_file}" 2>&1
        exit 1
    fi
}

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"
bench_options="-b sycl -w 1 -i 2 -j off -t 1024 -l all -c all"

cmd="CCL_LOG_LEVEL=debug mpiexec -l -n 1 ${SCRIPT_DIR}/benchmark ${bench_options}"
cmd+=" > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_single_rank_case ${TEST_LOG}
check_log ${TEST_LOG}

rm ${TEST_LOG}
echo "Pass"
