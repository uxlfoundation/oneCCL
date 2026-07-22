#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

find_expected_algo() {
    found_algo_str=$1
    expected_strs=$2

    match_count=0
    for expected_str in ${expected_strs}
    do
        if [[ "$found_algo_str" == "$expected_str" ]]
        then
            match_count=$((match_count+1))
        fi
    done

    if [[ "$match_count" == "0" ]]
    then
        echo "Fail: couldn't find expected algo" >> ${TEST_LOG} 2>&1
    else
        echo "Found matches: ${match_count}" >> ${TEST_LOG} 2>&1
    fi
}

parse_algo() {
    expected_strs=$1

    found_algo_str=""
    while IFS='' read -r line
    do
        if [[ $line =~ "selected algo: coll allreduce, "(.*) ]]
        then
            found_sub_str=${BASH_REMATCH[1]}
            if [[ $found_sub_str =~ algo\ ([a-zA-Z0-9_]*) ]]
            then
                found_algo_str="${BASH_REMATCH[1]}"
                break
            fi
        fi
    done < ${TEST_LOG}

    find_expected_algo "${found_algo_str}" "${expected_strs}"
}

# disable SYCL kernels because topo algo for allreduce
# is executed in test to check
export CCL_ENABLE_SYCL_KERNELS=0
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

transports="mpi ofi"

expected_non_topo_algos="direct ring recursive_doubling"
bench_options="-w 1 -i 2 -c all -l allreduce -b sycl -f 2 -t 131072 $(get_default_bench_dtype)"

run_case() {
    expected_str=$1
    extra_env_str=$2
    extra_bench_ops=$3

    for transport in ${transports}
    do
        cmd="CCL_ATL_TRANSPORT=${transport}"
        cmd+=" CCL_LOG_LEVEL=debug"
        cmd+=" CCL_TOPO_COLOR=ze"
        cmd+=" ${extra_env_str}"
        cmd+=" mpiexec -l -n 2 ${SCRIPT_DIR}/benchmark"
        cmd+=" ${bench_options} ${extra_bench_ops} > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        parse_algo "${expected_str}"
        check_log ${TEST_LOG}
        rm ${TEST_LOG}
    done
}

run_case "topo"
run_case "${expected_non_topo_algos}" "CCL_TOPO_P2P_ACCESS=0"
if [[ ${PLATFORM_HW_GPU} = "ats" ]]
then
   run_case "${expected_non_topo_algos}" "" "-g 1"
   run_case "${expected_non_topo_algos}" "ZE_AFFINITY_MASK=0,2"
fi

echo "Pass"
