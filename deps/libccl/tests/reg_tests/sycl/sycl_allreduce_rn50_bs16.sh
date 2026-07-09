#!/bin/bash

set +x

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

mode=""

case $1 in
    "test"|"") mode="test";;
    "perf") mode="perf";;
    *) echo "unknown mode: $1";;
esac

function common_env() {
    export CCL_SYCL_OUTPUT_EVENT=0
    export EnableDirectSubmission=1
    export CCL_ZE_QUEUE_INDEX_OFFSET=0
    export CCL_OP_SYNC=1
}

function run_cmd() {
    echo "Run $1" >> ${TEST_LOG}
    eval $1 >> ${TEST_LOG} 2>&1
    echo "\n\n" >> ${TEST_LOG}
}

function single_test_run() {
    algo=$1
    out_of_order_mode=$2
    group_count=$3
    cache_mode=$4
    iter_count=$5
    proc_count=$6

    cmd_args=""
    env_cmd=""

    env_cmd+=" CCL_ALLREDUCE=$algo"

    if [[ $out_of_order_mode == "0" ]]
    then
        cmd_args+=" -o 0"
    else
        cmd_args+=" -o 1"
    fi

    if [[ $cache_mode == "0" ]]
    then
        cmd_args+=" -c 0"
    else
        cmd_args+=" -c 1"
    fi

    if [[ $group_api_mode == "1" ]]
    then
        cmd_args+=" -a"
    fi

    cmd_args+=" -i $iter_count"
    cmd_args+=" -g $group_count"

    run_cmd "$env_cmd mpiexec -n $proc_count ${SCRIPT_DIR}/${BINFILE} $cmd_args"
    rc=$?

    if [[ $rc -ne 0 ]]
    then
        echo "Fail"
    fi
}

function test_run() {
    algos="topo"
    cache_modes="0 1"
    out_of_order_modes="0 1"
    group_api_modes="0 1"
    iter_counts="1"
    proc_counts="2 4"
    group_counts="24,137 12,149"

    common_env

    rm -rf ${TEST_LOG}

    echo "Running in test mode" >> ${TEST_LOG}

    for algo in $algos
    do
        for out_of_order_mode in $out_of_order_modes
        do
            for group_count in $group_counts
            do
                for cache_mode in $cache_modes
                do
                    for iter_count in $iter_counts
                    do
                        for proc_count in $proc_counts
                        do
                            for group_api_mode in $group_api_modes
                            do
                                single_test_run $algo $out_of_order_mode $group_count $cache_mode $iter_count $proc_count $group_api_mode
                                check_log ${TEST_LOG}
                            done
                        done
                    done
                done
            done
        done
    done

    rm -f ${BINFILE} ${TEST_LOG}
    echo "Pass"
}

function run_cmd_no_log() {
    echo "Running $1"
    eval $1
    printf "\n\n"
}

function perf_run() {
    echo "Running in perf mode"
    common_env

    # strip perf arg from the list and use it for the binary
    args=$(echo $@ | sed 's/perf//g')

    run_cmd_no_log "mpiexec -n 2 -ppn 2 ${SCRIPT_DIR}/${BINFILE} $args"
    rc=$?
    if [[ ${rc} -ne 0 ]]
    then
        echo "Fail"
        exit 1
    fi

    echo "Pass"
}

if [[ $mode == "test" ]]
then
    test_run
else
    perf_run $@
fi

