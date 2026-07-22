#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

# Scale-up algos
export CCL_ALLREDUCE=topo
export CCL_ALLGATHER=topo
export CCL_ALLGATHERV=topo
export CCL_ALLTOALLV=topo
export CCL_ALLTOALL=topo
export CCL_REDUCE=topo
export CCL_REDUCE_SCATTER=topo
# Scale-out algos
export CCL_ALLREDUCE_SCALEOUT="recursive_doubling:0-8192;rabenseifner:8193-65536;ring:65537-max"
export CCL_ALLGATHER_SCALEOUT=ring
export CCL_ALLGATHERV_SCALEOUT=ring
export CCL_ALLTOALL_SCALEOUT=scatter
export CCL_ALLTOALLV_SCALEOUT=scatter
export CCL_REDUCE_SCALEOUT=tree
export CCL_REDUCE_SCATTER_SCALEOUT=naive

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

transports="mpi"
ppns="2"

multi_workers_modes="0 1"
cache_modes="0 1"

colls="allreduce reduce alltoallv alltoall allgather allgatherv reduce_scatter"

bench_options="-w 1 -i 2 -c all -j off -y 512,32768,2097152 -b sycl $(get_default_bench_dtype)"

clear_env() {
    unset CCL_ALLREDUCE_SCALEOUT
    unset CCL_ALLGATHER_SCALEOUT
    unset CCL_ALLGATHERV_SCALEOUT
    unset CCL_REDUCE_SCALEOUT
    unset CCL_ALLTOALL_SCALEOUT
    unset CCL_ALLTOALLV_SCALEOUT
    unset CCL_REDUCE_SCATTER_SCALEOUT
    unset CCL_ZE_MULTI_WORKERS
    unset CCL_WORKER_COUNT
}

function check_algo_count_not_zero() {
    algo_count=$1

    if [[ "${algo_count}" == "0" ]]
    then
        echo "Failed algo count: ${algo_count}"
        exit 1
    fi
}

function check_env_parsing() {
    log_path=$1
    env_to_parse=$2
    selection_str=$3

    parsed_str="${env_to_parse}: ${selection_str}"
    log_str=`grep -E -i "${parsed_str}" ${log_path}`
    if [[ "${log_str}" == "" ]]
    then
        echo "Failed ${env_to_parse} parsing!"
        exit 1
    fi
}

check_scaleout_allgather_selection () {
    log_path=$1

    check_env_parsing ${log_path} "CCL_ALLGATHER" ${CCL_ALLGATHER}
    check_env_parsing ${log_path} "CCL_ALLGATHER_SCALEOUT" ${CCL_ALLGATHER_SCALEOUT}

    # checking algorithms selection
    topo_count=`grep -E -i "get: selected algo: coll allgather, count [[:digit:]]+, algo topo" ${log_path} | wc -l`
    ring_count=`grep -E -i "get: selected scale-out algo: coll allgather, count [[:digit:]]+, algo ring" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${topo_count}
    # check_algo_count_not_zero ${ring_count}

    # checking algorithms building
    ring_build_count=`grep -E -i "ccl_coll_build_ring_allgather: build ring allgather" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${ring_build_count}
}

check_scaleout_allgatherv_selection () {
    log_path=$1

    check_env_parsing ${log_path} "CCL_ALLGATHERV" ${CCL_ALLGATHERV}
    check_env_parsing ${log_path} "CCL_ALLGATHERV_SCALEOUT" ${CCL_ALLGATHERV_SCALEOUT}

    # checking algorithms selection
    topo_count=`grep -E -i "get: selected algo: coll allgatherv, count [[:digit:]]+, algo topo" ${log_path} | wc -l`
    ring_count=`grep -E -i "get: selected scale-out algo: coll allgatherv, count [[:digit:]]+, algo ring" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${topo_count}
    # check_algo_count_not_zero ${ring_count}

    # checking algorithms building
    ring_build_count=`grep -E -i "ccl_coll_build_ring_allgatherv: build ring allgatherv" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${ring_build_count}
}

check_scaleout_alltoall_selection () {
    log_path=$1

    check_env_parsing ${log_path} "CCL_ALLTOALL" ${CCL_ALLTOALL}
    check_env_parsing ${log_path} "CCL_ALLTOALL_SCALEOUT" ${CCL_ALLTOALL_SCALEOUT}

    # checking algorithms selection
    topo_count=`grep -E -i "get: selected algo: coll alltoall, count [[:digit:]]+, algo topo" ${log_path} | wc -l`
    scatter_count=`grep -E -i "get: selected scale-out algo: coll alltoallv, count [[:digit:]]+, algo scatter" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${topo_count}
    # check_algo_count_not_zero ${scatter_count}

    # checking algorithms building
    scatter_build_count=`grep -E -i "ccl_coll_build_scatter_alltoallv: build scatter alltoallv" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${scatter_build_count}
}

check_scaleout_alltoallv_selection () {
    log_path=$1

    check_env_parsing ${log_path} "CCL_ALLTOALLV" ${CCL_ALLTOALLV}
    check_env_parsing ${log_path} "CCL_ALLTOALLV_SCALEOUT" ${CCL_ALLTOALLV_SCALEOUT}

    # checking algorithms selection
    topo_count=`grep -E -i "get: selected algo: coll alltoallv, count [[:digit:]]+, algo topo" ${log_path} | wc -l`
    scatter_count=`grep -E -i "get: selected scale-out algo: coll alltoallv, count [[:digit:]]+, algo scatter" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${topo_count}
    # check_algo_count_not_zero ${scatter_count}

    # checking algorithms building
    scatter_build_count=`grep -E -i "ccl_coll_build_scatter_alltoallv: build scatter alltoallv" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${scatter_build_count}
}

check_scaleout_reduce_selection () {
    log_path=$1

    check_env_parsing ${log_path} "CCL_REDUCE" ${CCL_REDUCE}
    check_env_parsing ${log_path} "CCL_REDUCE_SCALEOUT" ${CCL_REDUCE_SCALEOUT}

    # checking algorithms selection
    topo_count=`grep -E -i "get: selected algo: coll reduce, count [[:digit:]]+, algo topo" ${log_path} | wc -l`
    tree_count=`grep -E -i "get: selected scale-out algo: coll reduce, count [[:digit:]]+, algo tree" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${topo_count}
    # check_algo_count_not_zero ${tree_count}

    # checking algorithms building
    tree_build_count=`grep -E -i "ccl_coll_build_binomial_reduce: build binomial reduce" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${tree_build_count}
}

check_scaleout_allreduce_selection() {
    log_path=$1

    check_env_parsing ${log_path} "CCL_ALLREDUCE" ${CCL_ALLREDUCE}
    check_env_parsing ${log_path} "CCL_ALLREDUCE_SCALEOUT" ${CCL_ALLREDUCE_SCALEOUT}

    # checking algorithms selection
    topo_count=`grep -E -i "get: selected algo: coll allreduce, count [[:digit:]]+, algo topo" ${log_path} | wc -l`
    recursive_doubling_count=`grep -E -i "get: selected scale-out algo: coll allreduce, count [[:digit:]]+, algo recursive_doubling" ${log_path} | wc -l`
    rabenseifner_count=`grep -E -i "get: selected scale-out algo: coll allreduce, count [[:digit:]]+, algo rabenseifner" ${log_path} | wc -l`
    ring_count=`grep -E -i "get: selected scale-out algo: coll allreduce, count [[:digit:]]+, algo ring" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${topo_count}
    # check_algo_count_not_zero ${recursive_doubling_count}
    # check_algo_count_not_zero ${rabenseifner_count}
    # check_algo_count_not_zero ${ring_count}

    # checking algorithms building
    ring_build_count=`grep -E -i "ccl_coll_build_ring_allreduce: build ring allreduce" ${log_path} | wc -l`
    rabenseifner_build_count=`grep -E -i "ccl_coll_build_rabenseifner_allreduce: build Rabenseifner's allreduce" ${log_path} | wc -l`
    recursive_doubling_build_count=`grep -E -i "ccl_coll_build_recursive_doubling_allreduce: build recursive_doubling allreduce" ${log_path} | wc -l`

    # TODO: MLSL-1684 uncomment when completed
    # check_algo_count_not_zero ${ring_build_count}
    # check_algo_count_not_zero ${rabenseifner_build_count}
    # check_algo_count_not_zero ${recursive_doubling_build_count}
}

check_scaleout_reduce_scatter_selection () {
    log_path=$1

    check_env_parsing ${log_path} "CCL_REDUCE_SCATTER" ${CCL_REDUCE_SCATTER}
    check_env_parsing ${log_path} "CCL_REDUCE_SCATTER_SCALEOUT" ${CCL_REDUCE_SCATTER_SCALEOUT}
}

check_scaleout_selection() {
    log_path=$1
    coll=$2

    if [[ "${coll}" == "allreduce" ]]
    then
        check_scaleout_allreduce_selection ${log_path}
    elif [[ "${coll}" == "allgather" ]]
    then
        check_scaleout_allgather_selection ${log_path}
    elif [[ "${coll}" == "allgatherv" ]]
    then
        check_scaleout_allgatherv_selection ${log_path}
    elif [[ "${coll}" == "alltoall" ]]
    then
        check_scaleout_alltoall_selection ${log_path}
    elif [[ "${coll}" == "alltoallv" ]]
    then
        check_scaleout_alltoallv_selection ${log_path}
    elif [[ "${coll}" == "reduce" ]]
    then
        check_scaleout_reduce_selection ${log_path}
    elif [[ "${coll}" == "reduce_scatter" ]]
    then
        check_scaleout_reduce_scatter_selection ${log_path}
    else
        echo "Unknow collective!"
    fi
}

for coll in ${colls}
do
    for transport in ${transports}
    do
        for ppn in ${ppns}
        do
            for multi_worker in ${multi_workers_modes}
            do
                # alltoall(v) topo is supported with bidir only
                if [[ "${coll}" == "alltoallv" || "${coll}" == "alltoall" ]]
                then
                    export CCL_ZE_BIDIR_ALGO=1
                fi
                for cache_mode in ${cache_modes}
                do
                    cmd="CCL_ATL_TRANSPORT=${transport}"
                    cmd+=" CCL_ZE_MULTI_WORKERS=${multi_worker}"
                    cmd+=" CCL_LOG_LEVEL=debug"
                    cmd+=" mpiexec -l -n $((${ppn}*2)) -ppn ${ppn} ${SCRIPT_DIR}/benchmark"
                    cmd+=" ${bench_options} -p ${cache_mode} -l ${coll} > ${TEST_LOG} 2>&1"
                    run_cmd "${cmd}"
                    check_scaleout_selection ${TEST_LOG} ${coll}
                    check_log ${TEST_LOG}
                    rm ${TEST_LOG}
                done
            done
        done
    done
done
clear_env

# Check different combination cases with topo algorithms.
# No exceptions == passed

# Supported collectives
scaleout_coll_list="allreduce reduce alltoallv alltoall allgatherv reduce_scatter"

# Supported algorithms
# NOTE: direct with multiple workers (alltoall for all direct) and 2d for allreduce is disabled in the code
allreduce_scaleout_algo_list="direct rabenseifner nreduce ring ring_rma double_tree recursive_doubling 2d"
allgather_scaleout_algo_list="direct naive ring flat multi_bcast"
allgatherv_scaleout_algo_list="direct naive ring flat multi_bcast"
reduce_scaleout_algo_list="direct rabenseifner tree double_tree"
alltoallv_scaleout_algo_list="direct naive scatter"
reduce_scatter_scaleout_algo_list="direct ring naive"

# For this case size does not matter, so use just one option to reduce the test time
bench_options="-w 0 -i 1 -c all -j off -y 32768 -b sycl $(get_default_bench_dtype)"

# Should run on 2 nodes with 2 PPN
function run_scaleout_collective() {
    coll=$1
    coll_uppercase=${coll^^}
    shift
    algo_list=("$@")

    for algo in ${algo_list}
    do
        # define scale-out selection env variable
        export CCL_${coll_uppercase}_SCALEOUT=${algo}

        for multi_workers_mode in ${multi_workers_modes}
        do
            if [[ "${multi_workers_mode}" == "1" ]]; then
                # Special case, scheduler is partitioned between 2 worker threads
                export CCL_ZE_MULTI_WORKERS=1
                export CCL_WORKER_COUNT=2
            else
                # Default case - 1 worker thread handles all the schedules
                export CCL_ZE_MULTI_WORKERS=0
                export CCL_WORKER_COUNT=1
            fi
            cmd="CCL_ATL_TRANSPORT=mpi"
            cmd+=" mpiexec -l -n 4 -ppn 2 ${SCRIPT_DIR}/benchmark"
            cmd+=" ${bench_options} -p 0 -l ${coll} > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
            rm ${TEST_LOG}
        done
    done
    clear_env
}

for coll in ${scaleout_coll_list}
do
    # alltoall(v) topo is supported with bidir only
    if [[ "${coll}" == "alltoallv" || "${coll}" == "alltoall" ]]
    then
        export CCL_ZE_BIDIR_ALGO=1
    fi
    scaleout_algo_list=$(eval "echo \$${coll}_scaleout_algo_list")
    run_scaleout_collective ${coll} "${scaleout_algo_list[@]}"
done

echo "Pass"
