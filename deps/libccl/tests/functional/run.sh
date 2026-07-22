#!/bin/bash
set -e -o pipefail

BASENAME=`basename $0 .sh`
SCRIPT_DIR=`cd $(dirname "$BASH_SOURCE") && pwd -P`

declare -i exit_code=0
NORMAL_CTEST_CODE_FOR_FAILED_TESTS=8
TESTS_DIR="${SCRIPT_DIR}/build"

if [[ -z "$C_COMPILER" ]]; then
    echo "ERROR: C_COMPILER env variable must be defined!"
    exit 1
fi

if [[ -z "$CXX_COMPILER" ]]; then
    echo "ERROR: CXX_COMPILER env variable must be defined!"
    exit 1
fi

if [[ -z "$CCL_ATL_TRANSPORT_LIST" ]]; then
    echo "ERROR: CCL_ATL_TRANSPORT_LIST env variable must be defined!"
    exit 1
fi

# Print usage and help information
function print_help() {
    echo -e "Usage:\n" \
    "    ./${BASENAME}.sh <options> \n" \
    "<options>:\n" \
    "--hw:         set hardware type\n" \
    "--runtime:    set runtime\n" \
    "--proc-maps:  set value for n and ppns\n" \
    "--help:       print this help information"\
    "   example: ./${BASENAME}.sh --hw pvc --runtime ofi --proc_maps 2:1,2/4:1,2,4\n"
    exit 0
}

function parse_arguments() {
    if [ $# -eq 0 ]; then
        print_help
        exit 1
    fi

    while [ $# -ne 0 ]
    do
        case $1 in
            "--hw" )
                hw="${2}"
                shift
                ;;
            "--runtime" )
                runtime="${2}"
                shift
                ;;
            "--proc-maps" )
                PROC_MAPS="${2}"
                shift
                ;;
            *)
                echo "$(basename ${0}): ERROR: unknown option (${1})"
                print_help
                exit 1
                ;;
        esac
        shift
    done

    if [[ -z "$hw" || -z "$runtime" || -z "$PROC_MAPS" ]]; then
        echo "ERROR: All parameters are mandatory!"
        exit 1
    fi
}

function run_test_cmd() {
    local temp_exit_code=0
    local is_reported=""

    echo -e "\nTest command: ${1}\n"
    eval ${1} || temp_exit_code=$?
    echo "Command exit code: ${temp_exit_code}"

    if [[ $temp_exit_code -ne 0 && $temp_exit_code -ne $NORMAL_CTEST_CODE_FOR_FAILED_TESTS ]]; then
        exit_code+=1
    fi
}

function set_tests_option() {
    local options="${1}"
    local current_scope="${2}"

    for option in ${options}
    do
        local option_name=${option%=*}
        local pattern=$(echo ${current_scope} | grep -oE "${option_name}[^ ]*")
        if [[ -z ${pattern} ]]
        then
            current_scope="${current_scope} ${option}"
        else
            current_scope=${current_scope/${pattern}/${option}}
        fi
    done

    echo ${current_scope}
}

function make_tests() {
    mkdir -p ${TESTS_DIR}
    cd ${TESTS_DIR}

    if [[ -n "${COMPUTE_BACKEND}" ]]
    then
        FT_COMPUTE_BACKEND="-DCOMPUTE_BACKEND=${COMPUTE_BACKEND}"
    fi

    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="${C_COMPILER}" \
        -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
        -DPROC_MAPS="`echo ${PROC_MAPS} | tr '/' ';'`" \
        ${FT_COMPUTE_BACKEND}
    make -j all
}

function set_functional_tests_env() {
    echo "Use default env"

    func_exec_env+=" CCL_LOG_LEVEL=info"
    # flush cache inside ccl::barrier to avoid OOM
    # in case of caching and large number of different match_ids
    func_exec_env+=" CCL_CACHE_FLUSH=1"
    func_exec_env+=" CCL_MNIC=global"
    func_exec_env+=" I_MPI_DEBUG=12"
    func_exec_env+=" I_MPI_JOB_TIMEOUT=360"
    func_exec_env+=" CCL_MNIC_NAME=br,eno,eth,hfi,mlx,autoselect,^unknown"
}

function set_functional_tests_scope() {
    echo "Set default func tests scope"

    func_exec_env+=" CCL_TEST_DATA_TYPE=1"
    func_exec_env+=" CCL_TEST_SIZE_TYPE=1"
    func_exec_env+=" CCL_TEST_BUF_COUNT_TYPE=1"
    func_exec_env+=" CCL_TEST_PLACE_TYPE=1"
    func_exec_env+=" CCL_TEST_START_ORDER_TYPE=0"
    func_exec_env+=" CCL_TEST_COMPLETE_ORDER_TYPE=0"
    func_exec_env+=" CCL_TEST_COMPLETE_TYPE=0"
    func_exec_env+=" CCL_TEST_CACHE_TYPE=1"
    func_exec_env+=" CCL_TEST_SYNC_TYPE=0"
    func_exec_env+=" CCL_TEST_REDUCTION_TYPE=0"

    if [[ ${hw} == "pvc" ]]
    then
        func_exec_env+=" CCL_TEST_DYNAMIC_POINTER=0"
    else
        func_exec_env+=" CCL_TEST_DYNAMIC_POINTER=1"
    fi
}

func_exec_env=""
parse_arguments $@
make_tests
set_functional_tests_env
set_functional_tests_scope

case "$runtime" in
    ofi )
        func_exec_env+=" CCL_ATL_TRANSPORT=ofi"
        run_test_cmd "${func_exec_env} ctest --output-junit ${TESTS_DIR}/junit/default_ofi.junit.xml -V -C default"
        run_test_cmd "${func_exec_env} ctest --output-junit ${TESTS_DIR}/junit/regression_ofi.junit.xml -V -C regression"
        ;;
    mpi )
        func_exec_env+=" CCL_ATL_TRANSPORT=mpi"
        run_test_cmd "${func_exec_env} ctest --output-junit ${TESTS_DIR}/junit/default_mpi.junit.xml -V -C default"
        run_test_cmd "${func_exec_env} ctest --output-junit ${TESTS_DIR}/junit/regression_mpi.junit.xml -V -C regression"
        ;;
    ofi_adjust | mpi_adjust )

        allgatherv_algos="naive flat ring"
        allreduce_algos="rabenseifner nreduce ring double_tree recursive_doubling 2d"
        alltoall_algos="naive scatter"
        alltoallv_algos=${alltoall_algos}
        bcast_algos="ring double_tree naive"
        broadcast_algos="ring double_tree naive"
        reduce_algos="rabenseifner ring tree"
        reduce_scatter_algos="ring"

        if [ ${runtime} == "mpi_adjust" ]
        then
            allgatherv_algos="${allgatherv_algos} direct"
            allreduce_algos="${allreduce_algos} direct"
            alltoall_algos="${alltoall_algos} direct"
            alltoallv_algos=${alltoall_algos}
            bcast_algos="${bcast_algos} direct"
            broadcast_algos="${broadcast_algos} direct"
            reduce_algos="${reduce_algos} direct"
            reduce_scatter_algos="${reduce_scatter_algos} direct"

            func_exec_env+=" CCL_ATL_TRANSPORT=mpi"
        fi

        if [ ${runtime} == "ofi_adjust" ]
        then
            allgatherv_algos="${allgatherv_algos} multi_bcast"

            func_exec_env+=" CCL_ATL_TRANSPORT=ofi"
        fi

        if [[ ${hw} == "pvc" ]]
        then
            allreduce_algos="${allreduce_algos} topo"
            alltoall_algos="${alltoall_algos} topo"
            alltoallv_algos="${alltoallv_algos} topo"
            allgatherv_algos="${allgatherv_algos} topo"
            bcast_algos="${bcast_algos} topo"
            broadcast_algos="${broadcast_algos} topo"
            reduce_algos="${reduce_algos} topo"
        fi

        for proc_map in `echo ${PROC_MAPS} | tr '/' ' '`
        do
            n=`echo ${proc_map%:*}`
            ppns=`echo ${proc_map#*:} | tr ',' ' '`

            for ppn in $ppns
            do
                if [[ "$ppn" -gt "$n" ]]
                then
                    continue
                fi

                for algo in ${allgatherv_algos}
                do
                    allgatherv_exec_env=$(set_tests_option "CCL_ALLGATHERV=${algo}" "${func_exec_env}")
                    allgatherv_exec_env=$(set_tests_option "CCL_TEST_DYNAMIC_POINTER=0" "${allgatherv_exec_env}")
                    run_test_cmd "${allgatherv_exec_env} ctest --output-junit ${TESTS_DIR}/junit/allgatherv_${algo}_${n}_${ppn}.junit.xml -V -C allgatherv_${algo}_${n}_${ppn}"
                done

                for algo in ${allreduce_algos}
                do
                    allreduce_exec_env=$(set_tests_option "CCL_ALLREDUCE=${algo}" "${func_exec_env}")
                    run_test_cmd "${allreduce_exec_env} ctest --output-junit ${TESTS_DIR}/junit/allreduce_${algo}_${n}_${ppn}.junit.xml -V -C allreduce_${algo}_${n}_${ppn}"
                done

                for algo in ${alltoall_algos}
                do
                    alltoall_exec_env=$(set_tests_option "CCL_ALLTOALL=${algo}" "${func_exec_env}")
                    run_test_cmd "${alltoall_exec_env} ctest --output-junit ${TESTS_DIR}/junit/alltoall_${algo}_${n}_${ppn}.junit.xml -V -C alltoall_${algo}_${n}_${ppn}"
                done

                for algo in ${alltoallv_algos}
                do
                    alltoallv_exec_env=$(set_tests_option "CCL_ALLTOALLV=${algo}" "${func_exec_env}")
                    run_test_cmd "${alltoallv_exec_env} ctest --output-junit ${TESTS_DIR}/junit/alltoallv_${algo}_${n}_${ppn}.junit.xml -V -C alltoallv_${algo}_${n}_${ppn}"
                done

                for algo in ${bcast_algos}
                do
                    bcast_exec_env=$(set_tests_option "CCL_BCAST=${algo}" "${func_exec_env}")
                    run_test_cmd "${bcast_exec_env} ctest --output-junit ${TESTS_DIR}/junit/bcast_${algo}_${n}_${ppn}.junit.xml -V -C bcast_${algo}_${n}_${ppn}"
                done

                for algo in ${broadcast_algos}
                do
                    broadcast_exec_env=$(set_tests_option "CCL_BROADCAST=${algo}" "${func_exec_env}")
                    run_test_cmd "${broadcast_exec_env} ctest --output-junit ${TESTS_DIR}/junit/broadcast_${algo}_${n}_${ppn}.junit.xml -V -C broadcast_${algo}_${n}_${ppn}"
                done

                for algo in ${reduce_algos}
                do
                    reduce_exec_env=$(set_tests_option "CCL_REDUCE=${algo}" "${func_exec_env}")
                    run_test_cmd "${reduce_exec_env} ctest --output-junit ${TESTS_DIR}/junit/reduce_${algo}_${n}_${ppn}.junit.xml -V -C reduce_${algo}_${n}_${ppn}"
                done

                for algo in ${reduce_scatter_algos}
                do
                    reduce_scatter_exec_env=$(set_tests_option "CCL_REDUCE_SCATTER=${algo}" "${func_exec_env}")
                    run_test_cmd "${reduce_scatter_exec_env} ctest --output-junit ${TESTS_DIR}/junit/reduce_scatter_${algo}_${n}_${ppn}.junit.xml -V -C reduce_scatter_${algo}_${n}_${ppn}"
                done
            done
        done
        ;;
    priority_mode )
        # TODO: At the moment priority_mode and unordered_coll_mode launches only
        # with CCL_ATS_TRANSPORT=ofi, so these confs are missed in mpich_ofi testing.
        # We would like to add them in the future.
        func_exec_env+=" CCL_ATL_TRANSPORT=ofi"
        func_exec_env+=" CCL_PRIORITY=lifo"
        run_test_cmd "${func_exec_env} ctest --output-junit ${TESTS_DIR}/junit/default_lifo.junit.xml -V -C default"
        func_exec_env=$(set_tests_option "CCL_PRIORITY=direct" "${func_exec_env}")
        run_test_cmd "${func_exec_env} ctest --output-junit ${TESTS_DIR}/junit/default_direct.junit.xml -V -C default"
        ;;
    dynamic_pointer_mode )
        for transport in ${CCL_ATL_TRANSPORT_LIST}
        do
            func_exec_env=$(set_tests_option "CCL_ATL_TRANSPORT=${transport}" "${func_exec_env}")
            run_test_cmd "${func_exec_env} ctest --output-junit ${TESTS_DIR}/junit/default_${transport}.junit.xml -V -C default"
        done
        ;;
    fusion_mode )
        func_exec_env+=" CCL_FUSION=1"
        for transport in ${CCL_ATL_TRANSPORT_LIST}
        do
            func_exec_env=$(set_tests_option "CCL_ATL_TRANSPORT=${transport}" "${func_exec_env}")
            run_test_cmd "${func_exec_env} ctest --output-junit ${TESTS_DIR}/junit/allreduce_fusion_${transport}.junit.xml -V -C allreduce_fusion"
        done
        ;;
    * )
        echo "Please specify runtime mode: runtime=ofi|mpi|ofi_adjust|mpi_adjust|priority_mode|dynamic_pointer_mode|fusion_mode|"
        exit 1
        ;;
esac

exit $exit_code
