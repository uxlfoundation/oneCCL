#!/bin/bash

REG_TESTS_DIR=`cd $(dirname "$BASH_SOURCE") && pwd -P`

function run_cmd() {
    cmd="${1}"
    # echo "run_cmd: ${cmd}"
    eval ${cmd}
    rc=$?
    if [ ${rc} -ne 0 ]
    then
        echo "run_cmd: ${cmd}"
        echo "Fail"
        exit 1
    fi
}

function run_negative_cmd() {
    cmd="${1}"
    # echo "run_cmd: ${cmd}"
    eval ${cmd}
    rc=$?
    if [ ${rc} -eq 0 ]
    then
        echo "run_cmd: ${cmd}"
        echo "Succeeded, but expected failure"
        exit 1
    fi
}

function check_impi() {
    if [[ -z "${I_MPI_ROOT}" ]]
    then
        which mpiexec >/dev/null 2>&1
        if [[ $? != 0 ]]
        then
            echo "Error: MPI not found."
            exit 1
        fi
     fi
}

function check_ccl() {
    if [[ -z "${CCL_ROOT}" ]]
    then
        echo "Error: \$CCL_ROOT is not set. Please source vars.sh"
        exit 1
    fi
}

function check_log() {
    log_path=$1
    extra_passed_pattern=${2:-}

    passed_pattern="iteration|passed|# all done"
    if [ "${extra_passed_pattern}" != "" ]
    then
        passed_pattern+="|${extra_passed_pattern}"
    fi
    passed_count=`grep -E -c -i "${passed_pattern}" ${log_path}`
    if [ ${passed_count} -eq 0 ]
    then
        echo "Error: did not find pass in log ${log_path}"
        exit 1
    fi

    failed_pattern="abort|^bad$|corrupt|fail|^fault$|[^-W]invalid"
    failed_pattern+="|kill|runtime_error|terminate|timed|unexpected"
    failed_pattern+="|[^-W]error|exception|connection refused"
    failed_pattern+="|job ending due to application timeout"

    # patterns to exclude as they're printed in non-error cases
    exclude_pattern="\-\-abort\-signal|CCL_ABORT_ON_THROW"
    exclude_pattern+="|fi_strerror|MPI_Error_string|fake-path"
    exclude_pattern+="|MPI startup\(\): Set ptracer for parent pid \(.*\) failed"

    failed_strings=`grep -E -i "${failed_pattern}" ${log_path} | grep -Ev "${exclude_pattern}"`
    if [ "${failed_strings}" != "" ]
    then
        echo "Error: found error in log ${log_path}"
        echo ""
        echo "${failed_strings}"
        echo ""
        exit 1
    fi
}

function check_command_exit_code() {
    if [ ${1} -ne 0 ]
    then
        echo "ERROR: ${2}"
        exit ${1}
    fi
}

check_ret() {
    rc=$1
    if [[ $rc -ne 0 ]]
    then
        echo "Fail"
        exit 1
    fi
}

function get_bench() {
    dst_dir=$1
    log_path=$2
    backend=$3
    bench_type=$4

    src_root_dir=$(realpath "${REG_TESTS_DIR}/../../")

    build_dir="build"
    cmake_str=""

    if [[ -z ${PV_ENVIRONMENT} ]]
    then
        examples_dir="${src_root_dir}/examples"
        if [ "${bench_type}" == "pt2pt" ];
        then
            benchmark_dir="${examples_dir}/pt2pt"
        else
            benchmark_dir="${examples_dir}/benchmark"
        fi
    else
        examples_dir="${REG_TESTS_DIR}/../examples"
        if [ "${bench_type}" == "pt2pt" ];
        then
            benchmark_dir="${examples_dir}/build/pt2pt"
        else
            benchmark_dir="${examples_dir}/build/benchmark"
        fi
    fi

    if [ "${bench_type}" == "pt2pt" ];
    then
        bench_latency="${benchmark_dir}/ccl_latency"
        bench_bw="${benchmark_dir}/ccl_bw"
    else
        bench="${benchmark_dir}/benchmark"
    fi

    if [ ! -f "${benchmark_dir}/benchmark" ] || \
       ([ ! -f "${bench_latency}" ] && [ ! -f "${bench_bw}" ] && [ "${bench_type}" == "pt2pt" ] );
    then
        cd "${examples_dir}"

        if [ "${backend}" == "sycl" ]
        then
            build_dir="build_sycl"
            cmake_str="-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCOMPUTE_BACKEND=dpcpp"
        fi
        mkdir -p "${build_dir}"
        cd "${build_dir}"

        cmake .. $cmake_str &>> "${log_path}"
        if [ "${bench_type}" == "pt2pt" ];
        then
            make -j ccl_latency &>> "${log_path}"
            make -j ccl_bw &>> "${log_path}"
        else
            make -j benchmark &>> "${log_path}"
        fi

        check_command_exit_code $? "Benchmark build failed"

        if [ "${bench_type}" == "pt2pt" ];
        then
            cp "${examples_dir}/${build_dir}/pt2pt/ccl_latency" "${dst_dir}"
            cp "${examples_dir}/${build_dir}/pt2pt/ccl_bw" "${dst_dir}"
        else
            cp "${examples_dir}/${build_dir}/benchmark/benchmark" "${dst_dir}"
        fi
    else
        if [ "${bench_type}" == "pt2pt" ];
        then
            cp "${benchmark_dir}/pt2pt" "${dst_dir}"
        else
            cp "${benchmark_dir}/benchmark" "${dst_dir}"
        fi
    fi
}

function get_default_bench_dtype() {
    echo "-d int32"
}

function get_default_prov() {
    echo "tcp"
}

function get_default_and_native_provs() {
    # Respect the value when FI_PROVIDER is set before this script (e.g. sanity.sh)
    # rather than overwriting it.
    if [[ -z ${FI_PROVIDER} ]]
    then
        if [[ ${PV_ENVIRONMENT} == "yes" ]] || [[ ${TRANSPORT} == "mpich" ]]
        then
            echo "$(get_default_prov)"
        elif [[ ${PLATFORM_HW_GPU} == "ats" ]] || [[ -z ${PLATFORM_HW_GPU} ]]
        then
            echo "$(get_default_prov) psm3"
        else
            echo "$(get_default_prov)"
        fi
    else
        echo "$(get_default_prov)"
    fi
}

function get_default_and_ext_native_provs() {
    if [[ -z ${PLATFORM_HW_GPU} ]]
    then
        echo "$(get_default_and_native_provs) verbs"
    else
        echo "$(get_default_and_native_provs)"
    fi
}

function check_hmem_log() {
    log_path=$1
    hmem_mode=$2
    passed_pattern="in: { .* hmem: ${hmem_mode}"
    passed_count=`grep -E -c -i "${passed_pattern}" ${log_path}`
    if [[ ${passed_count} -eq 0 ]]
    then
        echo "Error: did not find input hmem enable in log ${log_path}"
        exit 1
    fi
    passed_pattern="out: { .* hmem: ${hmem_mode}"
    passed_count=`grep -E -c -i "${passed_pattern}" ${log_path}`
    if [[ ${passed_count} -eq 0 ]]
    then
        echo "Error: did not find output hmem enable in log ${log_path}"
        exit 1
    fi
}

make_common_actions() {
    work_dir=$1
    log_path=$2
    bench_backend=${3:-}
    bench_type=${4:-}

    # Some tests might emit ANSI escape codes, but
    # we don't want to see them inside logs.
    export NO_COLOR=1

    check_impi
    check_ccl

    get_bench ${work_dir} ${log_path} ${bench_backend} ${bench_type}

    cd ${work_dir}
}

create_ze_affinity_env() {
    device_list=$1

    result_env=""

    for device in ${device_list}
    do
        result_env+="env ZE_AFFINITY_MASK=${device};"
    done

    echo "I_MPI_GTOOL=\"$result_env"\"
}

get_ofi_path() {
    if [[ ${TRANSPORT} == "mpich" ]]
    then
        # taken from ats_* and pvc_helper.sh
        echo "${MPICH_LIBFABRIC_PATH}/libfabric.so.1"
    else
        echo "${I_MPI_ROOT}/libfabric/lib/libfabric.so.1"
    fi
}

get_mpi_path() {
    if [[ ${TRANSPORT} == "mpich" ]]
    then
        # get first libmpi.so.12 from LD_LIBRARY_PATH
        for dir in $(echo ${LD_LIBRARY_PATH}} | tr ":" "\n")
        do
            if [[ -f "${dir}/libmpi.so.12" ]]
            then
                echo "${dir}/libmpi.so.12"
                break
            fi
        done
    else
        echo "${I_MPI_ROOT}/lib/release/libmpi.so.12"
    fi
}
