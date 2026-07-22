#!/bin/bash

touch main_output.txt
exec | tee ./main_output.txt


BASENAME=`basename $0 .sh`
SCRIPT_DIR=`cd $(dirname "$BASH_SOURCE") && pwd -P`
JSON_REPORT="test_cases.json"
JUNIT_REPORT="summary.junit.xml"

echo "<testsuites><testsuite name=\"example_tests\" tests=\"0\" failures=\"0\" skipped=\"0\">" > "${SCRIPT_DIR}/${JUNIT_REPORT}"

unsupported_platforms="dg2 ats-m ats-m1 ats-m3 gen12"

# we need to use only appropriate ifaces, with this env variable we can exclude docker iface
export CCL_MNIC_NAME="eno,enp,ens7f0np0,eth,hfi,mlx,autoselect,^unknown"

create_logs_dir()
{
    SCRIPT_DIR=`cd $(dirname "$BASH_SOURCE") && pwd -P`
    export LOGS_DIR=${SCRIPT_DIR}/logs/$(date +%Y-%m-%d_%H%M%S)
    mkdir -p $LOGS_DIR
    echo "LOGS_DIR = " ${LOGS_DIR}
}

set_example_work_dir()
{
    SCRIPT_DIR=`cd $(dirname "$BASH_SOURCE") && pwd -P`
    export EXAMPLE_WORK_DIR=${SCRIPT_DIR}/build
}

create_work_dir()
{
    if [[ -z "${EXAMPLE_WORK_DIR}" ]]
    then
        set_example_work_dir
        rm -r ${EXAMPLE_WORK_DIR}
        mkdir -p ${EXAMPLE_WORK_DIR}
    fi
    echo "EXAMPLE_WORK_DIR =" ${EXAMPLE_WORK_DIR}
}

CheckCommandExitCode()
{
    if [ $1 -ne 0 ]; then
        echo "ERROR: ${2}" 1>&2
        exit $1
    fi
}

declare -i total_fails=0
declare -i total_skipped=0
declare -i log_idx=0
declare -i total_tests=0

check_test()
{
    local test_log=$1
    local test_file=$2
    local test="$(basename ${test_log%.*})"

    passed_pattern="passed|# all done"

    failed_pattern="abort|^bad$|corrupt|fail|^fault$|[^-W]invalid"
    failed_pattern+="|kill|runtime_error|terminate|timed|unexpected"
    failed_pattern+="|[^-W]error|exception"
    failed_pattern+="|job ending due to application timeout"

    exclude_pattern="\-\-abort\-signal|CCL_ABORT_ON_THROW"
    exclude_pattern+="|fi_strerror|MPI_Error_string"

    skipped_pattern="skip test|unavailable"

    test_passed=`grep -E -c -i "${passed_pattern}" ${test_log}`
    test_failed=`grep -E -i "${failed_pattern}" ${test_log} | grep -Evc "${exclude_pattern}"`
    test_skipped=`grep -E -c -i "${skipped_pattern}" ${test_log}`

    total_tests=$((total_tests + 1))
    result=""
    if ([ ${test_passed} -eq 0 ] || [ ${test_skipped} -eq 0 ]) && [ ${test_failed} -ne 0 ]
    then
        echo "Error: example $test_file testing failed"
        echo "See log ${test_log} for details"
        total_fails=${total_fails}+1
        result="failed"
        cat <<EOT >> "${SCRIPT_DIR}/${JUNIT_REPORT}"
    <testcase name="${test}">
      <failure type="TestFailed" message="Test ${test_file} failed">
        <![CDATA[
$(tail -30 "${test_log}")
        ]]>
      </failure>
    </testcase>
EOT

    elif [ ${test_skipped} -ne 0 ]
    then
        echo "Test $test_file has been skipped."
        echo "See log ${test_log} for details"
        total_skipped=${total_skipped}+1
        result="skipped"
        cat <<EOT >> "${SCRIPT_DIR}/${JUNIT_REPORT}"
    <testcase name="${test}">
      <skipped type="TestSkipped" message="Test skipped" />
    </testcase>
EOT

    else
        result="passed"
        echo "    <testcase name=\"${test}\" />" >> "${SCRIPT_DIR}/${JUNIT_REPORT}"
    fi

cat <<EOT >> "${SCRIPT_DIR}/${JSON_REPORT}.tmp"
    {
        "name":"${test}",
        "result": "${result}"
    }
    ,
EOT

    if [[ ${ENABLE_CLEANUP} == "yes" ]]
    then
        if [[ ${test_skipped} -eq 0 ]] && [[ ${test_failed} -eq 0 ]]
        then
            rm -f ${test_log}
        fi
    fi
}

set_ccl_cpu_env()
{
    #oneAPI setvars.sh case
    if [ -f ${CCL_ROOT}/env/vars.sh ]; then
        source ${CCL_ROOT}/env/vars.sh --ccl-configuration=cpu
    #oneAPI oneapi-vars.sh case
    elif [ -f ${CCL_ROOT}/../ccl/latest/env/vars.sh ]; then
        source ${CCL_ROOT}/../ccl/latest/env/vars.sh --ccl-configuration=cpu
    else
        echo "ERROR: vars.sh wasn't found, please set oneCCL environment"
        exit 1
    fi
}

check_dpcpp()
{
    which dpcpp
    COMPILER_INSTALL_CHECK=$?
    if [ "$COMPILER_INSTALL_CHECK" != "0" ]
    then
        echo "Warning: dpcpp compiler wasn't found, ccl-configuration=cpu will be sourced"
        set_ccl_cpu_env
        MODE="cpu"
    fi
}

check_environment()
{
    for platform in $unsupported_platforms;
    do
         if [[ "$DASHBOARD_PLATFORM_HW_DISCRETE_GPU" == $platform ]] ||
            [[ -z "$DASHBOARD_PLATFORM_HW_DISCRETE_GPU" && "$DASHBOARD_PLATFORM_HW_INTEGRATED_GPU" == $platform ]]
         then
            echo "Warning: oneCCL is not supported on $platform. Testing will not be run."
            # generate an empty json report, since no tests were processed
            echo "[]" > "${SCRIPT_DIR}/${JSON_REPORT}"
            exit 0
        fi
    done

    if [[ -z "${I_MPI_ROOT}" ]]
    then
        which mpiexec.hydra >/dev/null 2>&1
        if [[ $? != 0 ]]
        then
            echo "Error: MPI not found."
            exit 1
        fi
    fi

    if [[ -z "${CCL_ROOT}" ]]
    then
        echo "Error: CCL_ROOT wasn't found"
        exit 1
    fi

    if [[ ! -z ${FI_PROVIDER} ]]
    then
        provider="${FI_PROVIDER}"
    else
        provider="tcp"
    fi

    if [[ ! -z ${CCL_ATL_TRANSPORT_LIST} ]]
    then
        transport_list="${CCL_ATL_TRANSPORT_LIST}"
    else
        transport_list="ofi mpi"
    fi

    if [ ! -z "${CCL_WORKER_COUNT}" ]
    then
        worker_count="${CCL_WORKER_COUNT}"
    else
        worker_count="1"
    fi

    if [[ ! -z ${CCL_RUNTIME_LIST} ]]
    then
        allowed_runtime_list="${CCL_RUNTIME_LIST}"
    else
        allowed_runtime_list=""level_zero:gpu" "opencl:gpu""
    fi
    allowed_runtime_list+=" none"
}

run_benchmark()
{
    echo "================ENVIRONMENT=================="
    local ccl_extra_env=$1
    echo "ccl_extra_env: " $ccl_extra_env
    local dir_name=$2
    echo "dir_name: " $dir_name
    local transport=$3
    echo "transport: "$transport
    local example=$4
    echo "example: "$example
    local backend=$5
    echo "backend: "$backend
    local runtime=$6
    echo "runtime: "$runtime
    local n=$7
    echo "n: "$n
    local ppn=$8
    echo "ppn: "$ppn
    local coll=${9}
    echo "coll: " $coll
    local dtype=${10}
    echo "dtype: " $dtype
    local reduction=${11}
    echo "reduction: " $reduction
    local ranks=${12}
    echo "ranks per process: " ${ranks}
    echo "================ENVIRONMENT=================="

    log_idx=${log_idx}+1
    base_test_log="$LOGS_DIR/$dir_name/run"
    base_test_log="${base_test_log}_${transport}_${example}_b_${backend}_r_${runtime}_l_${coll}_d_${dtype}_${log_idx}"

    options="--check last"
    usm_list="none"
    ccl_bench_env="${ccl_extra_env}"

    if [ "${backend}" != "" ];
    then
        options="${options} --backend ${backend}"
    fi

    if [ "${coll}" != "" ];
    then
        options="${options} --coll ${coll}"
    fi

    if [ "${dtype}" != "" ];
    then
        options="${options} --dtype ${dtype}"
    fi

    if [ "${reduction}" != "" ];
    then
        options="${options} --reduction ${reduction}"
    fi

    # run benchmark with arbitraty sizes that should cover all the cases
    if [ "${DASHBOARD_PLATFORM_HW_MACHINE_TYPE}" == "virtual" ]
    then
        # reduce list of sizes for virtual machines to avoid high load the system
        options="${options} -y 1,2,8,133,16387"
        buf_count=4
        iter_count=4
    else
        # run benchmark with arbitraty sizes that should cover all the cases
        options="${options} -y 1,2,4,7,8,16,17,32,64,128,133,256,1077,16384,16387"
        buf_count=16
        iter_count=16
    fi

    extra_options=""
    if [ "${backend}" == "sycl" ];
    then
        buf_count=4
        iter_count=8
        extra_options="--sycl_mem_type usm"

        #usm_list="device shared" https://jira.devtools.intel.com/browse/MLSL-902
        usm_list="device"

        if [ "${runtime}" == "level_zero:gpu" ];
        then
            # TODO: submit compiler ticket
            buf_count=1
            # ZE_SERIALIZE hangs for in order queue which is by default now
            # ccl_bench_env="ZE_SERIALIZE=2 ${ccl_extra_env}"
        fi
    fi
    options="${options} --buf_count ${buf_count} --iters ${iter_count} ${extra_options}"

    for usm in $usm_list;
    do
        echo "usm: " $usm

        local test_log="${base_test_log}_${usm}.log"
        mkdir -p $(dirname "${test_log}")

        if [ "${backend}" == "sycl" ];
        then
            final_options="${options} --sycl_usm_type ${usm}"
        else
            final_options="${options}"
        fi

        if [ `echo $ccl_bench_env | grep -c CCL_LOG_LEVEL` -ne 1 ]
        then
            cmd=`echo $ccl_bench_env mpiexec.hydra -n $n -ppn $ppn -l ./$example ${final_options}`
            echo "Running: $cmd"
            eval $cmd 2>&1 | tee ${test_log}
            check_test ${test_log} ${example}
        else
            log_debug_test_log="${test_log}.${n}_ranks_ppn_${ppn}"
            echo "output for run with CCL_LOG_LEVEL=debug and ${n} ranks has been redirected to log file ${log_debug_test_log}"
            cmd=`echo $ccl_bench_env mpiexec.hydra -n $n -ppn $ppn -l ./$example ${final_options}`
            echo "Running: $cmd"
            eval $cmd > ${log_debug_test_log} 2>&1
            check_test ${log_debug_test_log} ${example}

            if [ "${transport}" == "ofi" ] && [ "${ppn}" == "1" ];
            then
                log_debug_test_log="${test_log}.${n}_rank"
                echo "output for run with CCL_LOG_LEVEL=debug and 1 rank has been redirected to log file ${log_debug_test_log}"
                cmd=`echo $ccl_bench_env mpiexec.hydra -n $n -ppn $ppn -l ./$example ${final_options}`
                echo "Running: $cmd"
                eval $cmd > ${log_debug_test_log} 2>&1
                check_test ${log_debug_test_log} ${example}
            fi
        fi
    done
}

run_example()
{
    echo "================ENVIRONMENT=================="
    local ccl_extra_env="$1"
    echo "ccl_extra_env: " "$ccl_extra_env"
    local dir_name=$2
    echo "dir_name: " $dir_name
    local transport=$3
    echo "transport: "$transport
    local n=$4
    echo "n: "$n
    local ppn=$5
    echo "ppn: "$ppn
    local example=$6
    echo "example: "$example
    local arg=$7
    echo "arg: "$arg
    echo "================ENVIRONMENT=================="

    log_idx=${log_idx}+1
    local test_log="$LOGS_DIR/$dir_name/run_${dir_name}_${transport}_${example}_${log_idx}.log"
    mkdir -p $(dirname "${test_log}")

    cmd=`echo $ccl_extra_env mpiexec.hydra -n $n -ppn $ppn -l ./$example $arg`
    echo "Running $cmd"
    eval $cmd 2>&1 | tee ${test_log}
    check_test ${test_log} ${example}
}

build()
{
    cd ${EXAMPLE_WORK_DIR}
    # please use line below for manual testing by run.sh
    # cp ${EXAMPLE_WORK_DIR}/../../ccl_oneapi/CMakeLists.txt ${EXAMPLE_WORK_DIR}/../
    if [[ ! -z ${COMPUTE_BACKEND} ]]
    then
        EX_COMPUTE_BACKEND="-DCOMPUTE_BACKEND=${COMPUTE_BACKEND}"
    fi
    echo "Building"
    local BUILD_OUTPUT_LOG=${LOGS_DIR}/build_output.log
    mkdir -p $(dirname "${BUILD_OUTPUT_LOG}")
    cmake .. -DCMAKE_C_COMPILER="${C_COMPILER}" \
                -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
                ${EX_COMPUTE_BACKEND} \
                2>&1 | tee ${BUILD_OUTPUT_LOG}
    make -j 2>&1 | tee -a ${BUILD_OUTPUT_LOG}
    error_count=`grep -E -i -c 'error|abort|fail'  ${BUILD_OUTPUT_LOG}` > /dev/null 2>&1
    if [ "${error_count}" != "0" ]
    then
        echo "building ... NOK"
        echo "See logs ${BUILD_OUTPUT_LOG}"
        exit 1
    else
        echo "OK"
    fi
}

run()
{
    cd ${EXAMPLE_WORK_DIR}/
    pwd

    n=`echo ${PROC_MAP%:*}`
    ppns=`echo ${PROC_MAP#*:} | tr ',' ' '`
    base_dtype_list="int8,int32,int64,float16,float32"
    ccl_base_env="FI_PROVIDER=${provider} CCL_WORKER_COUNT=${worker_count}"
    reduction_list="sum"
    sycl_example_selector_list="none"
    # fix for https://jira.devtools.intel.com/browse/MLSL-1764
    if [[ ${DASHBOARD_PLATFORM_HW_CPU_INSTRUCTION_SET_EXTENTIONS} != *"f16c"* ]]
    then
        base_dtype_list="int8,int32,float32"
        echo "WARNING: float16 datatype was skipped since DASHBOARD_PLATFORM_HW_CPU_INSTRUCTION_SET_EXTENTIONS doesn't have f16c instruction"
    fi

    if [[ "${DASHBOARD_PLATFORM_SW_LINUX_KERNEL_VERSION}" == *"WSL2"* ]]
    then
       echo "WARNING: CCL_ZE_IPC_EXCHANGE=sockets was set up for WSL2"
       ccl_base_env+=" CCL_ZE_IPC_EXCHANGE=sockets"
    fi

    if [[ ${MODE} = "cpu" ]]
    then
        dir_list="benchmark common cpu"
        bench_backend_list="host"

        # additional mode for testing external_launcher in lab only due to unstable result
        # in PV lab and https://jira.devtools.intel.com/browse/MLSL-808
        if [[ ${SCOPE} != "pv" ]]
        then
            dir_list+=" external_launcher"
        fi
    else
        if [[ ${SCOPE} != "pv" ]]
        then
            transport_list+=" mpi_gpu"
        fi

        #MLSL-1461
        if [ "${DASHBOARD_PLATFORM_HW_DISCRETE_GPU}" == "dg1" ]
        then
            ccl_base_env+=" CCL_ZE_DISABLE_FAMILY_CHECK=1"
        fi
        ccl_base_env+=" OverrideDefaultFP64Settings=1 IGC_EnableDPEmulation=1"
        
        #MLSL-3883
        if [ "${DASHBOARD_PLATFORM_HW_DISCRETE_GPU}" == "arcb" ]
        then
            # atomics do not work at arcb
            ccl_base_env+=" CCL_SYCL_CCL_BARRIER=1"
            ccl_base_env+=" EnableImplicitScaling=0"
            # some implementations require to disable compression for BMG
            ccl_base_env+=" RenderCompressedBuffersEnabled=0"
            # this is needed to support disabled compression
            ccl_base_env+=" NEOReadDebugKeys=1"
            # ccl_base_env+=" CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0"
            # commented out to run Zhong's code instead of Richard's code
            # ccl_base_env+=" CCL_SYCL_ALLREDUCE_ARC=1"
            # ccl_base_env+=" CCL_SYCL_ALLTOALL_ARC_LL=1"
        fi

        # removed ZE_ENABLE_VALIDATION_LAYER=1 as a w/a https://jira.devtools.intel.com/browse/MLSL-2149
        ccl_base_env+="  ZE_ENABLE_PARAMETER_VALIDATION=1"
        common_dir_list="benchmark common"

        if [[ ${SCOPE} = "pr" ]]
        then
            dir_list="${common_dir_list} sycl"
            bench_backend_list="sycl"
            sycl_example_selector_list="gpu"
        elif [[ ${SCOPE} = "abi" ]]
        then
            dir_list="${common_dir_list} cpu sycl"
            bench_backend_list="host sycl"
            sycl_example_selector_list="cpu gpu"
        elif [[ ${SCOPE} = "all" || ${SCOPE} = "pv" ]]
        then
            dir_list="${common_dir_list} cpu sycl"
            bench_backend_list="sycl" # host sycl
            sycl_example_selector_list="gpu" # cpu gpu
        fi
    fi

    echo "dir_list =" $dir_list "; sycl_example_selector_list =" $sycl_example_selector_list
    for dir_name in $dir_list
    do
        if [ ! -d "$(pwd)/${dir_name}" ]
        then
            echo "Directory $(pwd)/${dir_name} not found"
            continue
        fi

        cd $dir_name
        pwd
        for transport in $transport_list
        do
            ccl_transport_env="${ccl_base_env}"
            transport_name=${transport}
            if [ "$transport" == "mpi_gpu" ];
            then
                # TODO: https://jira.devtools.intel.com/browse/MLSL-1157
                continue

                if [ "$dir_name" != "sycl" ] && [ "$dir_name" != "benchmark" ]
                then
                    continue
                fi
                transport_name="mpi"
                ccl_transport_env+=" CCL_ATL_HMEM=1"
            fi
            ccl_transport_env+=" CCL_ATL_TRANSPORT=${transport_name}"

            if [[ "${transport}" == *"mpi"* ]]
            then
                examples_to_run=`find . -type f -executable -printf '%P\n' |
                    grep -v 'run_binary.sh' |
                    grep -v 'run.sh' |
                    grep -v 'external_launcher'`
            else
                examples_to_run=`find . -type f -executable -printf '%P\n' |
                    grep -v 'run_binary.sh' |
                    grep -v 'run.sh'`
            fi

            for example in $examples_to_run
            do
                for ppn in $ppns
                do
                    if [[ "$ppn" -gt "$n" ]];
                    then
                        continue
                    fi

                    if [ "$dir_name" == "benchmark" ];
                    then

                        coll_list="all"

                        for backend in $bench_backend_list
                        do
                            runtime_list="none"
                            if [ "$backend" == "sycl" ];
                            then
                                runtime_list=""opencl:gpu" "level_zero:gpu""
                            fi

                            if [ "$transport" == "mpi_gpu" ];
                            then
                                if [ "$backend" != "sycl" ];
                                then
                                    continue
                                else
                                    runtime_list="level_zero:gpu"
                                fi
                            fi

                            for runtime in $runtime_list
                            do
                                if [[ "${allowed_runtime_list}" != *"${runtime}"* ]]
                                then
                                    continue
                                fi

                                ccl_runtime_env="${ccl_transport_env}"
                                if [ "$runtime" != "none" ]
                                then
                                    ccl_runtime_env="ONEAPI_DEVICE_SELECTOR=${runtime} ${ccl_runtime_env}"
                                fi

                                if [[ "$runtime" == "level_zero:gpu" && "${DASHBOARD_PLATFORM_HW_DISCRETE_GPU}" == *"ats"* ]]
                                then
                                    dtype_list="${base_dtype_list},bfloat16"
                                else
                                    dtype_list="${base_dtype_list}"
                                fi

                                ccl_extra_env="${ccl_runtime_env}"
                                run_benchmark "${ccl_extra_env}" ${dir_name} ${transport} ${example} ${backend} ${runtime} ${n} ${ppn} ${coll_list}

                                ccl_extra_env="CCL_PRIORITY=lifo ${ccl_runtime_env}"
                                run_benchmark "${ccl_extra_env}" ${dir_name} ${transport} ${example} ${backend} ${runtime} ${n} ${ppn} ${coll_list}

                                ccl_extra_env="CCL_WORKER_OFFLOAD=0 ${ccl_runtime_env}"
                                run_benchmark "${ccl_extra_env}" ${dir_name} ${transport} ${example} ${backend} ${runtime} ${n} ${ppn} ${coll_list}

                                ccl_extra_env="CCL_WORKER_WAIT=0 ${ccl_runtime_env}"
                                run_benchmark "${ccl_extra_env}" ${dir_name} ${transport} ${example} ${backend} ${runtime} ${n} ${ppn} ${coll_list}

                                ccl_extra_env="CCL_KERNEL_GROUP_SIZE=0 ${ccl_runtime_env}"
                                run_benchmark "${ccl_extra_env}" ${dir_name} ${transport} ${example} ${backend} ${runtime} ${n} ${ppn} ${coll_list}

                                if [ "$runtime" != "opencl:gpu" ];
                                then
                                    # TODO: fix issue with OCL host usm, ticket to be filled
                                    ccl_extra_env="CCL_FUSION=1 ${ccl_runtime_env}"
                                    run_benchmark "${ccl_extra_env}" ${dir_name} ${transport} ${example} ${backend} ${runtime} ${n} ${ppn} allreduce
                                fi

                                ccl_extra_env="CCL_LOG_LEVEL=debug ${ccl_runtime_env}"
                                run_benchmark "${ccl_extra_env}" ${dir_name} ${transport} ${example} ${backend} ${runtime} ${n} ${ppn} ${coll_list}
                                run_benchmark "${ccl_extra_env}" ${dir_name} ${transport} ${example} ${backend} ${runtime} ${n} ${ppn} allreduce ${dtype_list} ${reduction_list}
                            done
                        done
                    elif [ "$dir_name" == "sycl" ];
                    then
                        for selector in $sycl_example_selector_list
                        do
                            if [ "$selector" == "gpu" ];
                            then
                                runtime_list=""opencl:gpu" "level_zero:gpu""
                            else
                                runtime_list="opencl:gpu"
                            fi

                            if [ "$transport" == "mpi_gpu" ];
                            then
                                if [ "$selector" != "gpu" ];
                                then
                                    continue
                                else
                                    runtime_list="level_zero:gpu"
                                fi
                            fi

                            if [[ "${example}" == *"_usm_"* ]]
                            then
                                if [ "$selector" == "gpu" ];
                                then
                                    #usm_list="device shared" https://jira.devtools.intel.com/browse/MLSL-902
                                    usm_list="device"
                                else
                                    usm_list="host shared"
                                fi
                            else
                                usm_list="default"
                            fi

                            for runtime in $runtime_list
                            do
                                if [[ "${allowed_runtime_list}" != *"${runtime}"* ]]
                                then
                                    continue
                                fi

                                if [ "$selector" == "cpu" ] && [ "$runtime" == "opencl:gpu" ] && [ "$usm_list" != "default" ];
                                then
                                    # MLSL-856
                                    continue
                                fi

                                for usm in $usm_list
                                do
                                    if [[ "${example}" == "sycl_allgatherv_usm_comm_split" || \
                                          "${example}" == "sycl_reduce_scatter_usm_comm_split" || \
                                          "${example}" == "sycl_allreduce_usm_comm_split" ]] && [[ "${n}" != 4 ]]; then
                                        continue
                                    fi
                                    echo "selector $selector, runtime $runtime, usm $usm"
                                    ccl_extra_env="ONEAPI_DEVICE_SELECTOR=${runtime} ${ccl_transport_env}"
                                    run_example "${ccl_extra_env}" ${dir_name} ${transport} ${n} ${ppn} ${example} "${selector} ${usm}"
                                done
                            done
                        done
                    elif [ "$dir_name" == "external_launcher" ];
                    then
                        if [ "$transport" != "ofi" ];
                        then
                            continue
                        fi

                        if [ -z "${I_MPI_HYDRA_HOST_FILE}" ];
                        then
                            echo "WARNING: I_MPI_HYDRA_HOST_FILE was not set."
                            echo localhost > ${SCRIPT_DIR}/${dir_name}/hostfile
                            export I_MPI_HYDRA_HOST_FILE=${SCRIPT_DIR}/${dir_name}/hostfile
                        fi

                        ccl_hosts=`echo $I_MPI_HYDRA_HOST_FILE`
                        ccl_world_size=4
                        ccl_vars="${CCL_ROOT}/env/setvars.sh"
                        if [ ! -f "$ccl_vars" ];
                        then
                            # not a standalone version
                            ccl_vars="${CCL_ROOT}/env/vars.sh"
                        fi

                        test_log="$LOGS_DIR/$dir_name/log.txt"
                        mkdir -p $(dirname "${test_log}")
                        ${EXAMPLE_WORK_DIR}/${dir_name}/run.sh -v $ccl_vars -h $ccl_hosts -s $ccl_world_size 2>&1 | tee ${test_log}
                        check_test ${test_log} "external_launcher"

                    else
                        if [[ "${example}" == "sycl_allgatherv_usm_comm_split" || \
                              "${example}" == "sycl_reduce_scatter_usm_comm_split" || \
                              "${example}" == "sycl_allreduce_usm_comm_split" ]] && [[ "${n}" != 4 ]]; then
                            continue
                        fi
                        ccl_extra_env="${ccl_transport_env}"
                        run_example "${ccl_extra_env}" ${dir_name} ${transport} ${n} ${ppn} ${example}
                    fi
                done
            done
        done
        cd - > /dev/null 2>&1
    done

    cat <<EOT >> "${SCRIPT_DIR}/${JSON_REPORT}"
[
$(sed '$d'  "${SCRIPT_DIR}/${JSON_REPORT}.tmp")
]
EOT

    # Finalize JUnit XML
    sed -i "s/tests=\"0\"/tests=\"${total_tests}\"/; s/failures=\"0\"/failures=\"${total_fails}\"/; s/skipped=\"0\"/skipped=\"${total_skipped}\"/" "${SCRIPT_DIR}/${JUNIT_REPORT}"
    echo "  </testsuite></testsuites>" >> "${SCRIPT_DIR}/${JUNIT_REPORT}"

    if [ ${total_fails} != 0 ]
    then
        echo "There are ${total_fails} failed tests"
        sleep 20
        exit 1
    elif [ ${total_skipped} != 0 ]
    then
        echo "Tests passed, except ${total_skipped} skipped tests"
        sleep 20
        exit 0
    else
        echo "All tests passed!"
        sleep 20
        exit 0
    fi
}

echo_log()
{
    echo -e "$*"
}

print_help()
{
    echo_log "Usage: ${BASENAME}.sh <options>"
    echo_log "  --mode mode             cpu|gpu mode"
    echo_log "  --scope scope           pr|abi|pv|all scope (default: pv)"
    echo_log "  --build-only            build only"
    echo_log "  --run-only              run only"
    echo_log "  --cleanup               cleanup"
    echo_log "  --proc-map <n:ppns>     set value for n and ppns"
    echo_log "  --help                  print help information"
    echo_log
    echo_log "Usage examples:"
    echo_log "  ${BASENAME}.sh "
    echo_log "  ${BASENAME}.sh --mode cpu"
    echo_log "  ${BASENAME}.sh --mode gpu"
    echo_log "  ${BASENAME}.sh --mode gpu --scope pr"
    echo_log "  ${BASENAME}.sh --mode gpu --scope pr --proc-map 4:1,2,4"
    echo_log ""
    echo_log "Available knobs:"
    echo_log "  C_COMPILER = clang|icc|icx|gcc|<full path to compiler>"
    echo_log "  CXX_COMPILER = clang++|icpc|icpx|g++|<full path to compiler>"
    echo_log ""
}

parse_arguments()
{
    while [ $# -ne 0 ]
    do
        case $1 in
            "-h" | "--help" | "-help")                  print_help ; exit 0 ;;
            "--mode" | "-mode")                         MODE=$2 ; shift ;;
            # ./run.sh cpu|gpu used for backward compatibility for PV
            "gpu")                                      MODE="gpu" ;;
            "cpu")                                      MODE="cpu" ;;
            "pr")                                       SCOPE="pr" ;;
            "--scope" | "-scope")                       SCOPE=$2 ; shift ;;
            "--build-only" | "-build-only")             BUILD_ONLY="yes" ;;
            "--run-only" | "-run-only")                 RUN_ONLY="yes" ;;
            "--cleanup" | "-cleanup")                   ENABLE_CLEANUP="yes" ;;
            "--proc-map" | "-proc-map")                 PROC_MAP=$2 ; shift ;;
            *)
            echo "$(basename $0): ERROR: unknown option ($1)"
            print_help
            exit 1
            ;;
        esac
        shift
    done

    echo "Parameters:"
    [[ ! -z "$MODE" ]] && echo "MODE: $MODE"
    [[ ! -z "$SCOPE" ]] && echo "SCOPE: $SCOPE"
    [[ ! -z "$BUILD_ONLY" ]] && echo "BUILD_ONLY: $BUILD_ONLY"
    [[ ! -z "$RUN_ONLY" ]] && echo "RUN_ONLY: $RUN_ONLY"
    [[ -z "$PROC_MAP"  ]] && PROC_MAP="2:1,2"; echo "PROC_MAP: $PROC_MAP"
    echo
}

check_mode()
{
    if [[ -z ${MODE} ]]
    then
        if [[ ! ",${DASHBOARD_INSTALL_TOOLS_INSTALLED}," == *",dpcpp,"* ]] || [[ ! -n ${DASHBOARD_GPU_DEVICE_PRESENT} ]]
        then
            echo "WARNING: DASHBOARD_INSTALL_TOOLS_INSTALLED variable doesn't contain 'dpcpp' or the DASHBOARD_GPU_DEVICE_PRESENT variable is missing"
            echo "WARNING: Using cpu configuration of the library"
            set_ccl_cpu_env
            MODE="cpu"
        else
            MODE="gpu"
        fi
    fi

    case $MODE in
    "cpu" )
        if [[ -z "${C_COMPILER}" ]]; then
            if [[ ",${DASHBOARD_INSTALL_TOOLS_INSTALLED}," == *",icx,"* ]]; then
                C_COMPILER=icx
            elif [[ ",${DASHBOARD_INSTALL_TOOLS_INSTALLED}," == *",icc,"* ]]; then
                C_COMPILER=icc
            else
                C_COMPILER=gcc
            fi
        fi
        if [[ -z "${CXX_COMPILER}" ]]; then
            if [[ ",${DASHBOARD_INSTALL_TOOLS_INSTALLED}," == *",icx,"* ]]; then
                CXX_COMPILER=icpx
            elif [[ ",${DASHBOARD_INSTALL_TOOLS_INSTALLED}," == *",icc,"* ]]; then
                CXX_COMPILER=icpc
            else
                CXX_COMPILER=g++
            fi
        fi
        ;;
    "gpu"|* )
        check_dpcpp
        if [ -z "${C_COMPILER}" ]
        then
            C_COMPILER=icx
        fi
        if [ -z "${CXX_COMPILER}" ]
        then
            CXX_COMPILER=icpx
        fi
        if [ -z "${COMPUTE_BACKEND}" ]
        then
            COMPUTE_BACKEND=dpcpp
        fi
        ;;
    esac
}

check_scope()
{
    if [[ ! -z ${SCOPE} ]]
    then
        case $SCOPE in
            "pr" | "abi" | "pv" | "all") ;;
            *)
            echo "Unsupported scope: $SCOPE"
            print_help
            exit 1
            ;;
        esac
    else
        SCOPE="pv"
    fi
}

parse_arguments $@
check_environment
check_mode
create_logs_dir

if [[ ${RUN_ONLY} != "yes" ]]
then
    check_scope
    create_work_dir
    if [[ -z ${BUILD_ONLY} ]] || [[ ${BUILD_ONLY} == "no" ]]
    then
        build
    else
        build
        exit 0
    fi
fi

if [[ -z ${BUILD_ONLY} ]] || [[ ${BUILD_ONLY} == "no" ]]
then
    if [[ -z ${RUN_ONLY} ]] || [[ ${RUN_ONLY} == "no" ]]
    then
        run
    else
        set_example_work_dir
        run
        exit 0
    fi
fi
