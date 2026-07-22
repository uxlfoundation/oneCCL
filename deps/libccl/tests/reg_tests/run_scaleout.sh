#!/bin/bash

BASENAME=$(basename $0 .sh)
SCRIPT_DIR=`cd $(dirname "$BASH_SOURCE") && pwd -P`
JUNIT_REPORT="summary.junit.xml"
JSON_REPORT="test_cases.json"

CORE_COUNT=$(( $(lscpu | grep "^Socket(s):" | awk '{print $2}' ) * $(lscpu | grep "^Core(s) per socket:" | awk '{print $4}') ))
MAKE_JOB_COUNT=$(( CORE_COUNT / 3 > 4 ? CORE_COUNT / 3 : 4 ))

source ${SCRIPT_DIR}/utils.sh

pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

print_help() {
    echo ""
    echo "Usage: ${BASENAME}.sh <options>"
    echo "  --mode <mode> (default gpu)"
    echo "      cpu|gpu mode"
    echo "  --exclude-list <file>"
    echo "      Specific path to the file with excluded tests"
    echo "  --build-only"
    echo "      Enable only the build stage"
    echo "  --platform <ats|gen|pvc>"
    echo "      Discrete GPU platform name"
    echo "  --transport <impi|mpich>"
    echo "      Transport name (default impi)"
    echo "  --pv "
    echo "      Set up PV testing environment"
    echo "  --inference <0|1> (default 0)"
    echo "      Test inference portion (only with gpu mode)"
    echo ""
    echo "Usage examples:"
    echo "  ${BASENAME}.sh --mode cpu"
    echo "  ${BASENAME}.sh --exclude-list /p/pdsd/Users/asoluyan/GitHub/oneccl/tests/reg_tests/exclude_list"
    echo "  ${BASENAME}.sh --build-only"
    echo "  ${BASENAME}.sh --mode gpu --platform gen"
    echo ""
}

escape_xml() {
    local in=$1

    echo "$( sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g; s/'"'"'/\&#39;/g' <<< "$in" )"
}

print_header() {
    echo "#"
    echo "# ${1}"
    echo "#"
}

is_exclude_test() {
    local TEST_NAME=${1}
    rc=0
    while read -r res
    do
        is_exclude_platform=0
        is_exclude_transport=0
        EXCLUDE_RULES=$(echo $res | sed "s/^${TEST_NAME}\s//g")
        for item in ${EXCLUDE_RULES}
        do
            prop="$(echo ${item} | awk -F "=" '{print $1}')"
            value="$(echo ${item} | awk -F "=" '{print $2}')"
            if [[ ${prop} = "platform" ]]
            then
                if [[ ${value} = "${PLATFORM_HW_GPU}" || ${value} = "*" ]]
                then
                    is_exclude_platform=1
                fi
            elif [[ ${prop} = "transport" ]]
            then
                if [[ ${value} = *"${TRANSPORT}"* || ${value} = "*" ]]
                then
                    is_exclude_transport=1
                fi
            else
                echo "WARNING: unknown excluded property (${prop})"
            fi
        done
        if [[ "${is_exclude_platform}" = 1 && "${is_exclude_transport}" = 1 ]]
        then
            rc=1
            break
        fi
    done < <(grep -w ${TEST_NAME} ${EXCLUDE_LIST})
    echo ${rc}
}

set_default_values() {
    ENABLE_BUILD="yes"
    ENABLE_TESTING="yes"
    EXCLUDE_LIST="${SCRIPT_DIR}/exclude_list_scaleout"

    export I_MPI_JOB_TIMEOUT=600
    export CCL_LOG_LEVEL=info
    export CCL_WORKER_COUNT=1
    export I_MPI_DEBUG=12
}

set_gen_env() {
    if [[ ${PLATFORM_HW_GPU} = "gen" ]]
    then
        export CCL_YIELD=sched_yield
    fi
}

parse_arguments() {
    while [ $# -ne 0 ]
    do
        case $1 in
            "-h" | "--help" | "-help")
                print_help
                exit 0
                ;;
            "-mode"|"--mode")
                export MODE=${2}
                shift
                ;;
            "-exclude-list"|"--exclude-list")
                export EXCLUDE_LIST=${2}
                shift
                ;;
            "-build-only"|"--build-only")
                ENABLE_TESTING="no"
                ;;
            "-platform"|"--platform")
                export PLATFORM_HW_GPU=${2}
                shift
                ;;
            "-transport"|"--transport")
                export TRANSPORT=${2}
                shift
                ;;
            "-pv"|"--pv")
                export PV_ENVIRONMENT="yes"
                ;;
            "-inference"|"--inference")
                export INFERENCE_TESTS=${2}
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

    if [[ ${PV_ENVIRONMENT} = "yes" ]]
    then
        echo "$(cat ${EXCLUDE_LIST} ${SCRIPT_DIR}/exclude_list_pv)" > ${SCRIPT_DIR}/exclude_list_pv
        EXCLUDE_LIST=${SCRIPT_DIR}/exclude_list_pv
    fi

    check_mode

    if [[ ${MODE} = "gpu" ]] && [[ -z ${PLATFORM_HW_GPU} ]]
    then
        if [[ ${PV_ENVIRONMENT} = "yes" ]]
        then
            if [[ ! -z ${DASHBOARD_PLATFORM_HW_DISCRETE_GPU} ]]
            then
                if [[ ${DASHBOARD_PLATFORM_HW_DISCRETE_GPU} = "ats"* ]]
                then
                    export PLATFORM_HW_GPU="ats"
                elif [[ ${DASHBOARD_PLATFORM_HW_DISCRETE_GPU} = "pvc" ]]
                then
                    export PLATFORM_HW_GPU="pvc"
                elif [[ ${DASHBOARD_PLATFORM_HW_DISCRETE_GPU} = "dg"* ]]
                then
                    export PLATFORM_HW_GPU="dg"
                else
                    echo "Platform not defined: ${DASHBOARD_PLATFORM_HW_DISCRETE_GPU}"
                fi
            else
                export PLATFORM_HW_GPU="gen"
            fi
        else
            export PLATFORM_HW_GPU="ats"
        fi
    fi

    if [[ -z ${TRANSPORT} ]]
    then
        export TRANSPORT="impi"
    fi

    if [[ -z ${INFERENCE_TESTS} ]]
    then
        export INFERENCE_TESTS="0"
    fi


    echo "-----------------------------------------------------------"
    echo "PARAMETERS"
    echo "-----------------------------------------------------------"
    echo "MODE                     = ${MODE}"
    echo "ENABLE_BUILD             = ${ENABLE_BUILD}"
    echo "ENABLE_TESTING           = ${ENABLE_TESTING}"
    echo "EXCLUDE_LIST             = ${EXCLUDE_LIST}"
    echo "PLATFORM_HW_GPU          = ${PLATFORM_HW_GPU}"
    echo "TRANSPORT                = ${TRANSPORT}"
    echo "PV_ENVIRONMENT           = ${PV_ENVIRONMENT}"
    echo "INFERENCE_TESTS               = ${INFERENCE_TESTS}"
}

check_mode() {
    if [[ -z ${MODE} ]]
    then
        if [[ ! ",${DASHBOARD_INSTALL_TOOLS_INSTALLED}," == *",dpcpp,"* ]] || [[ ! -n ${DASHBOARD_GPU_DEVICE_PRESENT} ]]
        then
            echo "WARNING: DASHBOARD_INSTALL_TOOLS_INSTALLED variable doesn't contain 'dpcpp' or the DASHBOARD_GPU_DEVICE_PRESENT variable is missing"
            echo "WARNING: Using cpu configuration of the library"
            source ${CCL_ROOT}/env/vars.sh --ccl-configuration=cpu
            export MODE="cpu"
        else
            export MODE="gpu"
        fi
    fi
}

define_compiler() {
    case ${MODE} in
    "cpu" )
        if [[ -z "${C_COMPILER}" ]]
        then
            if [[ ",${DASHBOARD_INSTALL_TOOLS_INSTALLED}," == *",icx,"* ]]
            then
                C_COMPILER=icx
            elif [[ ",${DASHBOARD_INSTALL_TOOLS_INSTALLED}," == *",icc,"* ]]
            then
                C_COMPILER=icc
            else
                C_COMPILER=gcc
            fi
        fi
        if [[ -z "${CXX_COMPILER}" ]]
        then
            if [[ ",${DASHBOARD_INSTALL_TOOLS_INSTALLED}," == *",icx,"* ]]
            then
                CXX_COMPILER=icpx
            elif [[ ",${DASHBOARD_INSTALL_TOOLS_INSTALLED}," == *",icc,"* ]]
            then
                CXX_COMPILER=icpc
            else
                CXX_COMPILER=g++
            fi
        fi
        ;;
    "gpu"|* )
        if [ -z "${C_COMPILER}" ]
        then
            C_COMPILER=icx
        fi
        if [ -z "${CXX_COMPILER}" ]
        then
            CXX_COMPILER=icpx
            COMPUTE_BACKEND="dpcpp"
        fi
        ;;
    esac
}

build() {
    if [[ ${ENABLE_BUILD} = "yes" ]]
    then
        print_header "Build tests..."
        rm -rf ${SCRIPT_DIR}/build
        mkdir ${SCRIPT_DIR}/build
        pushd ${SCRIPT_DIR}/build
        if [[ ! -z ${COMPUTE_BACKEND} ]]
        then
            REG_COMPUTE_BACKEND="-DCOMPUTE_BACKEND=${COMPUTE_BACKEND}"
        fi
        cmake .. -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" -DPV_ENVIRONMENT="${PV_ENVIRONMENT}" \
            ${REG_COMPUTE_BACKEND}
        make VERBOSE=1 -j${MAKE_JOB_COUNT} install
        check_command_exit_code $? "build failed"
        popd
    fi
}

run_tests() {
    if [[ ${ENABLE_TESTING} = "yes" ]]
    then
        local time_stamp=$(date "+%Y-%m-%d-%H-%M-%S")
        print_header "Run tests..."
        declare -a failed_test
        test_dirs="common"

        if [[ ${MODE} = "cpu" ]]
        then
            test_dirs="${test_dirs} cpu cpu/pt2pt cpu/scaleout"
        fi

        if [[ ${MODE} = "gpu" ]]
        then
            if [[  ${INFERENCE_TESTS} = "1" ]]
            then
                test_dirs="sycl/inference"
            else
                test_dirs="${test_dirs} sycl sycl/pt2pt sycl/scaleout"
            fi
        fi

        local tests_count=0
        local skipped_count=0

        for dir in ${test_dirs}
        do
            pushd ${SCRIPT_DIR}/${dir}
            echo "DEBUG: Processing a ${dir} directory"

            all_tests=$(ls *.sh 2>/dev/null)

            # filter scripts that contain both "-n" and "-ppn"
            selected_tests=$(grep -lE 'mpiexec .* -n [^ ]+.*-ppn [^ ]+' *.sh 2>/dev/null)

            eliminated_tests=""
            for test in ${all_tests}; do
                if ! echo "${selected_tests}" | grep -q "${test}"; then
                    eliminated_tests+="${test} "
                fi
            done

            echo "DEBUG: Selected test scripts:"
            echo "${selected_tests}"

            echo "DEBUG: Eliminated test scripts:"
            echo "${eliminated_tests}"

            for test in ${selected_tests}
            do
                local name=$(basename ${test} .sh)
                is_exclude=$(is_exclude_test ${name})
                if [[ ${is_exclude} -ne 0 ]]
                then
                    ((skipped_count++))
                    echo "${test} was excluded. Skip"
                    cat <<EOT >> "${SCRIPT_DIR}/${JUNIT_REPORT}.tmp"
    <testcase name="${test}">
      <skipped type ="TestSkipped">
        message="Test is in $(basename $EXCLUDE_LIST)"
      </skipped>
    </testcase>
EOT
                    cat <<EOT >> "${SCRIPT_DIR}/${JSON_REPORT}.tmp"
    {
        "name":"${test}",
        "result": "skipped"
    }
    ,
EOT
                    continue
                fi
                ((tests_count++))
                echo "${test}"
                local t1=$(date +%s.%N)
                ./${test}
                local err=$?
                local t2=$(date +%s.%N)
                local delta_t=$(echo "$t2 $t1" | awk '{print $1-$2}')
                if [[ $err -ne 0 ]]
                then
                    cat ./${name}.log
                    failed_test+=(${test})
                    cat <<EOT >> "${SCRIPT_DIR}/${JUNIT_REPORT}.tmp"
    <testcase name="$((tests_count-1)) - ${test}" time="${delta_t}">
        <failure type="TestFailed" message="$test has been failed:">
$( escape_xml "$( cat -v ./${name}.log )" )
        </failure>
    </testcase>
EOT
                    cat <<EOT >> "${SCRIPT_DIR}/${JSON_REPORT}.tmp"
    {
        "name":"${test}",
        "result": "failed"
    }
    ,
EOT
                else
                    cat <<EOT >> "${SCRIPT_DIR}/${JUNIT_REPORT}.tmp"
    <testcase name="$((tests_count-1)) - ${test}" time="${delta_t}"></testcase>
EOT
                    cat <<EOT >> "${SCRIPT_DIR}/${JSON_REPORT}.tmp"
    {
        "name":"${test}",
        "result": "passed"
    }
    ,
EOT
                fi
            done
            popd
        done

        cat <<EOT >> "${SCRIPT_DIR}/${JUNIT_REPORT}"
<testsuites>
  <testsuite failures="${#failed_test[@]}"
             skipped="${skipped_count}"
             tests="${tests_count}"
             date="${time_stamp}"
             name="reg_tests">
$(cat "${SCRIPT_DIR}/${JUNIT_REPORT}.tmp")
  </testsuite>
</testsuites>
EOT
        cat <<EOT >> "${SCRIPT_DIR}/${JSON_REPORT}"
[
$(sed '$d'  "${SCRIPT_DIR}/${JSON_REPORT}.tmp")
]
EOT

        if [[ ${#failed_test[@]} > 0 ]]
        then
            echo "${#failed_test[@]} FAILED TESTS: "
            for test in "${failed_test[@]}"
            do
                echo "    ${test}"
            done
        else
            echo "All tests passed"
            exit 0
        fi
    fi
}

set_default_values
parse_arguments $@
set_gen_env
define_compiler
build
run_tests
