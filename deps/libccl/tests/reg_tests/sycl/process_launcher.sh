#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

parse_str() {
    expected_str=$1
    match_count=0
    while IFS='' read -r line;
    do
        if [[ "$line" == *"$expected_str"* ]];
        then
            match_count=$((match_count+1))
            break
        fi
    done < ${TEST_LOG}
    if [[ "$match_count" == "0" ]];
    then
        echo "Fail: couldn't find expected value" >> ${TEST_LOG} 2>&1
    else
        echo "Found matches: ${match_count}" >> ${TEST_LOG} 2>&1
    fi
}

if [[ ${PLATFORM_HW_GPU} = "ats" ]] || [[ ${PLATFORM_HW_GPU} = "dg" ]] || [[ ${PLATFORM_HW_GPU} = "pvc" ]]
then
    proc_count="2"
elif [[ ${PLATFORM_HW_GPU} = "gen" ]]
then
    proc_count="4"
else
    echo "Failed: unexpected platform: ${PLATFORM_HW_GPU}" >> ${TEST_LOG} 2>&1
    check_log ${TEST_LOG}
fi

proc_launchers="hydra none"
test_binaries="multi_comms benchmark"

test_options=""
multi_comms_options="-c direct -o direct"
bench_options="-b sycl -w 1 -i 2 -c all -l allreduce -y 131072"
none_mode_envs="set unset"
ccl_worker_count=2

for binary in ${test_binaries}
do
    for proc_launcher in ${proc_launchers}
    do
        for none_mode_env in ${none_mode_envs}
        do
            if [[ ${proc_launcher} == "hydra" ]] && [[ ${none_mode_env} == "set" || ${none_mode_env} == "unset" ]]
            then
                continue
            fi

            cmd="CCL_ATL_TRANSPORT=mpi"
            cmd+=" CCL_LOG_LEVEL=info"
            cmd+=" CCL_WORKER_COUNT=${ccl_worker_count}"
            cmd+=" CCL_WORKER_AFFINITY=1-$((${proc_count}*${ccl_worker_count}))"
            if [ "$proc_launcher" == "none" ];
            then
                if [[ "$binary" == "multi_comms" ]] && [[ "$none_mode_env" == "unset" ]]
                then
                    continue
                fi
                if [ "$none_mode_env" == "set" ];
                then
                    cmd+=" I_MPI_GTOOL=\"env CCL_LOCAL_RANK=0:0;env CCL_LOCAL_RANK=1:1;env CCL_LOCAL_RANK=0:2;env CCL_LOCAL_RANK=1:3\""
                    cmd+=" CCL_LOCAL_SIZE=2"
                fi
            fi
            cmd+=" CCL_PROCESS_LAUNCHER=${proc_launcher}"
            cmd+=" mpiexec -n ${proc_count} -ppn 2 ${SCRIPT_DIR}/${binary}"

            if [ "multi_comms" == ${binary} ];
            then
                test_options="${multi_comms_options}"
            elif [ "benchmark" == ${binary} ];
            then
                test_options="${bench_options}"
            else
                echo "Failed: unexpected binary" >> ${TEST_LOG} 2>&1
                check_log ${TEST_LOG}
            fi
            cmd+=" ${test_options} > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"

            parse_str "local process [0:2]: worker: 0, cpu: 1"
            parse_str "local process [0:2]: worker: 1, cpu: 2"
            parse_str "local process [1:2]: worker: 0, cpu: 3"
            parse_str "local process [1:2]: worker: 1, cpu: 4"

            if [[ "$proc_launcher" == "none" ]] && [[ "$none_mode_env" == "unset" ]]
            then
                parse_str "local_proc_idx: 0, local_proc_count: 2 are set by ATL transport"
                parse_str "local_proc_idx: 1, local_proc_count: 2 are set by ATL transport"
            else
                parse_str ", local_proc_idx: 0, local_proc_count: 2"
                parse_str ", local_proc_idx: 1, local_proc_count: 2"
            fi

            check_log ${TEST_LOG}
            rm ${TEST_LOG}
        done
    done
done

echo "Pass"
