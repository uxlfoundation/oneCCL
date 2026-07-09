#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

transports="ofi mpi"
topo_colors="ze fixed env"
pvc_random_device_list="3.0 1.0 3.1 4.1 0.0 4.0 2.1 2.0 1.1 5.1 0.1 5.0"
bench_ops="-w 0 -i 1 -c all -l allgatherv,allreduce,reduce -b sycl -y 1024"

check_string() {
    input_str=$1
    if [ -z "$input_str" ]
    then
        echo "Fail: string is empty" >> ${TEST_LOG} 2>&1
    fi
}

check_color() {
    current_str=$1
    pattern=$2
    if [ "$current_str" != "$pattern" ]
    then
        echo "Fail: $current_str, $pattern" >> ${TEST_LOG} 2>&1
    fi
}

parse_colors() {
    expected_intra_str=$1
    expected_inter_str=$2

    intra_str=""
    inter_str=""

    while IFS='' read -r line
    do
        if [[ $line =~ intra_card_colors:\ ([0-9\ ]+) ]]
        then
            intra_str="${BASH_REMATCH[1]}"
            break
        fi
    done < "${TEST_LOG}"

    while IFS='' read -r line
    do
        if [[ $line =~ inter_card_colors:\ ([0-9\ ]+) ]]
        then
            inter_str="${BASH_REMATCH[1]}"
            break
        fi
    done < "${TEST_LOG}"

    check_string "$intra_str"
    check_string "$inter_str"

    check_color "$intra_str" "$expected_intra_str"
    check_color "$inter_str" "$expected_inter_str"
}

run_case() {
    n=$1
    ppn=$2
    intra_colors=$3
    inter_colors=$4
    topo_map=$5
    env_var=${6:-}
    extra_bench_ops=${7:-}

    original_inter_colors=$inter_colors

    for transport in ${transports}
    do
        for topo_color in ${topo_colors}
        do
            cmd="CCL_ATL_TRANSPORT=${transport}"
            cmd+=" CCL_ALLGATHERV=topo"
            cmd+=" CCL_ALLREDUCE=topo"
            cmd+=" CCL_REDUCE=topo"
            cmd+=" CCL_LOG_LEVEL=info"

            if [ "${topo_color}" == "env" ]  && [ ! -z "$topo_map" ]
            then
                cmd+=" CCL_TOPO_COLOR=${topo_map}"
            else
                cmd+=" CCL_TOPO_COLOR=${topo_color}"
            fi

            cmd+=" ${env_var}"
            cmd+=" mpiexec -l -n $n ${SCRIPT_DIR}/benchmark"
            cmd+=" ${bench_ops} ${extra_bench_ops} > ${TEST_LOG} 2>&1"

            run_cmd "${cmd}"

            if [ "${transport}" == "mpi" ] && [ "${topo_color}" == "ze" ] && [ $n -eq 4 ] && [ $ppn -eq 4 ]
            then
                #When ze API is used to determine inter_colors and that API
                #functions well, then a port connection inconsistency is expected.
                #For that particular case parse_colors function should take
                #another value for the second parameter. Potentially this value
                #might differ for various systems. Should we detect other
                #inconsistency patterns, this test must be revised. Originally
                #the test was supposed to check topo_manager color assignment
                #ability for the regular port connection case that could be
                #observed when OFI transport is enabled along with ze API.
                #The reason for that is the port filter being not precise enough
                #when OFI is used.
                cnt=$( cat "${TEST_LOG}" | grep "no ports detected" | wc -l )

                #it seems like GPUs on orpre-smc and anbhscsr compute nodes don't have
                #that inconsistency problem. In that case the check should
                #use the regular pattern
                smc_host=$( hostname | grep -E "ss[1-2]|^anpvcx" | wc -l )
                if [ $cnt -eq 0 ] && [ $smc_host -eq 0 ]
                then
                    inter_colors="0 1 1 0"
                fi
            else
                inter_colors=$original_inter_colors
            fi

            parse_colors "${intra_colors} " "${inter_colors} "
            check_log ${TEST_LOG}
            rm ${TEST_LOG}
        done
    done
}

if [[ ${PLATFORM_HW_GPU} = "ats" || ${PLATFORM_HW_GPU} = "gen" || ${PLATFORM_HW_GPU} = "pvc" ]]
then
    # [ATS/GEN]: case 1: 1 node, 2 ranks
    run_case 2 2 "0 0" "0 1" "\"card:{0,1};plane:{0},{1}\""

    # [ATS/GEN]: case 2: 1 node, 4 ranks
    run_case 4 4 "0 0 1 1" "0 1 0 1" "\"card:{0,1},{2,3};plane:{0,2},{1,3}\""

    if [[ ${PLATFORM_HW_GPU} = "ats" ]]
    then
        # [ATS/GEN]: case 3: 1 node, 8 ranks
        run_case 8 8 "0 0 1 1 2 2 3 3" "0 1 0 1 0 1 0 1" \
                     "\"card:{0,1},{2,3},{4,5},{6,7};plane:{0,2,4,6},{1,3,5,7}"\"
    fi

elif [[ ${PLATFORM_HW_GPU} = "pvc_aurora" ]]
then
    # [PVC] case 1: 1 node, 6 ranks, 0-th tiles only
    affinity_env="ZE_AFFINITY_MASK=0,2,4,6,8,10"
    run_case 6 6 "0 1 2 3 4 5" "0 0 0 0 0 0" \
                 "\"card:{0},{1},{2},{3},{4},{5};plane:{0,1,2,3,4,5}\"" "${affinity_env}"

    # [PVC] case 2: 1 node, 6 ranks, root devices only
    run_case 6 6 "0 1 2 3 4 5" "0 0 0 0 0 0" \
                 "\"card:{0},{1},{2},{3},{4},{5};plane:{0,1,2,3,4,5}\"" "" "-g 1"

    # [PVC] case 3: 1 node, 6 ranks
    affinity_env="ZE_AFFINITY_MASK=0,1,2,3,4,5"
    run_case 6 6 "0 0 1 1 2 2" "0 1 0 1 0 1" \
                 "\"card:{0,1},{2,3},{4,5};plane:{0,2,4},{1,3,5}\"" "${affinity_env}"

    # [PVC] case 4: 1 node, 12 ranks
    run_case 12 12 "0 0 1 1 2 2 3 3 4 4 5 5" "0 1 0 1 0 1 0 1 0 1 0 1" \
                   "\"card:{0,1},{2,3},{4,5},{6,7},{8,9},{10,11};plane:{0,2,4,6,8,10},{1,3,5,7,9,11}\""

    # [PVC] case 5: 1 node, 12 ranks, affinity mask is pseudo-random
    affinity_env=$(create_ze_affinity_env "${pvc_random_device_list}")
    run_case 12 12 "0 0 1 1 2 2 3 3 4 4 5 5" "0 1 0 1 0 1 0 1 0 1 0 1" \
                   "\"card:{0,1},{2,3},{4,5},{6,7},{8,9},{10,11};plane:{0,2,4,6,8,10},{1,3,5,7,9,11}\"" \
                   "${affinity_env}"

fi

echo "Pass"
