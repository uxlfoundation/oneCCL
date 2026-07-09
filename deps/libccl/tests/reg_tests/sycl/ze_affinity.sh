#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export FI_PROVIDER="$(get_default_prov)"

transports="mpi ofi"
proc_counts="2"

regular_affinity_masks="0 0,1 0,2"
narrow_affinity_mask_1="0 1"
narrow_affinity_mask_2="0 2"

bench_options="-w 1 -i 2 -c all -l all -b sycl -y 1024 $(get_default_bench_dtype)"

run_case() {
    transport=$1
    proc_count=$2
    affinity_env=$3

    # part of scenarios rely on IMPI gtool so use mpiexec from IMPI package
    
    cmd="CCL_ATL_TRANSPORT=${transport}"
    cmd+=" ${affinity_env}"
    cmd+=" ${I_MPI_ROOT}/bin/mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
    cmd+=" ${bench_options} > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"
    check_log ${TEST_LOG}
}

for transport in ${transports}
do
    for proc_count in ${proc_counts}
    do
        for regular_affinity_mask in ${regular_affinity_masks}
        do
            run_case ${transport} ${proc_count} "ZE_AFFINITY_MASK=${regular_affinity_mask}"
        done

        if [[ ${PLATFORM_HW_GPU} = "ats" ]]
        then
            affinity_env=$(create_ze_affinity_env "${narrow_affinity_mask_1}")
            run_case ${transport} ${proc_count} "${affinity_env}"

            affinity_env=$(create_ze_affinity_env "${narrow_affinity_mask_2}")
            run_case ${transport} ${proc_count} "${affinity_env}"
        fi
    done
done

rm ${TEST_LOG}
echo "Pass"
