#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

worker_counts="1 4 8"
proc_counts="2"
transports="ofi mpi"
# MLSL-1542: issue test system
ofi_provs="$(get_default_and_native_provs)"

bench_options="-c all -b host -t 2097152 $(get_default_bench_dtype)"

if [[ $(hostname) == *"nnlmpiclx03"* ]]
then
    export FI_VERBS_DEVICE_NAME=hfi
else
    export FI_VERBS_DEVICE_NAME=mlx
fi
export I_MPI_PIN_PROCESSOR_LIST=1,2

for worker_count in ${worker_counts}
do
    for proc_count in ${proc_counts}
    do
        for transport in ${transports}
        do
            for ofi_prov in ${ofi_provs}
            do
                export FI_PROVIDER=${ofi_prov}
                export CCL_ATL_TRANSPORT=${transport}
                export CCL_WORKER_COUNT=${worker_count}
                export CCL_WORKER_AFFINITY="3-$((3-1+${worker_count}*${proc_count}))"
                mpiexec -l -n ${proc_count} -ppn 1 ${SCRIPT_DIR}/benchmark ${bench_options} > ${TEST_LOG} 2>&1
                rc=$?
                if [ ${rc} -ne 0 ]
                then
                    echo "Fail"
                    exit 1
                fi
                check_log ${TEST_LOG}
            done
        done
    done
done

echo "Pass"
