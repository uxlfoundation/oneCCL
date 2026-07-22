#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export CFESingleSliceDispatchCCSMode=1
export EngineInstancedSubDevices=1

export CCL_LOG_LEVEL=info
export CCL_ZE_DISABLE_FAMILY_CHECK=1

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=0
#,1,2,3,4,5

export FI_PROVIDER="$(get_default_prov)"

transports="mpi"
proc_counts="4"
colls="allreduce"
algos="ring topo"

bench_options="-w 2 -i 2 -c all -j off -b sycl -t 131072 $(get_default_bench_dtype)"

for transport in ${transports}
do
    for proc_count in ${proc_counts}
    do
        for algo in ${algos}
        do
            for coll in ${colls}
            do
                export CCL_ATL_TRANSPORT=${transport}
                export CCL_ALLREDUCE=${algo}
                mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark ${bench_options} -l ${coll} > ${TEST_LOG} 2>&1
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

rm ${TEST_LOG}
echo "Pass"
