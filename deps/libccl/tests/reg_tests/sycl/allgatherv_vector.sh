#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_ZE_AUTO_TUNE_PORTS=0

proc_counts="1 2 4"
inplace_modes="0 1"
transports="ofi mpi"
algos="direct topo multi_bcast flat"
monolithic_pipeline_modes="0 1"
allgatherv_read_modes="0 1"

for proc_count in ${proc_counts}
do
    for inplace_mode in ${inplace_modes}
    do
        for transport in ${transports}
        do
            for algo in ${algos}
            do
                for monolithic_pipeline_mode in ${monolithic_pipeline_modes}
                do
                    for allgatherv_read_mode in ${allgatherv_read_modes}
                    do
                        cmd="CCL_ATL_TRANSPORT=${transport}"
                        cmd+=" CCL_ALLGATHERV=${algo}"
                        if [[ "${algo}" == "topo" ]]
                        then
                            cmd+=" CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL=${monolithic_pipeline_mode}"
                            cmd+=" CCL_ALLGATHERV_TOPO_READ=${allgatherv_read_mode}"
                        fi
                        if [[ ( "${monolithic_pipeline_mode}" == "1" || "${allgatherv_read_mode}" == "1" ) && "${algo}" != "topo" ]]
                        then
                            continue
                        fi
                        cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/${BINFILE} ${inplace_mode} > ${TEST_LOG} 2>&1"
                        run_cmd "${cmd}"
                        check_log ${TEST_LOG}
                    done
                done
            done
        done
    done
done

rm ${BINFILE} ${TEST_LOG}
echo "Pass"
