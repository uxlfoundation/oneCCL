#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

proc_counts="2 4"
inplace_modes="0 1"
transports="ofi mpi"
algos="topo direct naive scatter"
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
                        cmd+=" CCL_ALLTOALLV=${algo}"
                        if [[ "${algo}" == "topo" ]]
                        then
                            # TODO check if the names below make sense and if all algos are covered
                            cmd+=" CCL_ALLTOALLV_MONOLITHIC_KERNEL=${monolithic_pipeline_mode}"
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

rc=$?
if [[ ${rc} -ne 0 ]]
then
    echo "Fail"
    echo "DEBUG: ${SCRIPT_DIR}/${BINFILE} fails for mpiexec"
    exit 1
fi

check_log ${TEST_LOG}

rm -f ${BINFILE} ${TEST_LOG}
echo "Pass"
