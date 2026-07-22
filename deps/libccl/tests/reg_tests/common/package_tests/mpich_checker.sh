#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

LOG=${SCRIPT_DIR}/${TEST_LOG}

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_ATL_TRANSPORT=mpi

# 1. check mpi lib type
mpiexec -l -n 2 -ppn 1 ${SCRIPT_DIR}/benchmark > ${LOG} 2>&1
rc=$?
if [ ${rc} -ne 0 ]
then
    echo "Fail"
    exit 1
fi

lib_type=$(cat ${LOG} | grep -m 1 "mpi_lib_attr.type" | awk '{print $NF}')
if [[ "${lib_type}" != "mpich" ]]
then
    echo "Fail: lib_type ${lib_type} != mpich"
    exit 1
fi

# 2. check multi-nic

export CCL_LOG_LEVEL=debug

mnic_types="local global"
mnic_counts="1 2"
bench_options="-w 0 -i 1 -y 64 -b host $(get_default_bench_dtype)"

for mnic_type in ${mnic_types}
do
    for mnic_count in ${mnic_counts}
    do
        export CCL_MNIC=${mnic_type}
        export CCL_MNIC_COUNT=${mnic_count}

        mpiexec -l -n 2 -ppn 1 ${SCRIPT_DIR}/benchmark ${bench_options} > ${LOG} 2>&1
        rc=$?
        if [ ${rc} -ne 0 ]
        then
            echo "Fail"
            exit 1
        fi

        check_log ${LOG}

        in_mnic_type=$(cat ${LOG} | grep "in:" | grep -m 1 -o "mnic_type: [[:alpha:]]*" | awk '{print $2}')
        in_mnic_count=$(cat ${LOG} | grep "in:" | grep -m 1 -o "mnic_count: [[:digit:]]*" | awk '{print $2}')
        out_mnic_type=$(cat ${LOG} | grep "out:" | grep -m 1 -o "mnic_type: [[:alpha:]]*" | awk '{print $2}')
        out_mnic_count=$(cat ${LOG} | grep "out:" | grep -m 1 -o "mnic_count: [[:digit:]]*" | awk '{print $2}')
        mpich_nic_count=$(cat ${LOG} | grep -m 1 -o "num_nics: [[:digit:]]*" | awk '{print $2}')

        if [[ ${in_mnic_type} != ${mnic_type} ]]
        then
            echo "Fail: in_mnic_type(${in_mnic_type}) != mnic_type(${mnic_type})"
            exit 1
        fi

        if [[ ${in_mnic_count} != ${mnic_count} ]]
        then
            echo "Fail: in_mnic_count(${in_mnic_count}) != mnic_count(${mnic_count})"
            exit 1
        fi

        if [[ ${out_mnic_type} != ${mnic_type} ]]
        then
            echo "Fail: out_mnic_type(${out_mnic_type}) != mnic_type(${mnic_type})"
            exit 1
        fi

        if [[ ${out_mnic_count} > ${mnic_count} ]]
        then
            echo "Fail: out_mnic_count(${out_mnic_count}) != mnic_count(${mnic_count})"
            exit 1
        fi

        if [[ ${out_mnic_count} > ${mpich_nic_count} ]]
        then
            echo "Fail: out_mnic_count(${out_mnic_count}) != mpich_nic_count(${mpich_nic_count})"
            exit 1
        fi
    done
done

echo "Pass"
