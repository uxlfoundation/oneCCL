#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

proc_counts="4 8"
transports="ofi mpi"
ofi_provs="$(get_default_and_native_provs)"

bench_options="-c all -b host -t 2097152 $(get_default_bench_dtype)"

for proc_count in ${proc_counts}
do
    for transport in ${transports}
    do
        for ofi_prov in ${ofi_provs}
        do
            export FI_PROVIDER=${ofi_prov}
            export CCL_ATL_TRANSPORT=${transport}
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

echo "Pass"