#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

proc_counts="4"
worker_counts="1 2"
transports="ofi mpi"
ofi_provs="$(get_default_prov)"
pmi_types="simple_pmi internal_pmi"
for pmi_type in ${pmi_types}
do
    for proc_count in ${proc_counts}
    do
        for worker_count in ${worker_counts}
        do
            for transport in ${transports}
            do
                for ofi_prov in ${ofi_provs}
                do
                    cmd="CCL_WORKER_COUNT=${worker_count}"
                    cmd+=" CCL_ATL_TRANSPORT=${transport}"
                    cmd+=" FI_PROVIDER=${ofi_prov}"
                    cmd+=" mpiexec -l -n ${proc_count} -ppn 2"
                    cmd+=" ${SCRIPT_DIR}/allreduce ${pmi_type} > ${TEST_LOG} 2>&1"
                    run_cmd "${cmd}"
                    check_log ${TEST_LOG}
                done
            done
        done
    done
done

rm allreduce ${TEST_LOG}

echo "Pass"
