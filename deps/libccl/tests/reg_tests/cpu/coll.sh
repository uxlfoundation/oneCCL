#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

tests="allgatherv alltoallv"
tests+=" broadcast custom_allreduce external_kvs"
tests+=" priority_allreduce reduce reduce_scatter"

proc_counts="4"
worker_counts="1 2"
transports="ofi mpi"
ofi_provs="$(get_default_prov)"
in_places="0 1"

for test in ${tests}
do
    for proc_count in ${proc_counts}
    do
        for worker_count in ${worker_counts}
        do
            for transport in ${transports}
            do
                for ofi_prov in ${ofi_provs}
                do
                    for in_place in ${in_places}
                    do
                        if [[ ${transport} = "mpi" ]]
                        then
                            if [[ ${test} = "custom_allreduce" ]]
                            then
                                continue
                            fi
                        fi

                        if [[ ${test} = "external_kvs" ]] &&
                            [[ ${worker_count} = "2" ]]
                        then
                            continue
                        fi

                        if [[ ${test} != "allgatherv" ]] && [[ ${test} != "allgather" ]] &&
                            [[ ${test} != "reduce_scatter" ]] && [[ ${in_place} = "1" ]]
                        then
                            continue
                        fi

                        cmd="CCL_WORKER_COUNT=${worker_count}"
                        cmd+=" CCL_ATL_TRANSPORT=${transport}"
                        cmd+=" FI_PROVIDER=${ofi_prov}"
                        cmd+=" mpiexec -l -n ${proc_count} -ppn 2"
                        cmd+=" ${SCRIPT_DIR}/${test} ${in_place} > ${TEST_LOG} 2>&1"
                        run_cmd "${cmd}"
                        check_log ${TEST_LOG}
                    done
                done
            done
        done
    done
done

for test in ${tests}
do
    rm ${test}
done
rm ${TEST_LOG}

echo "Pass"
