#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

export CCL_ATL_TRANSPORT=ofi
export CCL_ALLREDUCE=nreduce

proc_counts="2 8"
inplace_modes="0 1"
buffering_modes="0 1"
segment_sizes="16384 262144"
bench_options="-w 1 -i 4 -c all -b host -t 2097152 $(get_default_bench_dtype)"

for proc_count in ${proc_counts}
do
    for inplace_mode in ${inplace_modes}
    do
        for buffering_mode in ${buffering_modes}
        do
            for segment_size in ${segment_sizes}
            do
                export CCL_ALLREDUCE_NREDUCE_BUFFERING=${buffering_mode}
                export CCL_ALLREDUCE_NREDUCE_SEGMENT_SIZE=${segment_size}
                mpiexec -l -n ${proc_count} -ppn 1 ${SCRIPT_DIR}/benchmark ${bench_options} -q ${inplace_mode} > ${TEST_LOG} 2>&1
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
