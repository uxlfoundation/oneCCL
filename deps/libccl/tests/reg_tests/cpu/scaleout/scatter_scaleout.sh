#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

# Scatter has only "naive" and "topo" main algos; scaleout table falls back to naive.
# Sweep both transports and both scaleout algo choices.
transports="ofi mpi"
proc_counts="4"
scatter_algos="naive"
scatter_scaleout_algos="naive"

bench_options="-w 1 -i 4 -l scatter -c all -b host -t 1048576 $(get_default_bench_dtype)"

for transport in ${transports}
do
    for proc_count in ${proc_counts}
    do
        for algo in ${scatter_algos}
        do
            for scaleout_algo in ${scatter_scaleout_algos}
            do
                cmd="CCL_ATL_TRANSPORT=${transport}"
                cmd+=" CCL_SCATTER=${algo}"
                cmd+=" CCL_SCATTER_SCALEOUT=${scaleout_algo}"
                cmd+=" mpiexec -l -n ${proc_count} -ppn 2"
                cmd+=" ${SCRIPT_DIR}/benchmark ${bench_options} > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_log ${TEST_LOG}
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
