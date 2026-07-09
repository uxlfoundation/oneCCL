#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"
PATH_TO_BINFILE="${SCRIPT_DIR}/${BINFILE}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "cpu"

# common
algos="direct offload"
transports="mpi ofi"

# case 1
peer_pairs="0,1 3,1 2,3 3,0"

# case with 4 ranks
for algo in ${algos}
do
    for peer_pair in ${peer_pairs}
    do
        cmd="CCL_ATL_TRANSPORT=mpi"
        cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
        cmd+=" mpiexec -l -n 4 -ppn 4"
        cmd+=" ${PATH_TO_BINFILE} --test_case 1"
        cmd+=" --peers ${peer_pair}"
        cmd+=" --iter 3 > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

# case with 2 ranks
for algo in ${algos}
do
    for transport in ${transports}
    do
        cmd="CCL_ATL_TRANSPORT=${transport}"
        cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
        cmd+=" mpiexec -l -n 2 ${PATH_TO_BINFILE} --test_case 1"
        cmd+=" --peers 1,0"
        cmd+=" --iter 2 > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

# case 2 with 3 ranks
peer_trios="0,1,2 1,2,0 2,0,1"
proc_counts_c2="3 6"
for algo in ${algos}
do
    for proc_count in ${proc_counts_c2}
    do
        for peer_trio in ${peer_trios}
        do
            cmd="CCL_ATL_TRANSPORT=mpi"
            cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
            cmd+=" mpiexec -l -n ${proc_count} ${PATH_TO_BINFILE} --test_case 2 --peers ${peer_trio}"
            cmd+=" > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done

# case 3: allreduce + pt2pt + allgatherv
for algo in ${algos}
do
    for transport in ${transports}
    do
        cmd="CCL_ATL_TRANSPORT=${transport}"
        cmd+=" CCL_ALLREDUCE=ring CCL_ALLGATHERV=ring"
        cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
        cmd+=" mpiexec -l -n 4 ${PATH_TO_BINFILE} --test_case 3"
        cmd+=" --iter 4 > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm ${TEST_LOG}
echo "Pass"
