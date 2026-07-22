#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"
PATH_TO_BINFILE="${SCRIPT_DIR}/${BINFILE}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

export CCL_ZE_AUTO_TUNE_PORTS=0

# common
algos="direct topo offload"
transports="mpi ofi"

# case 1
cache_modes="0"
peer_pairs="0,1 3,1 2,3 3,0"

serialize_modes="0 1"
queue_modes="0 1"
wait_modes="0 1"
read_modes="0 1"
sync_barriers="0 1"

# case with 4 ranks
for algo in ${algos}
do
    for wait_mode in ${wait_modes}
    do
        if [ ${wait_mode} == "0" ] && [ ${algo} == "topo" ];
        then
            # topo supports only blocking mode
            continue
        fi
        for cache_mode in ${cache_modes}
        do
            for queue_mode in ${queue_modes}
            do
                for peer_pair in ${peer_pairs}
                do
                    for sync_barrier in ${sync_barriers}
                    do
                        cmd="CCL_ATL_TRANSPORT=mpi"
                        cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
                        cmd+=" CCL_BARRIER_SYNC=${sync_barrier}"
                        cmd+=" mpiexec -l -n 4"
                        cmd+=" ${PATH_TO_BINFILE} --test_case 1 --queue ${queue_mode}"
                        cmd+=" --cache ${cache_mode} --wait ${wait_mode} --peers ${peer_pair}"
                        cmd+=" --iter 3 > ${TEST_LOG} 2>&1"
                        run_cmd "${cmd}"
                        check_log ${TEST_LOG}
                    done
                done
            done
        done
    done
done

# case with 2 ranks
for algo in ${algos}
do
    for transport in ${transports}
    do
        for cache_mode in ${cache_modes}
        do
            for queue_mode in ${queue_modes}
            do
                cmd="CCL_ATL_TRANSPORT=${transport}"
                cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
                cmd+=" mpiexec -l -n 2 ${PATH_TO_BINFILE} --test_case 1 --queue ${queue_mode}"
                cmd+=" --cache ${cache_mode} --wait ${wait_mode} --peers 1,0"
                cmd+=" --iter 2 > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_log ${TEST_LOG}
            done
        done
    done
done

# case with 3 ranks
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

# case with serialize
for read_mode in ${read_modes}
do
    for algo in ${algos}
    do
        cmd="CCL_ATL_TRANSPORT=mpi"
        cmd+=" CCL_ZE_AUTO_TUNE_PORTS=${read_mode} CCL_SEND=${algo} CCL_RECV=${algo}"
        cmd+=" mpiexec -l -n ${proc_count} ${PATH_TO_BINFILE} --test_case 1 --serialize 1 --wait 1"
        cmd+=" > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

# case with pidfd
cmd="CCL_ATL_TRANSPORT=mpi"
cmd+=" CCL_ZE_IPC_EXCHANGE=pidfd CCL_SEND=${algo} CCL_RECV=${algo}"
cmd+=" mpiexec -l -n ${proc_count} ${PATH_TO_BINFILE} --test_case 1 --wait 1"
cmd+=" > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_log ${TEST_LOG}

# case with read/write protocols for topo algo
ze_affinity_mask_modes="0,1 0,2"

for read_mode in ${read_modes}; do
    for ze_affinity_mask_mode in ${ze_affinity_mask_modes}; do

        cmd="CCL_ATL_TRANSPORT=mpi"
        if [ "$read_mode" = "0" ] && [ "$ze_affinity_mask_mode" = "0,1" ]; then
            cmd+=" CCL_LOG_LEVEL=debug"
        fi
        cmd+=" ZE_AFFINITY_MASK=${ze_affinity_mask_mode}"
        cmd+=" CCL_SEND=topo CCL_RECV=topo CCL_ZE_PT2PT_READ=${read_mode}"
        cmd+=" mpiexec -l -n 2 ${PATH_TO_BINFILE} --test_case 1 --wait 1"
        cmd+=" > ${TEST_LOG} 2>&1"
        run_cmd "${cmd}"

        if [ "$read_mode" = "0" ] && [ "$ze_affinity_mask_mode" = "0,1" ]; then
            if ! grep -q "pt2pt: force read algo for within card execution case" ${TEST_LOG};
            then
                echo "FAILED: The required string was not found in the log" > ${TEST_LOG} 2>&1
                exit 1
            fi
        fi
        check_log ${TEST_LOG}
    done
done

# case 4: allreduce + pt2pt + allgatherv
for algo in ${algos}
do
    for transport in ${transports}
    do
        for cache_mode in ${cache_modes}
        do
            for queue_mode in ${queue_modes}
            do
                cmd="CCL_ATL_TRANSPORT=${transport}"
                cmd+=" CCL_ALLREDUCE=ring CCL_ALLGATHERV=ring"
                cmd+=" CCL_SEND=${algo} CCL_RECV=${algo}"
                cmd+=" mpiexec -l -n 4 ${PATH_TO_BINFILE} --test_case 4 --queue ${queue_mode}"
                cmd+=" --cache ${cache_mode} --iter 4 > ${TEST_LOG} 2>&1"
                run_cmd "${cmd}"
                check_log ${TEST_LOG}
            done
        done
    done
done

# case 5: group api without ccl init
cmd="mpiexec -l -n 2 ${PATH_TO_BINFILE} --test_case 5 > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_log ${TEST_LOG}

# case 6: scatter case
cmd="mpiexec -l -n 4 ${PATH_TO_BINFILE} --test_case 6 --count 33554432 --wait 1 > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_log ${TEST_LOG}

rm ${TEST_LOG}
echo "Pass"
