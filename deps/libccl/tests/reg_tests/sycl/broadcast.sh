#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export I_MPI_JOB_TIMEOUT=360

transports="ofi mpi"
proc_counts="2 4"
sched_cache_modes="0 1"
backends="sycl host"
algos="direct ring double_tree naive topo"
inplace_modes="0 1"

bench_options="-w 2 -i 2 -j off -y 0,1,2,3,7,8,16,17,64,133,1077,16384,65539,131072,133073"
bench_options+=" -c all -d int8,int32,float16,float32,bfloat16 -l broadcast"

for algo in ${algos}
do
    for sched_cache_mode in ${sched_cache_modes}
    do
        for backend in ${backends}
        do
            for transport in ${transports}
            do
                for proc_count in ${proc_counts}
                do
                    for inplace_mode in ${inplace_modes}
                    do
                        cmd="CCL_BROADCAST=${algo}"
                        cmd+=" CCL_ATL_TRANSPORT=${transport}"
                        cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
                        cmd+=" ${bench_options} -p ${sched_cache_mode} -b ${backend} -q ${inplace_mode} > ${TEST_LOG} 2>&1"
                        run_cmd "${cmd}"
                        check_log ${TEST_LOG}
                    done
                done
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
