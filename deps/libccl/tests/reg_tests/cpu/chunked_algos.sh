#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

transports="mpi ofi"
proc_counts="2 8"
worker_counts="1 2"
inplace_modes="0 1"
chunk_modes="0 1"
allreduce_algos="ring 2d"
alltoallv_algos="scatter"

bench_options="-w 1 -i 4 -j off -c all -b host -y 17,1024,65536,1048576 $(get_default_bench_dtype)"

for transport in ${transports}
do
    for proc_count in ${proc_counts}
    do
        for worker_count in ${worker_counts}
        do
            for inplace_mode in ${inplace_modes}
            do
                for chunk_mode in ${chunk_modes}
                do
                    base_env="CCL_ATL_TRANSPORT=${transport}"
                    base_env+=" CCL_WORKER_COUNT=${worker_count}"
                    base_env+=" CCL_LOG_LEVEL=info"

                    for allreduce_algo in ${allreduce_algos}
                    do
                        algo_env="CCL_ALLREDUCE=${allreduce_algo}"
                        if [[ ${chunk_mode} = "1" ]]
                        then
                            if [[ ${allreduce_algo} = "ring" ]]
                            then
                                algo_env+=" CCL_RS_CHUNK_COUNT=2"
                            elif [[ ${allreduce_algo} = "2d" ]]
                            then
                                algo_env+=" CCL_ALLREDUCE_2D_CHUNK_COUNT=2"
                            fi
                        fi

                        cmd="${base_env} ${algo_env}"
                        cmd+=" mpiexec -l -n ${proc_count} -ppn 1 ${SCRIPT_DIR}/benchmark"
                        cmd+=" ${bench_options} -q ${inplace_mode} -l allreduce"
                        cmd+=" > ${TEST_LOG} 2>&1"
                        run_cmd "${cmd}"
                        check_log ${TEST_LOG}
                    done

                    for alltoallv_algo in ${alltoallv_algos}
                    do
                        algo_env="CCL_ALLTOALLV=${alltoallv_algo}"
                        if [[ ${chunk_mode} = "1" ]]
                        then
                            if [[ ${alltoallv_algo} = "scatter" ]]
                            then
                                algo_env+=" CCL_ALLTOALL_SCATTER_MAX_OPS=2"
                                algo_env+=" CCL_CHUNK_COUNT=${worker_count}"
                            fi
                        fi

                        cmd="${base_env} ${algo_env}"
                        cmd+=" mpiexec -l -n ${proc_count} -ppn 1 ${SCRIPT_DIR}/benchmark"
                        cmd+=" ${bench_options} -q ${inplace_mode} -l alltoallv"
                        cmd+=" > ${TEST_LOG} 2>&1"
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
