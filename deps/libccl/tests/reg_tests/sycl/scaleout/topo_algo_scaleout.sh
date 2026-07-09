#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_YIELD=sched_yield

export CCL_ALLGATHER=topo
export CCL_ALLGATHERV=topo
export CCL_ALLREDUCE=topo
export CCL_ALLTOALLV=topo
export CCL_BCAST=topo
export CCL_REDUCE=topo
export CCL_REDUCE_SCATTER=topo
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

transports="ofi mpi"
proc_counts="4"
if [[ ${PLATFORM_HW_GPU} = "ats" ]]
then
    proc_counts="4"
fi

colls="allgather,allgatherv,allreduce,alltoallv,bcast,reduce,reduce_scatter"

bidir_algo_modes="0 1"
cache_modes="0 1"
copy_ops_modes="0 1"
single_list_modes="1"
imm_cmd_lists="0 1"

bench_options="-w 2 -i 2 -j off -b sycl -y 0,1,2,3,7,8,16,17,64,133,1077,16384,65539,131072,133073"
bench_options+=" -c all -d int8,int32,float16,float32,bfloat16"

for transport in ${transports}
do
    for proc_count in ${proc_counts}
    do
        for bidir_algo_mode in ${bidir_algo_modes}
        do
            for copy_ops_mode in ${copy_ops_modes}
            do
                for single_list_mode in ${single_list_modes}
                do
                    for cache_mode in ${cache_modes}
                    do
                        for imm_cmd_list in ${imm_cmd_lists}
                        do
                            cmd="CCL_ZE_SINGLE_LIST=${single_list_mode}"
                            cmd+=" CCL_ZE_BIDIR_ALGO=${bidir_algo_mode}"
                            cmd+=" CCL_KERNEL_1S_USE_COPY_OPS=${copy_ops_mode}"
                            cmd+=" CCL_ATL_TRANSPORT=${transport}"
                            cmd+=" CCL_ENABLE_SYCL_KERNELS=0"
                            cmd+=" SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=${imm_cmd_list}"
                            cmd+=" mpiexec -l -n ${proc_count} -ppn 2 ${SCRIPT_DIR}/benchmark"
                            cmd+=" ${bench_options} -l ${colls} -p ${cache_mode}"
                            cmd+=" > ${TEST_LOG} 2>&1"
                            run_cmd "${cmd}"
                            check_log ${TEST_LOG}
                        done
                    done
                done
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
