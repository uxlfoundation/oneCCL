#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export I_MPI_JOB_TIMEOUT=120

worker_counts="1"
send_proxy_modes="none regular"
hmem_modes="0 1"
algos="topo rabenseifner"
proc_counts="2 4"

bench_options="-l allreduce,reduce -w 1 -i 2 -j off -c all -b sycl -t 131072 $(get_default_bench_dtype)"

artefact_dir="/home/sys_ctlab/prog"
ofi_bench_path="${artefact_dir}/dmabuf-rdma-tests/fi-rdmabw-xe"
ofi_lib_path="${artefact_dir}/libfabric_dmabuf_peer_mem/lib"

dmabuf_peer_mem_env="FI_HOOK=dmabuf_peer_mem"
dmabuf_peer_mem_env+=" FI_PROVIDER=verbs"
dmabuf_peer_mem_env+=" FI_VERBS_INLINE_SIZE=0"
dmabuf_peer_mem_env+=" MLX5_SCATTER_TO_CQE=0"
dmabuf_peer_mem_env+=" LD_LIBRARY_PATH=${ofi_lib_path}:${LD_LIBRARY_PATH}"
dmabuf_peer_mem_env+=" FI_PROVIDER_PATH=${ofi_lib_path}/libfabric"
dmabuf_peer_mem_env+=" EnableImplicitScaling=0"
dmabuf_peer_mem_env+=" NEOReadDebugKeys=1"

check_dmabuf_peer_mem() {
    if [ -z "$(lsmod | grep dmabuf_peer_mem)" ]
    then
        echo "Fail: dmabuf_peer_mem driver is not found" >> ${TEST_LOG} 2>&1
        exit 1
    fi

    env="${dmabuf_peer_mem_env}"
    ofi_bench_options="-m device -t send -n 4"
    device_pairs="0_0 0_1 1_1 0.0_0.1 0.0_1.0 0.0_1.1"

    for device_pair in ${device_pairs}
    do
        server_device=$(echo $device_pair | cut -d_ -f1)
        client_device=$(echo $device_pair | cut -d_ -f2)
        cmd="${env} ${ofi_bench_path} ${ofi_bench_options} -d ${server_device} >> ${TEST_LOG} 2>&1 &"
        cmd+=" sleep 1 && ${env} ${ofi_bench_path} ${ofi_bench_options} -d ${client_device} localhost >> ${TEST_LOG} 2>&1"

        run_cmd "${cmd}"
        check_log ${TEST_LOG} "4194304"
        rm ${TEST_LOG}

        # to avoid "connection refused" issue
        sleep 15
    done
}

affinity=""

if [[ ${PLATFORM_HW_GPU} == "ats" || ${PLATFORM_HW_GPU} == "pvc" ]]
then
    libfabrics="default dmabuf_peer_mem"
    provs="verbs"
    check_dmabuf_peer_mem
else
    libfabrics="default"
    provs="cxi"
    affinity="ZE_AFFINITY_MASK=0,1"
fi

for worker_count in ${worker_counts}
do
    for send_proxy_mode in ${send_proxy_modes}
    do
        for hmem_mode in ${hmem_modes}
        do        
            for algo in ${algos}
            do   
                for proc_count in ${proc_counts}
                do
                    for libfabric in ${libfabrics}
                    do
                        for prov in ${provs}
                        do
                            cmd="CCL_WORKER_COUNT=${worker_count}"
                            cmd+=" CCL_ATL_TRANSPORT=ofi"
                            cmd+=" CCL_ATL_SEND_PROXY=${send_proxy_mode}"
                            cmd+=" CCL_ATL_HMEM=${hmem_mode}"
                            cmd+=" CCL_USE_HMEM=1"
                            cmd+=" CCL_LOG_LEVEL=info"
                            cmd+=" CCL_ALLREDUCE=${algo}"
                            cmd+=" FI_PROVIDER=${prov}"
                            cmd+=" ${affinity}"

                            if [[ ${libfabric} == "dmabuf_peer_mem" ]]
                            then
                                cmd+=" ${dmabuf_peer_mem_env}"
                            fi

                            cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
                            cmd+=" ${bench_options} > ${TEST_LOG} 2>&1"

                            run_cmd "${cmd}"
                            check_log ${TEST_LOG}
                            check_hmem_log ${TEST_LOG} ${hmem_mode}
                        done
                    done
                done
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
