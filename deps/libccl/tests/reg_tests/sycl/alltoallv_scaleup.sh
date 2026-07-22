#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_ALLTOALLV=topo
export CCL_ZE_AUTO_TUNE_PORTS=0

cache_modes="0 1"
monolithic_modes="0 1"
read_mono_modes="0 1"
read_topo_modes="0 1"
proc_counts="2 4"

bench_options="-w 2 -i 2 -j off -c all -b sycl -y 17,1024,131072 $(get_default_bench_dtype)"

for read_mono_mode in ${read_mono_modes}
do
    for read_topo_mode in ${read_topo_modes}
    do
        for monolithic_mode in ${monolithic_modes}
        do
            for proc_count in ${proc_counts}
            do
                for cache_mode in ${cache_modes}
                do
                    cmd="CCL_ALLTOALLV_MONOLITHIC_KERNEL=${monolithic_mode}"
                    cmd+=" CCL_ALLTOALLV_TOPO_READ=${read_topo_mode}"
                    cmd+=" CCL_ALLTOALLV_MONOLITHIC_READ_KERNEL=${read_mono_mode}"
                    cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
                    cmd+=" ${bench_options} -l alltoallv -p ${cache_mode}"
                    cmd+=" > ${TEST_LOG} 2>&1"
                    run_cmd "${cmd}"
                    check_log ${TEST_LOG}
                done
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
