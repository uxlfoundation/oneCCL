#!/bin/bash
#
# Reg-test for the pipelined-ring broadcast path (CCL_SYCL_BROADCAST_RING=1).
#
# Sweeps the main dials introduced by the broadcast-ring series:
#   * RING       : 0 (baseline)    vs  1 (pipelined ring)
#   * RING_AUTO  : 0 (raw ring)    vs  1 (consults the policy table)
#   * COPY_ENGINE: 0 (SYCL kernel) vs  1 (DMA)
#
# Counts span the small/large threshold and the AUTO table buckets so that
# both "table declines ring" and "table accepts ring" branches are hit.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export I_MPI_JOB_TIMEOUT=360
# The ring dispatcher is ARC/BMG-only. Force that hardware gate so this test
# exercises the ring on other CI GPU families instead of silently using IPC.
export CCL_SYCL_FORCE_PCIE=1

proc_counts="2 4"
ring_modes="0 1"
auto_modes="0 1"
copy_engine_modes="0 1"
inplace_modes="0 1"

# Counts chosen to cross the small/large threshold and land inside AUTO buckets.
# 1024 elem * 4 B = 4 KB (small path) ; 262144 elem * 4 B = 1 MB (large, below AUTO bucket) ;
# 16777216 elem * 4 B = 64 MB (large, inside AUTO bucket) ;
# 67108864 elem * 4 B = 256 MB (large, inside AUTO bucket).
bench_options="-w 1 -i 2 -j off -y 1024,262144,16777216,67108864"
bench_options+=" -c all -d int32,float32,bfloat16 -l broadcast -b sycl"

for proc_count in ${proc_counts}
do
    for ring in ${ring_modes}
    do
        for auto_mode in ${auto_modes}
        do
            # AUTO is only meaningful when the ring is on; skip the redundant sweep.
            if [ "${ring}" = "0" ] && [ "${auto_mode}" = "1" ]
            then
                continue
            fi
            for copy_engine in ${copy_engine_modes}
            do
                # Copy mechanism is only consulted when the ring is on.
                if [ "${ring}" = "0" ] && [ "${copy_engine}" = "1" ]
                then
                    continue
                fi
                for inplace in ${inplace_modes}
                do
                    cmd="CCL_SYCL_BROADCAST_RING=${ring}"
                    cmd+=" CCL_SYCL_BROADCAST_RING_AUTO=${auto_mode}"
                    cmd+=" CCL_SYCL_BROADCAST_RING_COPY_ENGINE=${copy_engine}"
                    cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
                    cmd+=" ${bench_options} -q ${inplace}"
                    cmd+=" > ${TEST_LOG} 2>&1"
                    echo "  |TEST CMD| ${cmd}"
                    run_cmd "${cmd}"
                    check_log ${TEST_LOG}
                done
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
