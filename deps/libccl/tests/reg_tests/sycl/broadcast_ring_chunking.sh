#!/bin/bash
#
# Reg-test for the slice-size floor knob (CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD)
# in the broadcast pipelined ring.
#
# The threshold acts as a MINIMUM slice size, layered on top of the auto-size:
#     desired_slice_bytes = total_bytes / (world * CHUNKS_PER_RANK)
#     real_slice_bytes    = max(desired_slice_bytes, threshold)   // threshold > 0
#     real_slice_bytes    = desired_slice_bytes                   // threshold == 0
#
# Edge cases this test catches:
#   * threshold == 0          -> pure auto, slice may be small
#   * threshold >> total      -> one big slice, no pipelining (nslices == 1)
#   * threshold between       -> normal floor-clamped multi-slice
#   * last slice < max_slice  -> tail-size handling in the pipeline loop
#
# AUTO is disabled here because AUTO overrides the user threshold; we want to
# observe the threshold behaviour directly. The companion broadcast_ring.sh
# covers the AUTO path.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export I_MPI_JOB_TIMEOUT=360
# Keep this test meaningful on non-ARC CI GPUs by enabling the ARC/PCIe ring
# dispatch gate explicitly.
export CCL_SYCL_FORCE_PCIE=1

# Pin to the ring, copy-engine path, no AUTO so the threshold dial is
# directly observable.
export CCL_SYCL_BROADCAST_RING=1
export CCL_SYCL_BROADCAST_RING_AUTO=0
export CCL_SYCL_BROADCAST_RING_COPY_ENGINE=1

proc_counts="2 4"

# Threshold values in bytes:
#   0           -> pure auto (no floor)
#   131072      -> 128 KB  (small floor, multi-slice for any large msg)
#   1048576     -> 1 MB    (medium floor)
#   33554432    -> 32 MB   (large floor; for a 16 MB msg this forces one slice)
thresholds="0 131072 1048576 33554432"

# Element counts spanning the floor values above, with a tail-causing count
# (16777217 = 16 Mi + 1) to exercise last-slice-smaller logic.
bench_options="-w 1 -i 2 -j off -y 131072,1048576,16777216,16777217"
bench_options+=" -c all -d int32,float32 -l broadcast -b sycl"

for proc_count in ${proc_counts}
do
    for threshold in ${thresholds}
    do
        cmd="CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD=${threshold}"
        cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
        cmd+=" ${bench_options} -q 0"
        cmd+=" > ${TEST_LOG} 2>&1"
        echo "  |TEST CMD| ${cmd}"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm ${TEST_LOG}
echo "Pass"
