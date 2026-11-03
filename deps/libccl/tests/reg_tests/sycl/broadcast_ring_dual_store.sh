#!/bin/bash
#
# Correctness reg-test for the dual-store root-copy path in the broadcast
# pipelined ring.
#
# Dual-store folds the root's out-of-place local copy (send_buf -> recv_buf)
# into each per-slice SEND kernel via copy_2dst_wrap: one load from send_buf
# feeds two stores (peer's recv_buf + root's own recv_buf). It applies only
# when:
#     CCL_SYCL_BROADCAST_RING=1
#   AND CCL_SYCL_BROADCAST_RING_COPY_ENGINE=0 (the SYCL kernel backend)
#   AND the call is out-of-place (send_buf != recv_buf)
#
# This test pins the ring and copy-kernel backend, then sweeps both inplace and
# out-of-place calls to verify that:
#   * out-of-place dual-store produces correct data on root, and
#   * inplace silently bypasses dual-store (the dual-store guard checks
#     send_buf != recv_buf, so inplace must remain unaffected).

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

# Pin the path that uses dual-store.
export CCL_SYCL_BROADCAST_RING=1
export CCL_SYCL_BROADCAST_RING_COPY_ENGINE=0
# Disable AUTO so the table cannot decline the ring on these sizes.
export CCL_SYCL_BROADCAST_RING_AUTO=0

proc_counts="2 4"
inplace_modes="0 1"    # dual-store only kicks in when inplace=0

# Element counts chosen to cross slice boundaries (small + large multi-slice).
bench_options="-w 1 -i 2 -j off -y 1024,262144,1048576,16777216"
bench_options+=" -c all -d int32,float32,bfloat16 -l broadcast -b sycl"

for proc_count in ${proc_counts}
do
    for inplace in ${inplace_modes}
    do
        cmd="mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
        cmd+=" ${bench_options} -q ${inplace}"
        cmd+=" > ${TEST_LOG} 2>&1"
        echo "  |TEST CMD| ${cmd}"
        run_cmd "${cmd}"
        check_log ${TEST_LOG}
    done
done

rm ${TEST_LOG}
echo "Pass"
