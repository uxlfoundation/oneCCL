#!/bin/bash
#
# Reg-test for the ARC/BMG "ring is ON by default" logic.
#
# On ARC/BMG hardware both broadcast_sycl_single_node and broadcast_large_impl
# default use_ring=true unless the user EXPLICITLY set CCL_SYCL_BROADCAST_RING.
# The two dispatchers read the env independently, so a regression in either
# site would cause them to disagree about whether the ring is in use — which
# is the bug class this test guards against.
#
# CI hosts may not be ARC/BMG, so the test uses CCL_SYCL_FORCE_PCIE=1 to make
# is_arc_card() return true on any hardware (including PVC). The cases:
#
#   case 1: FORCE_PCIE=1, RING unset       -> arc-default kicks in, ring used
#   case 2: FORCE_PCIE=1, RING=1 (explicit) -> ring used (same path, explicit)
#   case 3: FORCE_PCIE=1, RING=0 (explicit) -> ring disabled even on "arc"
# Each case checks broadcast correctness.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

source_benchmark="${ROOT_DIR}/../../examples/benchmark/benchmark"
workspace_root="$(realpath "${ROOT_DIR}/../../../..")"
prebuilt_benchmark="${workspace_root}/build/libccl/examples/benchmark/benchmark"
remove_staged_benchmark=0

# A top-level oneCCL-v2 build already contains the benchmark needed by this
# test. Reuse it for direct test runs instead of configuring a second build
# under deps/libccl/examples/build_sycl.
if [[ ! -x "${source_benchmark}" && -x "${prebuilt_benchmark}" ]]
then
    cp "${prebuilt_benchmark}" "${source_benchmark}" || exit 1
    remove_staged_benchmark=1
fi

cleanup_staged_benchmark() {
    if ((remove_staged_benchmark))
    then
        rm -f "${source_benchmark}"
    fi
}
trap cleanup_staged_benchmark EXIT

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export I_MPI_JOB_TIMEOUT=360

proc_counts="2 4"

bench_options="-w 1 -i 2 -j off -y 1024,1048576,16777216"
bench_options+=" -c all -d int32,float32 -l broadcast -b sycl"

run_one() {
    local label="$1"
    local extra_env="$2"
    local proc_count="$3"
    local inplace="$4"

    local cmd="CCL_SYCL_BROADCAST_RING_AUTO=0 ${extra_env}"
    cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
    cmd+=" ${bench_options} -q ${inplace}"
    cmd+=" > ${TEST_LOG} 2>&1"
    echo "=== ${label} ==="
    echo "  |TEST CMD| ${cmd}"
    run_cmd "${cmd}"
    check_log ${TEST_LOG}
}

for proc_count in ${proc_counts}
do
    for inplace in 0 1
    do
        # case 1: arc-default-on (RING var absent)
        run_one "arc-default-on (FORCE_PCIE=1, RING unset)" \
                "CCL_SYCL_FORCE_PCIE=1" \
                "${proc_count}" "${inplace}"

        # case 2: arc + explicit RING=1
        run_one "arc + explicit RING=1" \
                "CCL_SYCL_FORCE_PCIE=1 CCL_SYCL_BROADCAST_RING=1" \
                "${proc_count}" "${inplace}"

        # case 3: arc + explicit RING=0 (user override wins)
        run_one "arc + explicit RING=0" \
                "CCL_SYCL_FORCE_PCIE=1 CCL_SYCL_BROADCAST_RING=0" \
                "${proc_count}" "${inplace}"
    done
done

rm ${TEST_LOG}
echo "Pass"
