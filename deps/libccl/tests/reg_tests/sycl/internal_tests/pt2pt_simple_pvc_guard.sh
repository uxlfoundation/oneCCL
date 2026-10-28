#!/bin/bash
#
# E4 from the benchmark plan: run on a non-ARC host (PVC) with
# CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=0 and confirm latency is indistinguishable
# from CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=999999999999 (disabled). The dispatch
# is gated on is_arc_card(), so on non-ARC hardware the threshold is a no-op.
#
# Any measurable delta is a bug (likely an ARC check that false-positives
# or an include/build-time conditional drifting).
#
# Usage:
#   ./pt2pt_simple_pvc_guard.sh                 # 2 ranks
#   NUM_RANKS=2 ./pt2pt_simple_pvc_guard.sh
#
# Criterion: mean-of-means latency delta <= 5% per size bucket (noise floor).

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
require_pt2pt_bins || exit $?

# pt2pt examples don't expose --peers (TODO in pt2pt_base.hpp:135).
# Communication is hardcoded to ranks 0 and 1, so 2 ranks is the right count.
NUM_RANKS="${NUM_RANKS:-2}"

export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-mpi}"
export CCL_LOG_LEVEL="${CCL_LOG_LEVEL:-warn}"
ulimit -n 1048575 2>/dev/null || true

for mode in 0 1; do
    if [ "${mode}" -eq 0 ]; then
        export CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=999999999999
    else
        export CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=0
    fi
    log="${SCRIPT_DIR}/pt2pt_simple_pvc_guard_mode${mode}.log"
    echo "=== PVC guard CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=${CCL_SYCL_PT2PT_SIMPLE_THRESHOLD} -> ${log} ==="
    mpiexec -l -n "${NUM_RANKS}" "${LATENCY_BIN}" \
        -b gpu \
        -w 32 -i 500 \
        --min_elem_count 1 --max_elem_count 262144 \
        -c last \
        > "${log}" 2>&1 || {
            echo "FAILED (mode=${mode}); see ${log}"
            exit 1
        }
done

python3 "${SCRIPT_DIR}/pt2pt_simple_pvc_guard_check.py" "${SCRIPT_DIR}"
