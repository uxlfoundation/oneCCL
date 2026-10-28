#!/bin/bash
#
# Spot-check the "1 submit/side vs ~5-7 for LL" claim from the design doc.
#
# Runs ccl_latency with a single small size in both modes under
# CCL_LOG_LEVEL=debug, then greps the log for kernel-submit markers. The
# exact string to count depends on the debug output; this script tries a
# couple of common markers. Tweak SUBMIT_GREP if needed after first run.
#
# This is a qualitative check, not a pass/fail gate -- some submits may
# come from transport/MPI setup, not pt2pt. Inspect the log manually to
# confirm the delta between mode0 and mode1 matches expectations.

set -u
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
require_pt2pt_bins || exit $?

NUM_RANKS="${NUM_RANKS:-2}"
# pt2pt examples don't expose --peers (TODO in pt2pt_base.hpp:135);
# rank pair is hardcoded to (0, 1).
COUNT="${COUNT:-16}"  # 64 B

export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-mpi}"
export CCL_LOG_LEVEL=debug
ulimit -n 1048575 2>/dev/null || true

SUBMIT_GREP="${SUBMIT_GREP:-submit|launch|kernel|parallel_for}"

for mode in 0 1; do
    if [ "${mode}" -eq 0 ]; then
        export CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=999999999999
    else
        export CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=0
    fi
    log="${SCRIPT_DIR}/pt2pt_simple_submit_count_mode${mode}.log"
    echo "=== mode=${mode} (threshold=${CCL_SYCL_PT2PT_SIMPLE_THRESHOLD}) -> ${log} ==="
    mpiexec -l -n "${NUM_RANKS}" "${LATENCY_BIN}" \
        -b gpu -w 1 -i 1 \
        --elem_counts "${COUNT}" -c last \
        > "${log}" 2>&1 || {
            echo "ccl_latency failed, see ${log}"; exit 1;
        }
    hits=$(grep -cEi "${SUBMIT_GREP}" "${log}")
    echo "  grep '${SUBMIT_GREP}' hits: ${hits}"
done

echo
echo "Inspect logs manually for mode0 vs mode1 submit delta:"
echo "    diff <(grep -Ei '${SUBMIT_GREP}' pt2pt_simple_submit_count_mode0.log) \\"
echo "         <(grep -Ei '${SUBMIT_GREP}' pt2pt_simple_submit_count_mode1.log)"
