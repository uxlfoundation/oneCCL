#!/bin/bash
#
# Performance regression gate for the simple pt2pt protocol.
#
# Chains the existing bench infrastructure into a pass/fail boundary:
#   1. pt2pt_simple_bench.sh runs ccl_latency / ccl_bw with
#      CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=999999999999 (LL256 baseline) and
#      =0 (simple always-on), writing
#      pt2pt_simple_bench_{latency,bw}_mode{0,1}.log next to itself.
#   2. pt2pt_simple_bench_summary.py parses those logs and exits non-zero
#      when the simple protocol misses the design thresholds:
#         - >= 1.5x latency speedup at <= 16 KB
#         - within 10% of LL latency at >= 1 MB
#         - >= 90% of LL bandwidth across the sweep
#      (See pt2pt_simple_bench_summary.py top-of-file constants.)
#
# Skipped on hardware where is_arc_card() returns false (e.g. PVC). The
# dispatch in send_sycl.cpp/recv_sycl.cpp gates the simple-protocol branch
# on is_arc_card(); when that's false, both modes run LL256 and the
# threshold check is meaningless. is_arc_card() returns true for
# Arc client GPUs (family4-6) and Arc server GPUs (family7).

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

is_arc_host() {
    if ! command -v sycl-ls >/dev/null 2>&1; then
        return 1
    fi
    sycl-ls 2>/dev/null | grep -qiE 'arc'
}

if ! is_arc_host; then
    echo "skip: ARC GPUs not detected via sycl-ls; perf gate is ARC-only"
    exit 0
fi

BENCH="${SCRIPT_DIR}/pt2pt_simple_bench.sh"
SUMMARY="${SCRIPT_DIR}/pt2pt_simple_bench_summary.py"

if [[ ! -x "${BENCH}" ]]; then
    echo "ERROR: ${BENCH} missing or not executable"
    exit 2
fi
if [[ ! -f "${SUMMARY}" ]]; then
    echo "ERROR: ${SUMMARY} missing"
    exit 2
fi

echo "=== running pt2pt_simple_bench.sh (LL256 vs simple) ==="
if ! bash "${BENCH}"; then
    echo "FAILED: bench script exited non-zero"
    exit 1
fi

echo
echo "=== evaluating thresholds ==="
if ! python3 "${SUMMARY}" "${SCRIPT_DIR}"; then
    echo "FAILED: simple protocol regressed against LL256 thresholds"
    exit 1
fi

echo "Pass"
