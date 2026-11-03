#!/bin/bash
#
# Benchmark driver: compare CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=999999999999
# (disabled, huge value -- existing LL256) vs CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=0
# (always-on -- new fused simple protocol) using the pt2pt examples that
# oneCCL already ships.
#
# This script runs every experiment end-to-end and writes logs next to
# itself. It does not analyze results; use pt2pt_simple_bench_summary.py
# for that.
#
# Preconditions:
#   * Built with COMPUTE_BACKEND=dpcpp.
#   * ccl_latency and ccl_bw were produced under build/_install/examples/pt2pt/.
#   * ARC GPUs present and discoverable via sycl-ls.
#
# Overrides (exported before invocation):
#   NUM_RANKS     override rank count (default: 2; pt2pt examples are
#                 hardcoded to ranks 0 and 1, so 2 is the meaningful value
#                 unless you run a patched binary that supports --peers)
#   LAT_MIN       latency sweep min elem count (default: 1)
#   LAT_MAX       latency sweep max elem count (default: 262144)  -- 1 MB of int32
#   LAT_ITERS     latency iters (default: 1000)
#   LAT_WARMUP    latency warmup iters (default: 64)
#   BW_MIN        bw sweep min elem count (default: 1024)         -- 4 KB of int32
#   BW_MAX        bw sweep max elem count (default: 1048576)      -- 4 MB of int32
#   BW_ITERS      bw iters (default: 64)
#   BW_WARMUP     bw warmup iters (default: 16)
#   RUN_E2        set to 0 to skip bandwidth sweep (default: 1)
#   RUN_E3        set to 1 to run correctness sweep (default: 0 -- needs sycl-ls + install tree)
#
# Outputs (in ${SCRIPT_DIR}):
#   pt2pt_simple_bench_latency_mode${N}.log
#   pt2pt_simple_bench_bw_mode${N}.log
#   pt2pt_simple_bench_correctness_mode${N}.log
#   pt2pt_simple_bench_env.log (captured env for reproducibility)

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
require_pt2pt_bins || exit $?
echo "Using LATENCY_BIN=${LATENCY_BIN}"
echo "Using BW_BIN=${BW_BIN}"

# pt2pt examples (ccl_latency / ccl_bw) communicate between rank 0 and rank 1
# only — the --peers option in the upstream cmdline parser is still a TODO
# (see pt2pt_base.hpp:135). Forcing 2 ranks keeps the bench valid and avoids
# wasting cycles on idle ranks. Override only if you know the binary you are
# running has been patched to support --peers.
NUM_RANKS="${NUM_RANKS:-2}"

LAT_MIN="${LAT_MIN:-1}"
LAT_MAX="${LAT_MAX:-262144}"
LAT_ITERS="${LAT_ITERS:-1000}"
LAT_WARMUP="${LAT_WARMUP:-64}"

BW_MIN="${BW_MIN:-1024}"
# 2097152 elements * 4 B (int32) = 8 MiB, the upper test point for
# bandwidth sweeps. Override BW_MAX from the env to extend further.
BW_MAX="${BW_MAX:-2097152}"
BW_ITERS="${BW_ITERS:-64}"
BW_WARMUP="${BW_WARMUP:-16}"

RUN_E2="${RUN_E2:-1}"
RUN_E3="${RUN_E3:-0}"

# Shared runtime env; written into a log for reproducibility.
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-mpi}"
export CCL_WORKER_COUNT="${CCL_WORKER_COUNT:-1}"
export CCL_LOG_LEVEL="${CCL_LOG_LEVEL:-warn}"
ulimit -n 1048575 2>/dev/null || true

ENV_LOG="${SCRIPT_DIR}/pt2pt_simple_bench_env.log"
{
    echo "Timestamp:       $(date --iso-8601=seconds 2>/dev/null || date)"
    echo "Host:            $(hostname)"
    echo "NUM_RANKS:       ${NUM_RANKS}"
    echo "CCL_ATL_TRANSPORT: ${CCL_ATL_TRANSPORT}"
    echo "CCL_WORKER_COUNT: ${CCL_WORKER_COUNT}"
    echo "LATENCY_BIN:     ${LATENCY_BIN}"
    echo "BW_BIN:          ${BW_BIN}"
    echo "LAT_MIN/MAX:     ${LAT_MIN} / ${LAT_MAX}"
    echo "LAT_ITERS:       ${LAT_ITERS} (warmup ${LAT_WARMUP})"
    echo "BW_MIN/MAX:      ${BW_MIN} / ${BW_MAX}"
    echo "BW_ITERS:        ${BW_ITERS} (warmup ${BW_WARMUP})"
    echo "RUN_E2:          ${RUN_E2}"
    echo "RUN_E3:          ${RUN_E3}"
    if command -v sycl-ls >/dev/null 2>&1; then
        echo
        echo "--- sycl-ls ---"
        sycl-ls 2>&1 | head -40
    fi
} > "${ENV_LOG}"

echo "Env recorded in ${ENV_LOG}"

run_mode() {
    local mode="$1"
    local kind="$2"
    local logfile="$3"
    shift 3
    local args=( "$@" )

    if [[ "${mode}" == "0" ]]; then
        export CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=999999999999
    else
        export CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=0
    fi
    echo
    echo "=== ${kind} [CCL_SYCL_PT2PT_SIMPLE_THRESHOLD=${CCL_SYCL_PT2PT_SIMPLE_THRESHOLD}] ==="
    echo "cmd: mpiexec -l -n ${NUM_RANKS} ${args[*]}"
    echo "log: ${logfile}"

    mpiexec -l -n "${NUM_RANKS}" "${args[@]}" > "${logfile}" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "FAILED: exit ${rc} -- see ${logfile}"
        tail -20 "${logfile}"
        return 1
    fi
    # Echo the timing block for quick eyeballing; full log is on disk.
    grep -E '^\[0\][[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+\.' \
        "${logfile}" | head -40 || true
}

# -------- E1: latency sweep --------
for mode in 0 1; do
    run_mode "${mode}" "Latency" \
        "${SCRIPT_DIR}/pt2pt_simple_bench_latency_mode${mode}.log" \
        "${LATENCY_BIN}" \
        -b gpu \
        -w "${LAT_WARMUP}" -i "${LAT_ITERS}" \
        --min_elem_count "${LAT_MIN}" --max_elem_count "${LAT_MAX}" \
        -c last
done

# -------- E2: bandwidth sweep --------
if [[ "${RUN_E2}" == "1" ]]; then
    for mode in 0 1; do
        run_mode "${mode}" "Bandwidth" \
            "${SCRIPT_DIR}/pt2pt_simple_bench_bw_mode${mode}.log" \
            "${BW_BIN}" \
            -b gpu \
            -w "${BW_WARMUP}" -i "${BW_ITERS}" \
            --min_elem_count "${BW_MIN}" --max_elem_count "${BW_MAX}" \
            -c last
    done
fi

# -------- E3: correctness sweep with data check on every iter --------
if [[ "${RUN_E3}" == "1" ]]; then
    for mode in 0 1; do
        run_mode "${mode}" "Correctness (-c all)" \
            "${SCRIPT_DIR}/pt2pt_simple_bench_correctness_mode${mode}.log" \
            "${LATENCY_BIN}" \
            -b gpu \
            -w 8 -i 32 \
            --min_elem_count "${LAT_MIN}" --max_elem_count "${LAT_MAX}" \
            -c all
    done
fi

echo
echo "All runs finished."
echo "Summarize with:"
echo "    python3 ${SCRIPT_DIR}/pt2pt_simple_bench_summary.py ${SCRIPT_DIR}"
