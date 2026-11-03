# Shared environment setup for internal_tests/ scripts.
#
# Source this file from any script in this directory. After sourcing:
#   REPO_ROOT   absolute path of the oneccl-v2 repo root
#   BUILD_DIR   active build tree (build_release > build_debug > build)
#               unset if no build tree exists yet
#   LATENCY_BIN absolute path of ccl_latency, or empty if not built
#   BW_BIN      absolute path of ccl_bw, or empty if not built
# Side effect: sources <BUILD_DIR>/_install/env/setvars.sh if present.
#
# Search order for the pt2pt binaries: active build's _install, active build
# tree, then the legacy build_plain/libccl/. Inherited CCL_ROOT (e.g. a system
# install at /opt/intel/oneapi) is intentionally NOT consulted — that install
# does not ship ccl_latency / ccl_bw.

# This file is meant to be sourced. Resolve its own directory rather than
# relying on the caller's BASH_SOURCE.
_env_self="${BASH_SOURCE[0]}"
_env_dir=$(cd "$(dirname "${_env_self}")" && pwd -P)

# _env_dir = .../oneccl-v2/deps/libccl/tests/reg_tests/sycl/internal_tests
# Walk up 6 levels: internal_tests, sycl, reg_tests, tests, libccl, deps.
REPO_ROOT=$(cd "${_env_dir}/../../../../../.." && pwd -P)

BUILD_DIR=""
for cand in "${REPO_ROOT}/build_release" "${REPO_ROOT}/build_debug" "${REPO_ROOT}/build"; do
    if [[ -d "${cand}" ]]; then
        BUILD_DIR="${cand}"
        break
    fi
done

if [[ -n "${BUILD_DIR}" ]]; then
    SETVARS="${BUILD_DIR}/_install/env/setvars.sh"
    if [[ -f "${SETVARS}" ]]; then
        # shellcheck disable=SC1090
        source "${SETVARS}"
    fi
    # Sourcing the install-tree setvars.sh prepends an *installed* CCL path
    # to LD_LIBRARY_PATH, but on this machine the oneAPI setvars.sh that
    # users typically source first puts /opt/intel/oneapi/ccl/.../lib in
    # front. As a result ccl_latency loads the wrong libccl.so.1 and our
    # CCL_SYCL_PT2PT_SIMPLE_THRESHOLD knob is ignored (the symbol simply doesn't
    # exist in the system lib). build_release/env.sh prepends the active
    # build's libccl directly, so source it last to win the resolution.
    BUILD_ENV="${BUILD_DIR}/env.sh"
    if [[ -f "${BUILD_ENV}" ]]; then
        # shellcheck disable=SC1090
        source "${BUILD_ENV}"
    fi
fi

LATENCY_BIN=""
BW_BIN=""
# Active build only. The legacy build_plain/ tree is intentionally NOT searched:
# its binaries are linked against an older oneAPI toolchain and fail at load
# time with "libsycl.so.8: cannot open shared object file" against the SYCL
# runtime installed via current setvars.sh.
#
# Layout note: build_occl wraps libccl as a sub-build, so libccl examples land
# under <build>/libccl/examples/pt2pt/ (and <build>/libccl/_install/...).
# The bare <build>/examples/ tree belongs to the outer oneccl wrapper and does
# not contain ccl_latency / ccl_bw.
_pt2pt_search_prefixes=(
    "${BUILD_DIR:-/dev/null}/libccl/_install"
    "${BUILD_DIR:-/dev/null}/libccl"
    "${BUILD_DIR:-/dev/null}/_install"
    "${BUILD_DIR:-/dev/null}"
)
for _prefix in "${_pt2pt_search_prefixes[@]}"; do
    if [[ -x "${_prefix}/examples/pt2pt/ccl_latency" ]]; then
        LATENCY_BIN="${_prefix}/examples/pt2pt/ccl_latency"
        [[ -x "${_prefix}/examples/pt2pt/ccl_bw" ]] && BW_BIN="${_prefix}/examples/pt2pt/ccl_bw"
        break
    fi
done

# Helper: scripts that require the pt2pt binaries call this; it prints a
# guided error and returns non-zero so the caller can `|| exit`.
require_pt2pt_bins() {
    if [[ -z "${LATENCY_BIN}" ]]; then
        echo "ERROR: ccl_latency / ccl_bw not found in any of:" 1>&2
        for _prefix in "${_pt2pt_search_prefixes[@]}"; do
            echo "    ${_prefix}/examples/pt2pt/" 1>&2
        done
        echo "Build them with:  build_occl -r --examples" 1>&2
        echo "(or --full for everything)" 1>&2
        return 2
    fi
    if [[ -z "${BW_BIN}" ]]; then
        echo "WARNING: ccl_bw not found alongside ccl_latency at $(dirname "${LATENCY_BIN}")" 1>&2
    fi
    return 0
}

# _pt2pt_search_prefixes is intentionally kept in scope: require_pt2pt_bins
# reads it to format the "not found in any of:" error message.
unset _env_self _env_dir _prefix
