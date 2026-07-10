# Copyright 2016-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# shellcheck shell=sh
# shellcheck source=/dev/null

# ############################################################################

DEFAULT_CONFIGURATION="cpu_gpu_dpcpp"
DEFAULT_BUNDLED_MPI="yes"

print_help() {
    echo ""
    echo "Usage: vars.sh [--ccl-configuration[=cpu|cpu_gpu_dpcpp] --ccl-bundled-mpi[=yes|no]]"
    echo ""
    echo "By default, --ccl-configuration=${DEFAULT_CONFIGURATION} --ccl-bundled-mpi=${DEFAULT_BUNDLED_MPI}"
    echo "If the above arguments to the sourced script are ignored,"
    echo "you may be using Dash or a similiar POSIX-strict shell."
    echo ""
}

get_script_path() (
    script="$1"
    while [ -L "$script" ] ; do
        # combining next two lines fails in zsh shell
        script_dir=$(command dirname -- "$script")
        script_dir=$(cd "$script_dir" && command pwd -P)
        script="$(readlink "$script")"
        case $script in
            (/*) ;;
            (*) script="$script_dir/$script" ;;
        esac
    done
    # combining next two lines fails in zsh shell
    script_dir=$(command dirname -- "$script")
    script_dir=$(cd "$script_dir" && command pwd -P)
    printf "%s" "$script_dir"
)

_vars_get_proc_name() {
    if [ -n "${ZSH_VERSION:-}" ] ; then
        script="$(ps -p "$$" -o comm=)"
    else
        script="$1"
        while [ -L "$script" ] ; do
            script="$(readlink "$script")"
        done
    fi
    basename -- "$script"
}

_vars_this_script_name="vars.sh"
if [ "$_vars_this_script_name" = "$(_vars_get_proc_name "$0")" ] ; then
    echo "   ERROR: Incorrect usage: this script must be sourced."
    echo "   Usage: . path/to/${_vars_this_script_name}"
    # shellcheck disable=SC2317
    return 255 2>/dev/null || exit 255
fi

prepend_path() (
  path_to_add="$1"
  path_is_now="$2"

  if [ "" = "${path_is_now}" ]; then   # avoid dangling ":"
    printf "%s" "${path_to_add}"
  else
    printf "%s" "${path_to_add}:${path_is_now}"
  fi
)

vars_script_name=""
vars_script_shell="$(ps -p "$$" -o comm=)"

if [ -n "${ZSH_VERSION:-}" ] && [ -n "${ZSH_EVAL_CONTEXT:-}" ] ; then
    # shellcheck disable=SC2249,SC2296
    case $ZSH_EVAL_CONTEXT in (*:file*) vars_script_name="${(%):-%x}" ;; esac ;
elif [ -n "${KSH_VERSION:-}" ] ; then
    if [ "$(set | grep -Fq "KSH_VERSION=.sh.version" ; echo $?)" -eq 0 ] ; then
        # shellcheck disable=SC2296
        vars_script_name="${.sh.file}" ;
    else
        # shellcheck disable=SC2296
        vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
        vars_script_name="$(expr "${vars_script_name:-}" : '^.*sh: \(.*\)\[[0-9]*\]:')" ;
    fi
elif [ -n "${BASH_VERSION:-}" ] ; then
    # shellcheck disable=SC2128,SC3028
    (return 0 2>/dev/null) && vars_script_name="${BASH_SOURCE}" ;
elif [ "dash" = "$vars_script_shell" ] ; then
    # shellcheck disable=SC2296
    vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
    vars_script_name="$(expr "${vars_script_name:-}" : '^.*dash: [0-9]*: \(.*\):')" ;
elif [ "sh" = "$vars_script_shell" ] ; then
    # shellcheck disable=SC2296
    vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
    if [ "$(printf "%s" "$vars_script_name" | grep -Eq "sh: [0-9]+: .*vars\.sh: " ; echo $?)" -eq 0 ] ; then
        vars_script_name="$(expr "${vars_script_name:-}" : '^.*sh: [0-9]*: \(.*\):')" ;
    fi
else
    # shellcheck disable=SC2296
    vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
    if [ "$(printf "%s" "$vars_script_name" | grep -Eq "^.+: [0-9]+: .*vars\.sh: " ; echo $?)" -eq 0 ] ; then # dash
        vars_script_name="$(expr "${vars_script_name:-}" : '^.*: [0-9]*: \(.*\):')" ;
    else
        vars_script_name="" ;
    fi
fi

if [ "" = "$vars_script_name" ] ; then
    >&2 echo "   ERROR: Unable to proceed: possible causes listed below."
    >&2 echo "   This script must be sourced. Did you execute or source this script?" ;
    >&2 echo "   Unrecognized/unsupported shell (supported: bash, zsh, ksh, m/lksh, dash)." ;
    >&2 echo "   May fail in dash if you rename this script (assumes \"vars.sh\")." ;
    >&2 echo "   Can be caused by sourcing from ZSH version 4.x or older." ;
    # shellcheck disable=SC2317
    return 255 2>/dev/null || exit 255
fi

# ############################################################################

args=$*
for arg in $args
do
    case "$arg" in
        --ccl-configuration=*|-ccl-configuration=*)
            ccl_configuration="${arg#*=}"
            ;;
        --ccl-bundled-mpi=*|-ccl-bundled-mpi=*)
            ccl_bundled_mpi="${arg#*=}"
            ;;
    esac
done

ccl_configuration_path=""
if [ -z "${ccl_configuration:-}" ]; then
    ccl_configuration="${DEFAULT_CONFIGURATION}"
elif [ "$ccl_configuration" = "cpu" ]; then
    ccl_configuration_path="ccl/cpu/lib"
elif [ "$ccl_configuration" = "cpu_gpu_dpcpp" ]; then
    ccl_configuration_path=""
else
    echo ":: WARNING: ccl-configuration=${ccl_configuration} is unrecognized."
    echo ":: ccl-configuration will be set to ${DEFAULT_CONFIGURATION}"
    ccl_configuration="${DEFAULT_CONFIGURATION}"
fi

CCL_CONFIGURATION="${ccl_configuration}"; export CCL_CONFIGURATION
CCL_CONFIGURATION_PATH="${ccl_configuration_path}"; export CCL_CONFIGURATION_PATH

if [ -z "${ccl_bundled_mpi:-}" ]; then
    ccl_bundled_mpi="${DEFAULT_BUNDLED_MPI}"
elif [ "$ccl_bundled_mpi" != "yes" ] && [ "$ccl_bundled_mpi" != "no" ]; then
    echo ":: WARNING: ccl_bundled_mpi=${ccl_bundled_mpi} is unrecognized."
    echo ":: ccl_bundled_mpi will be set to ${DEFAULT_BUNDLED_MPI}"
    ccl_bundled_mpi="${DEFAULT_BUNDLED_MPI}"
fi

# ############################################################################

# If this script is located in `etc/<component-name>/vars.sh` it must be
# _sourced_ by the top-level setvars.sh script and is not a stand-alone
# script. It is assumed to be located in a 2024 layout.

# If located in `<component>/<version>/env/vars.sh` this vars.sh script is a
# stand-alone script and is located in a 2023 or earlier layout.

# NOTE: See the setvars.sh script for a list of the top-level environment
# variables that it provides. Also, if a comoponent vars.sh script must
# augment an existing environment variable it should use the prepend_path()
# and prepend_manpath() functions.

WORK_DIR=$(get_script_path "${vars_script_name:-}")

if [ "$(basename "${WORK_DIR}")" = "env" ] ; then   # assume a 2023 layout

    CCL_ROOT="$(dirname "${WORK_DIR}")"

    if [ -z "${SETVARS_CALL:-}" ] ; then
        if [ "$ccl_bundled_mpi" = "yes" ] ; then
            if [ -f "$CCL_ROOT/../../mpi/latest/env/vars.sh" ] ; then
                if [ -z ${I_MPI_ROOT:-} ] ; then
                    source "$CCL_ROOT/../../mpi/latest/env/vars.sh"
                else
                    bundled_mpi_root=$(realpath "$CCL_ROOT/../../mpi/latest/")
                    if [ ${I_MPI_ROOT} != ${bundled_mpi_root} ] ; then
                        echo ":: WARNING: Found I_MPI_ROOT = ${I_MPI_ROOT}"
                        echo ":: Use bundled MPI from ${bundled_mpi_root}"
                        echo ":: for best compatibility."
                    fi
                fi
            else
                if [ -z ${I_MPI_ROOT:-} ] ; then
                    echo ":: WARNING: '--ccl-bundled-mpi' is set to '${ccl_bundled_mpi}'"
                    echo ":: but the package does not provide Intel® MPI Library distribution."
                    echo ":: Please source an MPI implementation for oneCCL to work correctly."
                fi
            fi
        fi
    fi

    if [ "$ccl_configuration" = "cpu" ]; then
        # Add directory containing libccl.so.2.0 but with lower priority than the
        # directory with CPU only libccl.so.1.0
        LIBRARY_PATH=$(prepend_path "${CCL_ROOT}/lib" "${LIBRARY_PATH:-}"); export LIBRARY_PATH
        LD_LIBRARY_PATH=$(prepend_path "${CCL_ROOT}/lib" "${LD_LIBRARY_PATH:-}"); export LD_LIBRARY_PATH
    fi

    LIBRARY_PATH=$(prepend_path "${CCL_ROOT}/lib/${ccl_configuration_path}" "${LIBRARY_PATH:-}"); export LIBRARY_PATH
    LD_LIBRARY_PATH=$(prepend_path "${CCL_ROOT}/lib/${ccl_configuration_path}" "${LD_LIBRARY_PATH:-}"); export LD_LIBRARY_PATH

    CPATH=$(prepend_path "${CCL_ROOT}/include" "${CPATH:-}"); export CPATH
    CMAKE_PREFIX_PATH=$(prepend_path "${CCL_ROOT}/lib/cmake/oneCCL" "${CMAKE_PREFIX_PATH:-}"); export CMAKE_PREFIX_PATH
    PKG_CONFIG_PATH=$(prepend_path "${CCL_ROOT}/lib/pkgconfig/${ccl_configuration_path}" "${PKG_CONFIG_PATH:-}"); export PKG_CONFIG_PATH

else                                                # must be a 2024 layout
    if [ -z "${SETVARS_CALL:-}" ] ; then
    >&2 echo " "
    >&2 echo ":: ERROR: This script must be sourced by setvars.sh."
    >&2 echo "   Try 'source <install-dir>/setvars.sh --help' for help."
    >&2 echo " "
    return 255
    fi

    if [ -z "${ONEAPI_ROOT:-}" ] ; then
    >&2 echo " "
    >&2 echo ":: ERROR: This script requires that the ONEAPI_ROOT env variable is set."
    >&2 echo "   Try 'source <install-dir>\setvars.sh --help' for help."
    >&2 echo " "
    return 254
    fi

    CCL_ROOT=${ONEAPI_ROOT}
    if [ -n "${ccl_configuration_path}" ] ; then
        CPATH=$(prepend_path "${ONEAPI_ROOT}/include" "${CPATH:-}"); export CPATH
        LIBRARY_PATH=$(prepend_path "${ONEAPI_ROOT}/lib/${ccl_configuration_path}" "${LIBRARY_PATH:-}"); export LIBRARY_PATH
        LD_LIBRARY_PATH=$(prepend_path "${ONEAPI_ROOT}/lib/${ccl_configuration_path}" "${LD_LIBRARY_PATH:-}"); export LD_LIBRARY_PATH
        PKG_CONFIG_PATH=$(prepend_path "${ONEAPI_ROOT}/lib/pkgconfig/${ccl_configuration_path}" "${PKG_CONFIG_PATH:-}"); export PKG_CONFIG_PATH
    fi

fi

export CCL_ROOT
