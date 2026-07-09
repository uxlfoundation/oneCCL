#!/bin/bash

#########################################################################################
#CCL# Title         : Checking files for excessive size increases.                      #
#CCL# Tracker ID    : INFRA-1835                                                        #
#CCL#                                                                                   #
#CCL# Implemented in: Intel(R) Collective Communication Library 2021.4 (Gold) Update  5 #
#########################################################################################

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BUILD_TYPE="release"

if [[ ! -z ${build_type} ]]
then
    BUILD_TYPE=${build_type}
fi

source ${ROOT_DIR}/utils.sh

check_ccl

THRESHOLD="1.5"
rc=0

# '<path_to_lib> <reference release size> <reference debug size>'
file_list=(
    "/lib/cpu/lib/libccl.so 7047392 60091792"
    "/lib/cpu/lib/libccl.a 16417716 212032324"
    "/lib/cpu_gpu_dpcpp/libccl.so 7802440 112966248"
    "/lib/cpu_gpu_dpcpp/libccl.a 19196186 450351872"
)

for item in "${file_list[@]}" ; do
    FILE=`echo "${item}" | awk '{print $1}'`
    if [[ ${BUILD_TYPE} = "release" ]]
    then
        OLD_SIZE=`echo "${item}" | awk '{print $2}'`
    else
        OLD_SIZE=`echo "${item}" | awk '{print $3}'`
    fi
    if [[ -f ${CCL_ROOT}/${FILE} ]]
    then
        NEW_SIZE=`wc -c ${CCL_ROOT}/${FILE} | awk '{print $1}'`
        # Error if new size > old size * 1.5
        if [[ "${NEW_SIZE}" -gt "$(( OLD_SIZE + OLD_SIZE / 2 ))" ]]; then
            echo "ERROR: ${FILE}, OLD_SIZE = ${OLD_SIZE}, NEW_SIZE = ${NEW_SIZE}, THRESHOLD = ${THRESHOLD}"
            rc=1
        fi
    fi
done

if [[ ${rc} -ne 0 ]] ; then
    echo "Fail"
    exit 1
else
    echo "Pass"
fi
