#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

unset CCL_ROOT
unset I_MPI_ROOT
unset FI_PROVIDER_PATH

pushd ${WORKSPACE} > /dev/null
mkdir -p build && cd build
cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCOMPUTE_BACKEND=dpcpp > ${TEST_LOG} 2>&1
make -j install >> ${TEST_LOG} 2>&1
popd > /dev/null
source ${WORKSPACE}/build/_install/env/setvars.sh
check_ccl
check_impi

echo "CCL_ROOT: " $CCL_ROOT >> ${TEST_LOG}
echo "I_MPI_ROOT: " $I_MPI_ROOT >> ${TEST_LOG}
echo "FI_PROVIDER_PATH: " $FI_PROVIDER_PATH >> ${TEST_LOG}

transports="ofi mpi"
ofi_provs="$(get_default_prov)"
for proc_count in ${proc_counts}
do
    for transport in ${transports}
    do
        for ofi_prov in ${ofi_provs}
        do
            cmd=" CCL_ATL_TRANSPORT=${transport}"
            cmd+=" FI_PROVIDER=${ofi_prov}"
            cmd+=" mpiexec -l -n 4 -ppn 2"
            cmd+=" ${WORKSPACE}/build/_install/examples/benchmark/benchmark >> ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done

rm ${TEST_LOG}

echo "Pass"

