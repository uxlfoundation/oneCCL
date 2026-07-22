#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

LOG=${SCRIPT_DIR}/${TEST_LOG}

source ${ROOT_DIR}/utils.sh

if [[ ${MODE} == "gpu" ]]
then
    make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"
else
    make_common_actions ${SCRIPT_DIR} ${TEST_LOG}
fi

check_run() {
    expected_str=$1
    log=$2
    grep "${expected_str}" ${log} > /dev/null 2>&1
    check_ret $?
}

bench_options="-j off -w 1 -i 2 -c all -l allreduce -y 1024"

# case 1: fake libfaric path
CCL_LOG_LEVEL=debug CCL_ATL_TRANSPORT=ofi CCL_OFI_LIBRARY_PATH="/lib/fake-path/libfabric.so.123" \
mpiexec -l -n 2 -ppn 2 ${SCRIPT_DIR}/benchmark ${bench_options} > ${LOG} 2>&1
check_ret $?
check_log ${LOG}
check_run "could not initialize OFI api" ${LOG}
rm ${LOG}

# case 2: fake mpi path
CCL_LOG_LEVEL=debug CCL_ATL_TRANSPORT=mpi CCL_MPI_LIBRARY_PATH="/lib/fake-path/libmpi.so.123" \
mpiexec -l -n 2 -ppn 2 ${SCRIPT_DIR}/benchmark ${bench_options} > ${LOG} 2>&1
check_ret $?
check_log ${LOG}
check_run "could not initialize MPI api" ${LOG}
rm ${LOG}

# case 3: fake libfaric and mpi path
CCL_OFI_LIBRARY_PATH="/lib/fake-path/libfabric.so.123" CCL_MPI_LIBRARY_PATH="/lib/fake-path/libmpi.so.123" \
mpiexec -l -n 2 -ppn 2 ${SCRIPT_DIR}/benchmark ${bench_options} > ${LOG} 2>&1
rc=$?
if [[ $rc -eq 0 ]]
then
    echo "Fail"
    exit -1
fi
check_run "could not initialize any transport library" ${LOG}
rm ${LOG}

#case 4: path with non-secure '..' characters
CCL_LOG_LEVEL=debug CCL_ATL_TRANSPORT=mpi CCL_MPI_LIBRARY_PATH="/lib/../../fake-path/libmpi.so.123" \
mpiexec -l -n 2 -ppn 2 ${SCRIPT_DIR}/benchmark ${bench_options} > ${LOG} 2>&1
rc=$?
check_ret $?
check_run "could not initialize MPI api" ${LOG}
check_run "error: path contains invalid characters" ${LOG}
rm ${LOG}

#case 5: path with non-secure '%' characters
CCL_LOG_LEVEL=debug CCL_ATL_TRANSPORT=mpi CCL_MPI_LIBRARY_PATH="/lib/fake-%01%02/libmpi.so.123" \
mpiexec -l -n 2 -ppn 2 ${SCRIPT_DIR}/benchmark ${bench_options} > ${LOG} 2>&1
rc=$?
check_ret $?
check_run "could not initialize MPI api" ${LOG}
check_run "error: path contains invalid characters" ${LOG}
rm ${LOG}

# case 6: correct OFI path
CCL_ATL_TRANSPORT=ofi CCL_OFI_LIBRARY_PATH="$(get_ofi_path)" \
mpiexec -l -n 2 -ppn 2 ${SCRIPT_DIR}/benchmark ${bench_options} > ${LOG} 2>&1
check_ret $?
check_log ${LOG}
rm ${LOG}

# case 7: correct MPI path
CCL_ATL_TRANSPORT=mpi CCL_MPI_LIBRARY_PATH="$(get_mpi_path)" \
mpiexec -l -n 2 -ppn 2 ${SCRIPT_DIR}/benchmark ${bench_options} > ${LOG} 2>&1
check_ret $?
check_log ${LOG}
rm ${LOG}

echo "Pass"
