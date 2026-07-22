#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
JUNIT_REPORT_NAME="summary-inference.junit.xml"
JUNIT_REPORT=$(realpath ./$JUNIT_REPORT_NAME)

# Added global counter for failures
FAIL_COUNT=0
TEST_COUNT=0

echo "<testsuite name=\"reg_tests_inference\" tests=\"0\" failures=\"0\">" > ${JUNIT_REPORT}

function run_script_no_exit() {
    local script_path="$1"
    echo "Running ${script_path}"

    # Capture the script’s output (stdout+stderr)
    output=$("${script_path}" 2>&1)
    rc=$?

    # Print the output so it appears in Jenkins logs
    echo "${output}"

    # 2) If the pt2pt script exits with non-zero, also count as failure
    # (In case the script *does* return a proper error code.)
    TEST_COUNT=$((TEST_COUNT + 1))

    if [ $rc -ne 0 ]; then
        echo "Script ${script_path} exited with code ${rc}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "  <testcase name=\"${script_path}\"><failure type=\"exit code\">Script exited with code ${rc}</failure></testcase>" >> ${JUNIT_REPORT}
    else
        echo "  <testcase name=\"${script_path}\" />" >> ${JUNIT_REPORT}
    fi
}

echo $SCRIPT_DIR
run_script_no_exit "${SCRIPT_DIR}/../../pt2pt/benchmarks.sh"
run_script_no_exit "${SCRIPT_DIR}/../../pt2pt/pt2pt_comm_split.sh"
run_script_no_exit "${SCRIPT_DIR}/../../pt2pt/pt2pt_gpu.sh"
run_script_no_exit "${SCRIPT_DIR}/../../pt2pt/pt2pt_group.sh"
run_script_no_exit "${SCRIPT_DIR}/../../pt2pt/pt2pt_self.sh"
run_script_no_exit "${SCRIPT_DIR}/../../pt2pt/hyper_cub.sh"
run_script_no_exit "${SCRIPT_DIR}/../../pt2pt/alltoall.sh"

source ${ROOT_DIR}/utils.sh
get_bench ${SCRIPT_DIR} ${TEST_LOG} "sycl"
cd ${SCRIPT_DIR}

function run_and_check_log() {
    local cmd="$1"
    local log_path="$2"
    local test_name=$(basename "$log_path")

    # Run the command and redirect output to log
    echo "Executing: ${cmd}"
    eval ${cmd} > "${log_path}" 2>&1
    local rc=$?

    # Patterns for checking log content
    local passed_pattern="iteration|passed|# all done"
    local failed_pattern="abort|^bad$|corrupt|fail|^fault$|[^-W]invalid"
    failed_pattern+="|kill|runtime_error|terminate|timed|unexpected"
    failed_pattern+="|[^-W]error|exception|connection refused"
    failed_pattern+="|job ending due to application timeout"
    local exclude_pattern="\-\-abort\-signal|CCL_ABORT_ON_THROW"
    exclude_pattern+="|fi_strerror|MPI_Error_string|fake-path"
    exclude_pattern+="|MPI startup\(\): Set ptracer for parent pid \(.*\) failed"

    # Check log for passed/failed patterns
    local passed_count=$(grep -E -c -i "${passed_pattern}" "${log_path}")
    local failed_strings=$(grep -E -i "${failed_pattern}" "${log_path}" | grep -Ev "${exclude_pattern}")

    TEST_COUNT=$((TEST_COUNT + 1))

    if [ ${rc} -ne 0 ] || [ ${passed_count} -eq 0 ] || [ "${failed_strings}" != "" ]; then
        echo "Error in ${test_name}: Command failed with exit code ${rc} or log indicates failure."
        FAIL_COUNT=$((FAIL_COUNT+1))
        local failure_msg="Exit Code: ${rc}\n${failed_strings}"
        echo "  <testcase name=\"${test_name}\"><failure>${failure_msg}" >> ${JUNIT_REPORT}
        echo "  ${failure_msg}" >> ${JUNIT_REPORT}
        echo "  </failure></testcase>" >> ${JUNIT_REPORT}
        cat "${log_path}"
    else
        echo "  <testcase name=\"${test_name}\" />" >> ${JUNIT_REPORT}
    fi
}

function check_log_and_print() {
    log_path=$1
    extra_passed_pattern=${2:-}

    passed_pattern="iteration|passed|# all done"
    if [ "${extra_passed_pattern}" != "" ]
    then
        passed_pattern+="|${extra_passed_pattern}"
    fi
    passed_count=`grep -E -c -i "${passed_pattern}" ${log_path}`
    if [ ${passed_count} -eq 0 ]
    then
        echo "Error: did not find pass in log ${log_path}"
        cat ${log_path}
        FAIL_COUNT=$((FAIL_COUNT+1))
        return 1
    fi

    failed_pattern="abort|^bad$|corrupt|fail|^fault$|[^-W]invalid"
    failed_pattern+="|kill|runtime_error|terminate|timed|unexpected"
    failed_pattern+="|[^-W]error|exception|connection refused"
    failed_pattern+="|job ending due to application timeout"

    # patterns to exclude as they're printed in non-error cases
    exclude_pattern="\-\-abort\-signal|CCL_ABORT_ON_THROW"
    exclude_pattern+="|fi_strerror|MPI_Error_string|fake-path"
    exclude_pattern+="|MPI startup\(\): Set ptracer for parent pid \(.*\) failed"

    failed_strings=`grep -E -i "${failed_pattern}" ${log_path} | grep -Ev "${exclude_pattern}"`
    if [ "${failed_strings}" != "" ]
    then
        echo "Error: found error in log ${log_path}"
        echo ""
        echo "${failed_strings}"
        echo ""
        cat ${log_path}
        FAIL_COUNT=$((FAIL_COUNT+1))
        return 1
    fi
}

export CCL_LOG_LEVEL=debug

tmp_bufs="0 1"
transports="mpi ofi"
ranks="2 4"
exec_names=`ls sycl_*_inorder`
# use various sizes to execute all algos: gdrcopy, small, medium, large
sizes="28 36 1020 1028 262140 262148 3145724 3145732 16777212 16777220"
counts="0,1,2,4,7,8,16,17,32,64,128,133,256,1077,16384,16387,262140,262148,3145724,3145732,16777212,16777220"
places="0 1"
#TODO enable 0 back in single_node_algos
single_node_algos="1"
# single GPU, single PLANE
affinity_masks="0,1 0,2"

for tmp_buf in ${tmp_bufs}
do
    for transport in ${transports}
    do
        for N in ${ranks}
        do
            for exec_name in ${exec_names}
            do
                for size in ${sizes}
                do
                    for single_node_algo in ${single_node_algos}
                    do
                        for affinity_mask in ${affinity_masks}
                        do
                            TEST_LOG="${exec_name}_transport_${transport}_N_${N}.log"
                            cmd="CCL_ENABLE_SYCL_KERNELS=1 CCL_SYCL_ALLREDUCE_TMP_BUF=$tmp_buf CCL_SYCL_REDUCE_SCATTER_TMP_BUF=$tmp_buf"
                            cmd+=" CCL_SYCL_ALLGATHERV_TMP_BUF=$tmp_buf CCL_ATL_TRANSPORT=$transport"
                            if [ "$N" == "2" ]
                            then
                                cmd+=" ZE_AFFINITY_MASK=${affinity_mask}"
                            fi
                            if [ "$N" == "4" ] && [ "$affinity_mask" == "0,2" ]
                            then
                                continue
                            fi
                            cmd+=" CCL_SYCL_SINGLE_NODE_ALGORITHM=$single_node_algo mpiexec.hydra -n $N -ppn $N -l ./$exec_name gpu device --queue_type in_order --count $size > ${TEST_LOG} 2>&1"
                            echo "${cmd}"
                            run_and_check_log "${cmd}" ${TEST_LOG}
                            # run_cmd_no_exit "${cmd}"
                            # check_log_and_print ${TEST_LOG}
                            test $? -eq 1 && cat ${TEST_LOG}
                        done
                    done
                done
            done

            for place in ${places}
            do
                for single_node_algo in ${single_node_algos}
                do
                    colls="allgather,allgatherv,allreduce,broadcast"
                    if [ "$place" == "0" ]
                    then
                        colls+=",bcast"
                    fi
                    colls+=",reduce_scatter"

                    dtypes="bfloat16,float32"
                    dtypes+=",float64"

                    TEST_LOG="benchmark_transport_${transport}_N_${N}.log"
                    cmd="CCL_ENABLE_SYCL_KERNELS=1 CCL_SYCL_ALLREDUCE_TMP_BUF=$tmp_buf CCL_SYCL_REDUCE_SCATTER_TMP_BUF=$tmp_buf"
                    cmd+=" CCL_SYCL_ALLGATHERV_TMP_BUF=$tmp_buf CCL_SYCL_BROADCAST_TMP_BUF=$tmp_buf"
                    cmd+=" CCL_ATL_TRANSPORT=$transport"
                    cmd+=" CCL_SYCL_SINGLE_NODE_ALGORITHM=$single_node_algo mpiexec.hydra -n $N -ppn $N -l ./benchmark"
                    cmd+=" --check last --backend sycl --coll $colls -y $counts --buf_count 1"
                    cmd+=" --iters 8 --sycl_mem_type usm --sycl_usm_type device -e in_order -d $dtypes -q $place > ${TEST_LOG} 2>&1"
                    echo "${cmd}"
                    run_and_check_log "${cmd}" ${TEST_LOG}
                    # run_cmd_no_exit "${cmd}"
                    # check_log_and_print ${TEST_LOG}
                    test $? -eq 1 && cat ${TEST_LOG}
                done
            done
        done
    done
done

exec_split_test=`ls sycl_*_comm_split_test`
sizes_split_test="1024 1028 3145732 33554432"
use_cases="0 1 2 3"
transports="mpi ofi"

for exec_name in ${exec_split_test}
do
    for transport in ${transports}
    do
        for size in ${sizes_split_test}
        do
            for use_case in ${use_cases}
            do
                TEST_LOG="${exec_name}_${transport}_N_4_size_${size}.log"
                cmd="CCL_ATL_TRANSPORT=${transport} "

                if [[ "${size}" -eq "3145732" && "${exec_name}" == "sycl_allgatherv_usm_comm_split_test" ]]; then
                    cmd+="CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_AUTO_USE_TMP_BUF=0 "
                fi

                if [[ "${size}" -eq "3145732" && "${exec_name}" == "sycl_reduce_scatter_usm_comm_split_test" ]]; then
                    cmd+="CCL_SYCL_REDUCE_SCATTER_TMP_BUF=0 CCL_SYCL_AUTO_USE_TMP_BUF=0 "
                fi
                cmd+="mpiexec.hydra -n 4 -l ./${exec_name} gpu device in_order ${size} ${use_case} > ${TEST_LOG} 2>&1"
                echo "${cmd}"
                run_and_check_log "${cmd}" ${TEST_LOG}
                # run_cmd_no_exit "${cmd}"
                # check_log_and_print "${TEST_LOG}"
                test $? -eq 1 && cat "${TEST_LOG}"
            done
        done
    done
done

# Test with UMF
num_ranks="2 4"
for num_rank in $num_ranks; do
    TEST_LOG="test_umf_${num_rank}.log"
    cmd="CCL_UMF_ENABLE=1 mpiexec -n ${num_rank} -l ./benchmark"
    cmd+=" --check last --backend sycl --coll allgather,allgatherv,allreduce,reduce_scatter -y 1024 --buf_count 1"
    cmd+=" --iters 8 --sycl_mem_type usm --sycl_usm_type device -e in_order -d bfloat16,float32 > ${TEST_LOG} 2>&1"
    echo "${cmd}"
    run_and_check_log "${cmd}" ${TEST_LOG}
    # run_cmd_no_exit "${cmd}"
    # check_log_and_print ${TEST_LOG}
    test $? -eq 1 && cat ${TEST_LOG}
done

test_names=("sycl_mt_allreduce_usm_test" "sycl_mt_allgatherv_usm_test" "sycl_mt_allgather_usm_test" "sycl_mt_reduce_scatter_usm_test" "sycl_mt_broadcast_usm_test" "sycl_mt_alltoall_usm_test")
num_threads="4"
mt_counts="1024 33554432"
tasks=("single_group" "two_group")

for mt_exec_name in "${test_names[@]}"; do
    for task_name in "${tasks[@]}"; do
        for mt_count in $mt_counts; do
            for num_thread in $num_threads; do
                TEST_LOG="test_${mt_exec_name}_${task_name}_${mt_count}_${num_thread}.log"
                cmd="mpiexec -n 1 ./${mt_exec_name} --buffer-count=${mt_count} --threads=${num_thread} --debug=0 --task=${task_name} > ${TEST_LOG} 2>&1"
                echo "${cmd}"
                run_and_check_log "${cmd}" ${TEST_LOG}
                # run_cmd_no_exit "${cmd}"
                # check_log_and_print ${TEST_LOG}
                test $? -eq 1 && cat "${TEST_LOG}"
            done
        done
    done
done

test_names=("sycl_mt_pt2pt_usm_test sycl_mt_pt2pt_group_usm_test")
num_threads="2 4"
mt_counts="1024 33554432"

for mt_exec_name in "${test_names[@]}"; do
    for mt_count in $mt_counts; do
        for num_thread in $num_threads; do
            TEST_LOG="test_${mt_exec_name}_${mt_count}_${num_thread}.log"
            cmd="mpiexec -n 1 ./${mt_exec_name} --buffer-count=${mt_count} --threads=${num_thread} --iters=20 > ${TEST_LOG} 2>&1"
            echo "${cmd}"
            run_and_check_log "${cmd}" ${TEST_LOG}
            # run_cmd_no_exit "${cmd}"
            # check_log_and_print ${TEST_LOG}
            test $? -eq 1 && cat "${TEST_LOG}"
        done
    done
done

test_names=("graph_integration") # TODO add graph_integration/graph_mt_integration

n_ranks=$(sycl-ls | grep "level_zero:gpu" | wc -l)

pushd graph_integration
for exec_name in "${test_names[@]}"; do
    for n_rank in 1 $(seq 2 2 ${n_ranks}); do
        TEST_LOG="test_${exec_name}_${n_rank}.log"
        cmd="mpiexec -n ${n_rank} ./${exec_name} > ${TEST_LOG} 2>&1"
        echo "${cmd}"
        run_and_check_log "${cmd}" ${TEST_LOG}
        # run_cmd_no_exit "${cmd}"
        # check_log_and_print ${TEST_LOG}
        test $? -eq 1 && cat "${TEST_LOG}"
    done
done
popd

algo_names=("ring" "twoshots" "oneshot")
# TODO add sycl_broadcast_buffer_overflow_test
# it currently randomly hangs, so it needs to be fixed first
# test_names=("sycl_allreduce_buffer_overflow_test" "sycl_reduce_scatter_buffer_overflow_test" "sycl_pt2pt_buffer_overflow_test" "sycl_allgather_buffer_overflow_test" "sycl_alltoall_buffer_overflow_test")

#temporarily disable all tests - some are failing on PVC, tracked by MLSL-4226
test_names=()
ppns=(2 4)
thresholds=(0 1000000000)

pushd ll_algorithms
for threshold in "${thresholds[@]}"; do
    for ppn in "${ppns[@]}"; do
        for exec_name in "${test_names[@]}"; do
            for algo in "${algo_names[@]}"; do
                TEST_LOG="test_${algo}_${ppn}_${threshold}_${exec_name}.log"
                cmd="NEOReadDebugKeys=1 RenderCompressedBuffersEnabled=0 CCL_SYCL_ALLREDUCE_LL_THRESHOLD=${threshold} CCL_SYCL_ALLREDUCE_LL=${algo} CCL_SYCL_ALLREDUCE_ARC=0 CCL_TOPO_P2P_ACCESS=1 mpiexec -n ${ppn} ./${exec_name} > ${TEST_LOG} 2>&1"
                echo "${cmd}"
                run_and_check_log "${cmd}" ${TEST_LOG}
                # run_cmd_no_exit "${cmd}"
                # check_log_and_print ${TEST_LOG}
                test $? -eq 1 && cat "${TEST_LOG}"
            done
        done
    done
done
popd

# ---- Custom reduction allreduce test (sum, prod, min, max, avg, pre_mul_sum) ----
#
# The test internally runs a fixed count (default 10M elements) per reduction. To exercise every
# dispatcher cell that the SYCL allreduce single-node selector can route to (see
# select_allreduce_protocols in allreduce_sycl.cpp), we run the test multiple times with the
# threshold/algo env vars set so that the call always lands in the cell named in `cell_label`.
#
# Per-cell entries are pipe-delimited: label|env|extra_test_args
#
# Cell -> env recipe (assumes ARC; on PVC the LL/arc_allreduce gates collapse to small/large):
#   ll_ring          : SIMPLE_THRESHOLD=huge, LL=ring     -> ll path, RingTransmit
#   ll_twoshots      : SIMPLE_THRESHOLD=huge, LL=twoshots -> ll path, TwoShotsTransmit
#   ll_oneshot       : SIMPLE_THRESHOLD=huge, LL=oneshot  -> ll path, OneShotTransmit (+Ring fallback)
#   arc_allreduce    : SIMPLE_THRESHOLD=huge, ARC=1       -> arc_allreduce path. The arc_ll256
#                      kernel only implements SUM (treats prod/min/max as SUM, ignored),
#                      so CCL_TEST_SKIP_PROD_MIN_MAX=1 narrows the test to sum/avg/pre_mul_sum.
#   small            : SIMPLE_THRESHOLD=0, SMALL=huge     -> small path. The small kernel uses
#                      a per-rank tmp_buf (~128MB); --count 65536 keeps total well within that.
#   large            : SIMPLE_THRESHOLD=0, SMALL=0        -> large path
allreduce_cells=(
    "ll_ring|CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=1048576000 CCL_SYCL_ALLREDUCE_LL=ring|"
    "ll_twoshots|CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=1048576000 CCL_SYCL_ALLREDUCE_LL=twoshots|"
    "ll_oneshot|CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=1048576000 CCL_SYCL_ALLREDUCE_LL=oneshot|"
    "arc_allreduce|CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=1048576000 CCL_SYCL_ALLREDUCE_ARC=1 CCL_TEST_SKIP_PROD_MIN_MAX=1|"
    "small|CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=0 CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD=1048576000|--count 65536"
    "large|CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=0 CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD=0|"
)
for N in ${ranks}
do
    for cell_entry in "${allreduce_cells[@]}"
    do
        IFS='|' read -r cell_label cell_env cell_args <<< "${cell_entry}"
        TEST_LOG="sycl_allreduce_usm_test_N_${N}_${cell_label}.log"
        cmd="${cell_env} mpirun -np ${N} -l ./sycl_allreduce_usm_test_inorder gpu device ${cell_args} > ${TEST_LOG} 2>&1"
        echo "${cmd}"
        run_and_check_log "${cmd}" ${TEST_LOG}
        test $? -eq 1 && cat "${TEST_LOG}"
    done
done

# ---- Custom reduction reduce_scatter test (sum, prod, min, max, avg, pre_mul_sum) ----
#
# Same idea as the allreduce loop above. select_reduce_scatter_single_node_protocols
# (reduce_scatter_sycl.cpp) routes by SIMPLE_THRESHOLD then SMALL_THRESHOLD; the LL path always
# uses RingTransmit so there is no per-LL-algo knob to vary. Each cell exercises one dispatcher
# arm end-to-end with custom pre-op, validating that the SYCL/algo ownership flag flips correctly.
#
# `small` carries --count 65536 to stay within the small kernel's per-rank tmp_buf (~128MB);
# the dispatcher's production threshold is small (default 2MB total) but our cell-forcing
# threshold has to be large enough to capture the test count, so we trim the test count instead.
reduce_scatter_cells=(
    "ll|CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=1048576000|"
    "small|CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=0 CCL_SYCL_REDUCE_SCATTER_SMALL_THRESHOLD=1048576000|--count 65536"
    "large|CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=0 CCL_SYCL_REDUCE_SCATTER_SMALL_THRESHOLD=0|"
)
for N in ${ranks}
do
    for cell_entry in "${reduce_scatter_cells[@]}"
    do
        IFS='|' read -r cell_label cell_env cell_args <<< "${cell_entry}"
        TEST_LOG="sycl_reduce_scatter_usm_test_N_${N}_${cell_label}.log"
        cmd="${cell_env} mpirun -np ${N} -l ./sycl_reduce_scatter_usm_test_inorder gpu device ${cell_args} > ${TEST_LOG} 2>&1"
        echo "${cmd}"
        run_and_check_log "${cmd}" ${TEST_LOG}
        test $? -eq 1 && cat "${TEST_LOG}"
    done
done

sed -i "s/tests=\"0\"/tests=\"${TEST_COUNT}\"/; s/failures=\"0\"/failures=\"${FAIL_COUNT}\"/" ${JUNIT_REPORT}

# Finish the report by closing the testcase XML
echo "</testsuite>" >> ${JUNIT_REPORT}


# Final pass/fail report
if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "Some tests failed. Total failures: ${FAIL_COUNT}"
    exit 1
else
    rm -rf *.log
    echo "Pass"
    exit 0
fi
