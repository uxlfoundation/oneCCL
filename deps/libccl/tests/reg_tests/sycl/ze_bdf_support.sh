SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

proc_counts="2 4"

colls="allgatherv,allreduce,alltoallv,bcast,reduce"

drm_bdf_support_modes="0 1"
imm_cmd_lists="0 1"

bench_options="-w 1 -i 1 -j off -b sycl -y 0,1,2,3,7,8,16,17,64,133,1077,16384,65539,131072,133073"
bench_options+=" -c all -d int8,int32,float16,float32,bfloat16"

for proc_count in ${proc_counts}
do
    for drm_bdf_support_mode in ${drm_bdf_support_modes}
    do
        for imm_cmd_list in ${imm_cmd_lists}
        do
            cmd="CCL_ZE_DRM_BDF_SUPPORT=${drm_bdf_support_mode}"
            cmd+=" SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=${imm_cmd_list}"
            cmd+=" ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE"
            cmd+=" CCL_ALLGATHERV=topo"
            cmd+=" CCL_ALLREDUCE=topo"
            cmd+=" CCL_ALLTOALLV=topo"
            cmd+=" CCL_CCL_BCAST=topo"
            cmd+=" CCL_REDUCE=topo"
            cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark"
            cmd+=" ${bench_options} -l ${colls}"
            cmd+=" > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
