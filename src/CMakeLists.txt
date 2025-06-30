#
# Copyright 2016-2020 Intel Corporation
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
# builds CCL

set (EXTENSIONS_SRC)

if (CCL_ENABLE_SYCL AND CCL_ENABLE_ZE)
    list (APPEND EXTENSIONS_SRC

        ccl_cpp_utils.cpp

        sched/entry/ze/allreduce/ze_a2a_allreduce_entry.cpp
        sched/entry/ze/allreduce/ze_onesided_allreduce_entry.cpp
        sched/entry/ze/ze_a2a_allgatherv_entry.cpp
        sched/entry/ze/ze_a2a_reduce_scatter_entry.cpp
        sched/entry/ze/ze_a2a_pipeline_reduce_scatter_entry.cpp
        sched/entry/ze/ze_alltoallv_entry.cpp
        sched/entry/ze/ze_base_entry.cpp
        sched/entry/ze/ze_barrier_entry.cpp
        sched/entry/ze/cache/ze_cache.cpp
        sched/entry/ze/cache/ze_device_cache.cpp
        sched/entry/ze/ze_call.cpp
        sched/entry/ze/ze_cmdlist_timestamp.cpp
        sched/entry/ze/ze_cmdlist_event_signal_entry.cpp
        sched/entry/ze/ze_command.cpp
        sched/entry/ze/ze_copy_entry.cpp
        sched/entry/ze/ze_dummy_entry.cpp
        sched/entry/ze/ze_handle_exchange_entry.cpp
        sched/entry/ze/ze_kernel.cpp
        sched/entry/ze/ze_event_signal_entry.cpp
        sched/entry/ze/ze_event_wait_entry.cpp
        sched/entry/ze/ze_membarrier_entry.cpp
        sched/entry/ze/ze_onesided_reduce_entry.cpp
        sched/entry/ze/ze_primitives.cpp
        sched/entry/ze/ze_pt2pt_barrier_entry.cpp
        sched/entry/ze/ze_reduce_local_entry.cpp

        common/global/ze/ze_data.cpp
        common/global/ze/ze_fd_manager.cpp
        common/utils/sycl_utils.cpp
        common/api_wrapper/ze_api_wrapper.cpp

        sched/ze/ze_event_manager.cpp
        sched/ze/ze_handle_manager.cpp
        sched/ze/ze_ipc_event_pool_manager.cpp
        sched/ze/ze_list_manager.cpp

        coll/algorithms/allgatherv/sycl/allgatherv_sycl.cpp
        coll/algorithms/allgatherv/sycl/allgatherv_small_sycl.cpp
        coll/algorithms/allgatherv/sycl/allgatherv_medium_sycl.cpp
        coll/algorithms/allgatherv/sycl/allgatherv_large_sycl.cpp
        coll/algorithms/allgatherv/sycl/allgatherv_scaleout_sycl.cpp
        coll/algorithms/allgatherv/sycl/allgatherv_pcie.cpp

        coll/algorithms/alltoall/sycl/alltoall_sycl.cpp
        coll/algorithms/alltoall/sycl/alltoall_large_sycl.cpp

        coll/algorithms/allreduce/sycl/allreduce_small_sycl.cpp
        coll/algorithms/allreduce/sycl/allreduce_medium_sycl.cpp
        coll/algorithms/allreduce/sycl/allreduce_medium_sycl_fp16.cpp
        coll/algorithms/allreduce/sycl/allreduce_medium_sycl_bf16.cpp
        coll/algorithms/allreduce/sycl/allreduce_medium_sycl_fp32.cpp
        coll/algorithms/allreduce/sycl/allreduce_medium_sycl_int32.cpp
        coll/algorithms/allreduce/sycl/allreduce_large_sycl.cpp
        coll/algorithms/allreduce/sycl/allreduce_large_sycl_fp16.cpp
        coll/algorithms/allreduce/sycl/allreduce_large_sycl_bf16.cpp
        coll/algorithms/allreduce/sycl/allreduce_large_sycl_fp32.cpp
        coll/algorithms/allreduce/sycl/allreduce_large_sycl_int32.cpp
        coll/algorithms/allreduce/sycl/allreduce_sycl.cpp
        coll/algorithms/allreduce/sycl/allreduce_scaleout_sycl.cpp
        coll/algorithms/allreduce/sycl/allreduce_ring_ll256.cpp
        coll/algorithms/allreduce/sycl/allreduce_pcie.cpp

        coll/algorithms/reduce_scatter/sycl/reduce_scatter_sycl.cpp
        coll/algorithms/reduce_scatter/sycl/reduce_scatter_small_sycl.cpp
        coll/algorithms/reduce_scatter/sycl/reduce_scatter_medium_sycl.cpp
        coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl.cpp
        coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_int32.cpp
        coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_fp32.cpp
        coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_fp16.cpp
        coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_bf16.cpp
        coll/algorithms/reduce_scatter/sycl/reduce_scatter_scaleout_sycl.cpp

        coll/algorithms/broadcast/sycl/broadcast_sycl.cpp
        coll/algorithms/broadcast/sycl/broadcast_small_sycl.cpp
        coll/algorithms/broadcast/sycl/broadcast_large_sycl.cpp
        coll/algorithms/broadcast/sycl/broadcast_scaleout_sycl.cpp

        coll/algorithms/recv/sycl/recv_sycl.cpp
        coll/algorithms/send/sycl/send_sycl.cpp

        coll/algorithms/utils/sycl_coll_base.cpp
        coll/algorithms/utils/sycl_pipe_send.cpp
        coll/algorithms/utils/sycl_selection.cpp
        coll/algorithms/utils/sycl_ll256.cpp
    )
    if (ENABLE_UMF)
        list(APPEND EXTENSIONS_SRC
            umf/ipc.cpp
        )
    endif(ENABLE_UMF)
endif(CCL_ENABLE_SYCL AND CCL_ENABLE_ZE)

set(CCL_SRC

    atl/atl_base_comm.cpp
    atl/atl_base_transport.cpp
    atl/atl_def.cpp
    atl/mpi/atl_mpi.cpp
    atl/mpi/atl_mpi_comm.cpp
    atl/mpi/atl_mpi_ctx.cpp
    atl/ofi/atl_ofi.cpp
    atl/ofi/atl_ofi_comm.cpp
    atl/ofi/atl_ofi_helper.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable_simple.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable_simple_internal.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs_keeper.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs_server.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.cpp
    atl/util/pm/pmi_rt/pmi_simple.cpp
    atl/util/pm/pmi_rt/pmi/simple_pmi.c
    atl/util/pm/pmi_rt/pmi/simple_pmiutil.c

    coll/attr/ccl_common_op_attrs.cpp
    coll/attr/ccl_allgather_op_attr.cpp
    coll/attr/ccl_allgatherv_op_attr.cpp
    coll/attr/ccl_allreduce_op_attr.cpp
    coll/attr/ccl_alltoall_op_attr.cpp
    coll/attr/ccl_alltoallv_op_attr.cpp
    coll/attr/ccl_barrier_op_attr.cpp
    coll/attr/ccl_bcast_op_attr.cpp
    coll/attr/ccl_pt2pt_op_attr.cpp
    coll/attr/ccl_reduce_op_attr.cpp
    coll/attr/ccl_reduce_scatter_op_attr.cpp
    coll/coll_param.cpp
    coll/coll_util.cpp
    coll/algorithms/allgather.cpp
    coll/algorithms/allgatherv/allgatherv.cpp
    coll/algorithms/allreduce/allreduce.cpp
    coll/algorithms/allreduce/allreduce_rma.cpp
    coll/algorithms/algorithm_utils.cpp
    coll/algorithms/alltoall/alltoall.cpp
    coll/algorithms/alltoallv.cpp
    coll/algorithms/barrier/barrier.cpp
    coll/algorithms/broadcast/bcast.cpp
    coll/algorithms/broadcast/broadcast.cpp
    coll/algorithms/double_tree_ops.cpp
    coll/algorithms/recv/recv.cpp
    coll/algorithms/reduce.cpp
    coll/algorithms/reduce_scatter/reduce_scatter.cpp
    coll/algorithms/send/send.cpp
    coll/coll.cpp
    coll/coll_check.cpp
    coll/group/group.cpp
    coll/selection/selection.cpp
    coll/selection/selector_allgather.cpp
    coll/selection/selector_allgatherv.cpp
    coll/selection/selector_allreduce.cpp
    coll/selection/selector_alltoall.cpp
    coll/selection/selector_alltoallv.cpp
    coll/selection/selector_barrier.cpp
    coll/selection/selector_bcast.cpp
    coll/selection/selector_recv.cpp
    coll/selection/selector_reduce.cpp
    coll/selection/selector_reduce_scatter.cpp
    coll/selection/selector_send.cpp

    comm/atl_tag.cpp
    comm/mt_comm.cpp
    comm/comm.cpp
    comm/comm_selector.cpp

    common/context/context.cpp
    common/datatype/datatype.cpp
    common/device/device.cpp
    common/env/env.cpp
    common/env/env_parser.cpp
    common/event/ccl_event.cpp
    common/event/impls/host_event.cpp
    common/event/impls/native_event.cpp
    common/framework/framework.cpp
    common/global/global.cpp
    common/log/log.cpp
    common/request/request.cpp
    common/stream/stream.cpp
    common/utils/exchange_utils.cpp
    common/utils/fd_info.cpp
    common/utils/memcpy.cpp
    common/utils/profile.cpp
    common/utils/spinlock.cpp
    common/utils/utils.cpp
    common/utils/version.cpp
    common/utils/yield.cpp
    common/utils/comm_profile.cpp

    common/api_wrapper/api_wrapper.cpp
    common/api_wrapper/mpi_api_wrapper.cpp
    common/api_wrapper/ofi_api_wrapper.cpp
    common/api_wrapper/pmix_api_wrapper.cpp
    common/api_wrapper/umf_api_wrapper.cpp
    common/api_wrapper/openmp_wrapper.cpp

    comp/bf16/bf16.cpp
    comp/bf16/bf16_intrisics.cpp
    comp/comp.cpp
    comp/fp16/fp16.cpp
    comp/fp16/fp16_intrisics.cpp

    exec/exec.cpp
    exec/thread/base_thread.cpp
    exec/thread/listener.cpp
    exec/thread/service_worker.cpp
    exec/thread/worker.cpp

    fusion/fusion.cpp

    hwloc/hwloc_wrapper.cpp

    native_device_api/sycl/export.cpp

    parallelizer/parallelizer.cpp

    sched/buffer/buffer_cache.cpp
    sched/buffer/buffer_manager.cpp
    sched/cache/cache.cpp
    sched/cache/key.cpp
    sched/cache/recycle_storage.cpp
    sched/entry/coll/coll_entry.cpp
    sched/entry/copy/copy_entry.cpp
    sched/entry/copy/copy_helper.cpp
    sched/entry/deps_entry.cpp
    sched/entry/entry.cpp
    sched/entry/factory/chunked_entry_factory.cpp
    sched/entry/recv_copy_entry.cpp
    sched/entry/reduce_local_entry.cpp
    sched/queue/flow_control.cpp
    sched/queue/queue.cpp
    sched/queue/strict_queue.cpp
    sched/sched.cpp
    sched/sched_base.cpp
    sched/sched_group.cpp
    sched/sched_restart_manager.cpp
    sched/sched_timer.cpp

    topology/mt_topo_manager.cpp
    topology/topo_manager.cpp

    unordered_coll/unordered_coll.cpp

    ccl_api_functions.cpp
    ccl_app_api_coll_attr.cpp
    ccl_app_api_comm_attr.cpp
    ccl_app_api_comm_split_attr.cpp
    ccl_app_api_datatype_attr.cpp
    ccl_app_api_event.cpp
    ccl_app_api_init_attr.cpp
    ccl_app_api_kvs_attr.cpp
    ccl_cpp_communicator.cpp
    ccl_cpp_context.cpp
    ccl_cpp_device.cpp
    ccl_cpp_environment.cpp
    ccl_cpp_kvs.cpp
    ccl_cpp_stream.cpp
    ccl_cpp_utils.cpp
    ccl_empty_attr.cpp
    ccl_empty_coll_attr.cpp
    ccl_empty_comm_attr.cpp
    ccl_empty_comm_split_attr.cpp
    ccl_empty_init_attr.cpp
    ccl_empty_kvs_attr.cpp
    ccl_empty_stream.cpp

    ${EXTENSIONS_SRC})

if (ENABLE_STUB_BACKEND)
    set (CCL_SRC
            comm/stub_comm.cpp
            stub_kvs_impl.cpp
            ${CCL_SRC})
endif()

set(SRC_C_FLAGS)
set(SRC_CXX_FLAGS)
set(SRC_SHARED_LINKER_FLAGS)
set(SRC_INCLUDE_DIRS)
set(SRC_LINK_DIRS)
set(SRC_LINK_LIBS)

if (USE_SECURITY_FLAGS)
    set(SRC_C_FLAGS "${SRC_C_FLAGS} -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector")
    set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector")
    set(SRC_SHARED_LINKER_FLAGS "${SRC_SHARED_LINKER_FLAGS} -fPIE -fPIC -z noexecstack -z relro -z now")
    if (${CMAKE_C_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            set(SRC_C_FLAGS "${SRC_C_FLAGS} -fstack-protector-strong")
            set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -fstack-protector-strong")
        endif()
    endif()
endif()

if (ENABLE_LINKER_RUNPATH)
    set(SRC_SHARED_LINKER_FLAGS "${SRC_SHARED_LINKER_FLAGS} -Wl,--enable-new-dtags -Wl,-rpath='$ORIGIN'")
endif()

set(SRC_SHARED_LINKER_FLAGS "${SRC_SHARED_LINKER_FLAGS} -Wl,--version-script=${PROJECT_SOURCE_DIR}/ccl.map")

if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    # To suppress for 'offsetof applied to non-POD (Plain Old Data) types is nonstandar'
    set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -diag-disable=1875")
    # To suppress "overloaded virtual function is only partially overridden in class", as we intentially
    # don't override some base class' overloads in order to keep a default implementation for them
    set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -diag-disable=654")
endif()

if ((${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") OR (${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM"))
    if (USE_CODECOV_FLAGS)
        set(SRC_C_FLAGS "${SRC_C_FLAGS} -fprofile-instr-generate -fcoverage-mapping -Wno-error=unused-command-line-argument -Wno-error=option-ignored -fno-sycl-use-footer")
        set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping -Wno-error=unused-command-line-argument -Wno-error=option-ignored -fno-sycl-use-footer")
    endif()
endif()

list(APPEND SRC_INCLUDE_DIRS
     ${PROJECT_SOURCE_DIR}/include
     ${PROJECT_SOURCE_DIR}/src
     ${PROJECT_SOURCE_DIR}/src/atl
     ${LEVEL_ZERO_INCLUDE_DIR}
     ${LIBFABRIC_INCLUDE_DIR}
     ${HWLOC_INCLUDE_DIR}
     ${ITT_INCLUDE_DIR})

if (ENABLE_DRM)
    list(APPEND SRC_SYSTEM_INCLUDE_DIRS
         ${DRM_INCLUDE_DIR})
endif()

if (ENABLE_PMIX)
    list(APPEND SRC_INCLUDE_DIRS
         ${PMIX_INCLUDE_DIR})
endif()

if (ENABLE_UMF)
    list(APPEND SRC_INCLUDE_DIRS
         ${UMF_INCLUDE_DIR})
endif()

list(APPEND SRC_LINK_LIBS
     dl
     pthread
     ${EXTERNAL_LIBS}
     ${HWLOC_LIB_DIR}/libhwloc.a
     ${ITT_LIB_DIR}/libittnotify.a)

if (ENABLE_MPI)
    set(SRC_C_FLAGS "${SRC_C_FLAGS} -DCCL_ENABLE_MPI")
    set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -DCCL_ENABLE_MPI")
    list(APPEND SRC_INCLUDE_DIRS ${MPI_INCLUDE_DIR})
endif()

link_directories(${SRC_LINK_DIRS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SRC_C_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SRC_CXX_FLAGS} -pthread")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SRC_SHARED_LINKER_FLAGS}")

# special library that holds objects only
add_library(ccl-objects OBJECT ${CCL_SRC})
set_target_properties(ccl-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_include_directories(ccl-objects PRIVATE ${SRC_INCLUDE_DIRS})
target_include_directories(ccl-objects SYSTEM PRIVATE ${SRC_SYSTEM_INCLUDE_DIRS})

if (COMPUTE_BACKEND_TARGET_NAME)
    target_include_directories(ccl-objects PUBLIC ${INTERFACE_INCLUDE_DIRECTORIES})
endif()

# shared library
add_library(ccl SHARED $<TARGET_OBJECTS:ccl-objects>)
target_include_directories(ccl PUBLIC ${SRC_INCLUDE_DIRS})
target_link_libraries(ccl PRIVATE ${SRC_LINK_LIBS})

set_target_properties(ccl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
if (NOT LIB_SO_VERSION AND NOT LIB_MAJOR_VERSION)
    set(DEFAULT_LIB_SO_VERSION "1.0")
    set(DEFAULT_LIB_MAJOR_VERSION "1")
    set_target_properties(ccl PROPERTIES VERSION ${DEFAULT_LIB_SO_VERSION})
    set_target_properties(ccl PROPERTIES SOVERSION ${DEFAULT_LIB_MAJOR_VERSION})
else()
    set_target_properties(ccl PROPERTIES VERSION ${LIB_SO_VERSION})
    set_target_properties(ccl PROPERTIES SOVERSION ${LIB_MAJOR_VERSION})
endif()

message(STATUS "SRC C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "SRC CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "SRC SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "SRC INCLUDE_DIRS: ${SRC_INCLUDE_DIRS}")
message(STATUS "SRC LINK_DIRS: ${SRC_LINK_DIRS}")
message(STATUS "SRC LINK_LIBS: ${SRC_LINK_LIBS}")

install(TARGETS ccl LIBRARY DESTINATION ${CCL_INSTALL_LIB})
install(FILES
    "${PROJECT_SOURCE_DIR}/cmake/FindIntelSYCL.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/FindIntelSYCL_level_zero.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/FindNUMA.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/FindSYCL.cmake"
    DESTINATION ${CCL_INSTALL_LIB})

# static library
add_library(ccl-static STATIC $<TARGET_OBJECTS:ccl-objects>)
set_target_properties(ccl-static PROPERTIES OUTPUT_NAME ccl)
set_target_properties(ccl-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS ccl-static ARCHIVE DESTINATION ${CCL_INSTALL_LIB} OPTIONAL)

# API headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CCL_INSTALL_INCLUDE} FILES_MATCHING REGEX ".*\\.(h|hpp)$")

if ("${LIBFABRIC_DIR}" STREQUAL "")
    # internal libfabric is used, install it into package
    install(DIRECTORY ${DEPS_DIR}/ofi/lib/
            DESTINATION ${CMAKE_INSTALL_PREFIX}/opt/mpi/libfabric/lib)
endif()

if (ENABLE_MPI)
    file(GLOB mpi_bins "${DEPS_DIR}/mpi/bin/*")
    install(PROGRAMS ${mpi_bins}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/opt/mpi/bin)
    install(DIRECTORY ${DEPS_DIR}/mpi/include/
            DESTINATION ${CMAKE_INSTALL_PREFIX}/opt/mpi/include)
    install(DIRECTORY ${DEPS_DIR}/mpi/lib/
           DESTINATION ${CMAKE_INSTALL_PREFIX}/opt/mpi/lib)
    install(DIRECTORY ${DEPS_DIR}/mpi/opt/mpi/etc/
            DESTINATION ${CMAKE_INSTALL_PREFIX}/opt/mpi/etc/)
    install(DIRECTORY ${DEPS_DIR}/mpi/licensing/
            DESTINATION ${CCL_INSTALL_LICENSE}/mpi/)
endif()

add_subdirectory(ext)
