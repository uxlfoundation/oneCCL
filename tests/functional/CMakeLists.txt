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
cmake_minimum_required (VERSION 3.20)

file(GLOB sources "*_test.c" "*_test.cpp")

set(PROJECT_NAME "oneCCL functional tests")
project(${PROJECT_NAME})

message(STATUS "FT CMAKE_PROJECT_NAME: ${CMAKE_PROJECT_NAME}")
message(STATUS "FT PROJECT_NAME: ${PROJECT_NAME}")

#set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

#set default proc maps
if (NOT PROC_MAPS)
    set(PROC_MAPS "2:1,2")
endif()

message(STATUS "PROC_MAPS: ${PROC_MAPS}")

# standalone build
if (${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set(COMMON_CMAKE_DIR ${PROJECT_SOURCE_DIR}/../../cmake)
    include(${COMMON_CMAKE_DIR}/helpers.cmake)
    set_lp_env()
    define_compute_backend()
    if (COMPUTE_BACKEND)
        set_compute_backend(${COMMON_CMAKE_DIR})
    endif()
endif()

set(SERVICE_SRC
    conf.cpp
    lp.cpp
    transport.cpp)

if (DEFINED ENV{CCL_ROOT})
    set(CCL_ROOT "$ENV{CCL_ROOT}")
endif()

set(CCL_INSTALL_TESTS "$ENV{PWD}")
enable_testing()

set(GTEST_DIR ${PROJECT_SOURCE_DIR}/../googletest-release-1.8.1/)
add_subdirectory(${GTEST_DIR} gtest_build)
set(EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/../../examples)

set(INC_DIRS
    ${GTEST_DIR}/include
    ${GTEST_DIR}/src
    ${EXAMPLES_DIR}/include)

include_directories(${INC_DIRS})

message(STATUS "FT build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "FT CCL_ROOT: ${CCL_ROOT}")
message(STATUS "FT INC_DIRS: ${INC_DIRS}")

message(STATUS "FT COMPUTE_BACKEND: ${COMPUTE_BACKEND}")

if (${CMAKE_VERSION} VERSION_LESS 3.1)
#cmake version below 3.1 does not support CMAKE_C[XX}_STANDARD flags
#set manually
# TODO: Need to handle c++17 option for older cmake
    set(CXX_COMPILER_FLAGS "-std=gnu++11")
    set(C_COMPILER_FLAGS "-std=gnu99")
endif()

#common release/debug compilation settings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_COMPILER_FLAGS} -Wall -Wextra -Wno-unused-parameter -Werror -D_GNU_SOURCE -fvisibility=internal")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${C_COMPILER_FLAGS} -O0 -g -DENABLE_DEBUG")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${C_COMPILER_FLAGS} -O3")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${C_COMPILER_FLAGS} -O2 -g")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COMPILER_FLAGS} -Wall -Wextra -Wno-unused-parameter -Werror -D_GNU_SOURCE -fvisibility=internal")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_COMPILER_FLAGS} -O0 -g -DENABLE_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_COMPILER_FLAGS} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CXX_COMPILER_FLAGS} -O2 -g")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" OR(${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM"))
    set(CMAKE_CXX_STANDARD 17)
endif()

if (COMPUTE_BACKEND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPUTE_BACKEND_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${COMPUTE_BACKEND_LIBRARIES}")
endif()

link_directories(${EXAMPLES_LIB_DIRS})

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src} ${SERVICE_SRC})
    if (${executable} MATCHES ".*reduce.*")
        target_compile_definitions(${executable} PRIVATE TEST_CCL_REDUCE)
    endif()
    if (${executable} MATCHES ".*bcast.*")
        target_compile_definitions(${executable} PRIVATE TEST_CCL_BCAST)
    endif()
    if (${executable} MATCHES ".*broadcast.*")
        target_compile_definitions(${executable} PRIVATE TEST_CCL_BCAST)
    endif()
    target_include_directories(${executable} PRIVATE ${EXAMPLES_INC_DIRS})
    target_link_libraries(${executable} PUBLIC gtest_main)
    target_link_libraries(${executable} PUBLIC gtest)
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PUBLIC dl)
    # w/a for case with 2 mpi libs, should be fixed
    if (DEFINED ENV{I_MPI_ROOT})
        set(I_MPI_ROOT "$ENV{I_MPI_ROOT}")
    endif()
    target_link_libraries(${executable} PUBLIC ccl)
    target_link_libraries(${executable} PUBLIC -L${I_MPI_ROOT}/lib/release/)
    target_link_libraries(${executable} PUBLIC mpi)
    target_link_libraries(${executable} PUBLIC ${COMPUTE_BACKEND_TARGET_NAME})
    install(TARGETS ${executable} RUNTIME DESTINATION ${CCL_INSTALL_TESTS} OPTIONAL)
    add_test (NAME ${executable} CONFIGURATIONS default COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/${executable} --gtest_output=xml:${CCL_INSTALL_TESTS}/${executable}_default_report.junit.xml)
endforeach()

add_test (NAME allreduce_fusion CONFIGURATIONS allreduce_fusion COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allreduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allreduce_fusion_report.junit.xml)

foreach(proc_map ${PROC_MAPS})

    if (${proc_map} MATCHES "(.*):(.*)")
        set(N ${CMAKE_MATCH_1})
        set(PPNS ${CMAKE_MATCH_2})
        string(REGEX REPLACE "," ";" PPNS ${PPNS})
    else()
        message(FATAL_ERROR "Incorrect proc_map input: ${proc_map}, expected input <n:ppn>")
    endif()

    foreach(ppn ${PPNS})

        foreach(algo direct; naive; flat; multi_bcast; topo)
            add_test (NAME allgather_${algo}_${N}_${ppn} CONFIGURATIONS allgather_${algo}_${N}_${ppn} COMMAND mpiexec.hydra -l -n ${N} -ppn ${ppn} ${CCL_INSTALL_TESTS}/allgather_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allgather_${algo}_${N}_${ppn}_report.junit.xml)
        endforeach()

        foreach(algo direct; naive; flat; multi_bcast; topo)
            add_test (NAME allgatherv_${algo}_${N}_${ppn} CONFIGURATIONS allgatherv_${algo}_${N}_${ppn} COMMAND mpiexec.hydra -l -n ${N} -ppn ${ppn} ${CCL_INSTALL_TESTS}/allgatherv_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allgatherv_${algo}_${N}_${ppn}_report.junit.xml)
        endforeach()

        foreach(algo direct; rabenseifner; nreduce; ring; ring_rma; double_tree; recursive_doubling; 2d; topo)
            add_test (NAME allreduce_${algo}_${N}_${ppn} CONFIGURATIONS allreduce_${algo}_${N}_${ppn} COMMAND mpiexec.hydra -l -n ${N} -ppn ${ppn} ${CCL_INSTALL_TESTS}/allreduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allreduce_${algo}_${N}_${ppn}_report.junit.xml)
        endforeach()

        foreach(algo direct; naive; scatter; topo)
            add_test (NAME alltoall_${algo}_${N}_${ppn} CONFIGURATIONS alltoall_${algo}_${N}_${ppn} COMMAND mpiexec.hydra -l -n ${N} -ppn ${ppn} ${CCL_INSTALL_TESTS}/alltoall_test --gtest_output=xml:${CCL_INSTALL_TESTS}/alltoall_${algo}_${N}_${ppn}_report.junit.xml)
        endforeach()

        foreach(algo direct; naive; scatter; topo)
            add_test (NAME alltoallv_${algo}_${N}_${ppn} CONFIGURATIONS alltoallv_${algo}_${N}_${ppn} COMMAND mpiexec.hydra -l -n ${N} -ppn ${ppn} ${CCL_INSTALL_TESTS}/alltoallv_test --gtest_output=xml:${CCL_INSTALL_TESTS}/alltoallv_${algo}_${N}_${ppn}_report.junit.xml)
        endforeach()

        foreach(algo direct; ring; double_tree; naive; topo)
            add_test (NAME bcast_${algo}_${N}_${ppn} CONFIGURATIONS bcast_${algo}_${N}_${ppn} COMMAND mpiexec.hydra -l -n ${N} -ppn ${ppn} ${CCL_INSTALL_TESTS}/bcast_test --gtest_output=xml:${CCL_INSTALL_TESTS}/bcast_${algo}_${N}_${ppn}_report.junit.xml)
        endforeach()

        foreach(algo direct; ring; double_tree; naive; topo)
            add_test (NAME broadcast_${algo}_${N}_${ppn} CONFIGURATIONS broadcast_${algo}_${N}_${ppn} COMMAND mpiexec.hydra -l -n ${N} -ppn ${ppn} ${CCL_INSTALL_TESTS}/broadcast_test --gtest_output=xml:${CCL_INSTALL_TESTS}/broadcast_${algo}_${N}_${ppn}_report.junit.xml)
        endforeach()

        foreach(algo direct; rabenseifner; ring; tree; double_tree; topo)
            add_test (NAME reduce_${algo}_${N}_${ppn} CONFIGURATIONS reduce_${algo}_${N}_${ppn} COMMAND mpiexec.hydra -l -n ${N} -ppn ${ppn} ${CCL_INSTALL_TESTS}/reduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/reduce_${algo}_${N}_${ppn}_report.junit.xml)
        endforeach()

        foreach(algo direct; ring)
            add_test (NAME reduce_scatter_${algo}_${N}_${ppn} CONFIGURATIONS reduce_scatter_${algo}_${N}_${ppn} COMMAND mpiexec.hydra -l -n ${N} -ppn ${ppn} ${CCL_INSTALL_TESTS}/reduce_scatter_test --gtest_output=xml:${CCL_INSTALL_TESTS}/reduce_scatter_${algo}_${N}_${ppn}_report.junit.xml)
        endforeach()

    endforeach()
endforeach()
