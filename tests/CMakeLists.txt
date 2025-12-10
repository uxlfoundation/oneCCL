# Copyright 2016-2025 Intel Corporation
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

set(INSTALL_GMOCK ON CACHE BOOL OFF)
set(INSTALL_GTEST ON CACHE BOOL OFF)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.12.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL
)

# Make available GoogleTest's CMake targets
FetchContent_MakeAvailable(googletest)

include(CMakeParseArguments)

function(add_test_internal)
    cmake_parse_arguments(
        ARGS # Prefix for parsed arguments
        "" # Boolean arguments (none in this case)
        "TARGET;EXEC_PREFIX;TIMEOUT" # Mono-valued arguments
        "SRCS;PLUGIN_TARGETS;VALUATIONS" # Multi-valued arguments
        ${ARGN}
    )

    # Ensure that TARGET and SRCS are provided as they are essential
    if(NOT ARGS_TARGET)
        message(FATAL_ERROR "A TARGET must be specified.")
    endif()
    if(NOT ARGS_SRCS)
        message(FATAL_ERROR "Source files must be specified in SRCS.")
    endif()

    # Check PLUGIN_TARGETS availability
    foreach(plugin_target ${ARGS_PLUGIN_TARGETS})
        if(NOT TARGET ${plugin_target})
            message(WARNING "Skipping test for ${ARGS_TARGET}: required plugin target '${plugin_target}' is not available.")
            return()
        endif()
    endforeach()

    # Add an executable for the test
    add_executable(${ARGS_TARGET} ${ARGS_SRCS})

    # Find threads
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    # Find the MPI package
    find_package(MPI REQUIRED)

    # Link necessary libraries
    target_link_libraries(${ARGS_TARGET} PRIVATE
        oneCCL2
        ccl_null
        ittnotify
        gtest
        gtest_main
        MPI::MPI_CXX
        Threads::Threads
    )

    target_include_directories(${ARGS_TARGET} PRIVATE "${PROJECT_SOURCE_DIR}/src")

    # Add coverage
    target_compile_options(${ARGS_TARGET} PRIVATE "${COVERAGE_CXX_FLAGS}")
    target_link_libraries(${ARGS_TARGET} PRIVATE "${COVERAGE_CXX_FLAGS}")

    # Add common compatiblity flags
    target_compile_options(${ARGS_TARGET} PRIVATE "${COMMON_CXX_FLAGS}")

    # Split EXEC_PREFIX into individual arguments, like "mpiexec -n 2" into "mpiexec" "-n" "2"
    if(ARGS_EXEC_PREFIX)
        separate_arguments(ARGS_EXEC_PREFIX)
    endif()

    # Prepare the command with execution prefix if specified
    set(test_command ${ARGS_EXEC_PREFIX} ${PROJECT_SOURCE_DIR}/utils/debug_hook.sh $<TARGET_FILE:${ARGS_TARGET}>)
    add_test(NAME ${ARGS_TARGET} COMMAND ${test_command} CONFIGURATIONS unit_tests)

    # Handle Plugin Targets
    foreach(plugin_target ${ARGS_PLUGIN_TARGETS})
        set(ENV_MODIFICATION "${ENV_MODIFICATION}LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:${plugin_target}>;")
    endforeach()

    set_property(TEST ${ARGS_TARGET} PROPERTY ENVIRONMENT_MODIFICATION "${ENV_MODIFICATION}")
    set_tests_properties(${ARGS_TARGET} PROPERTIES TIMEOUT ${ARGS_TIMEOUT})

    if(${ONECCL_VALGRIND_TESTING} AND ${ARGS_VALUATIONS})
        add_test(NAME ${ARGS_TARGET}_valgrind COMMAND valgrind --tool=memcheck --leak-check=full --show-reachable=yes --error-exitcode=1 ${test_command} CONFIGURATIONS unit_tests)
        set_property(TEST ${ARGS_TARGET}_valgrind PROPERTY ENVIRONMENT_MODIFICATION "${ENV_MODIFICATION}")
    endif()

    if(${ONECCL_HELGRIND_TESTING} AND ${ARGS_VALUATIONS})
        add_test(NAME ${ARGS_TARGET}_helgrind COMMAND valgrind --tool=helgrind --error-exitcode=1 ${test_command} CONFIGURATIONS unit_tests)
        set_property(TEST ${ARGS_TARGET}_helgrind PROPERTY ENVIRONMENT_MODIFICATION "${ENV_MODIFICATION}")
    endif()
endfunction()

add_test_internal(
    TARGET version_api
    SRCS version_api.cpp
    PLUGIN_TARGETS "ccl_null"
    VALUATIONS ON
    EXEC_PREFIX ""
    TIMEOUT 10
)

add_test_internal(
    TARGET null_smoke
    SRCS null_api.cpp
    PLUGIN_TARGETS "ccl_null"
    VALUATIONS ON
    EXEC_PREFIX ""
    TIMEOUT 30
)

# Temporarly disenable valgrind checks for the test
add_test_internal(
    TARGET null_concurrent
    SRCS concurrent_execution.cpp
    PLUGIN_TARGETS "ccl_null"
    VALUATIONS OFF
    EXEC_PREFIX "mpiexec -n 2"
    TIMEOUT 30
)

if(IntelSYCL_FOUND)
    add_test_internal(
        TARGET acceptance_sycl
        SRCS acceptance_sycl.cpp
        PLUGIN_TARGETS "ccl_legacy"
        VALUATIONS ON
        EXEC_PREFIX "mpiexec -n 2"
        TIMEOUT 1200
    )

    if(TARGET acceptance_sycl)
        # acceptance_sycl could be set up correctly
        add_sycl_to_target(TARGET acceptance_sycl SOURCES acceptance_sycl.cpp)

        get_property(ENV_MODIFICATION TEST acceptance_sycl PROPERTY ENVIRONMENT_MODIFICATION)
        set(ENV_MODIFICATION "${ENV_MODIFICATION}CCL_KERNEL_PATH=set:${PROJECT_SOURCE_DIR}/deps/libccl/src/kernels/;")
        set_property(TEST acceptance_sycl PROPERTY ENVIRONMENT_MODIFICATION "${ENV_MODIFICATION}")
    endif()
endif()

if(NOT ${ONECCL_USE_SYSTEM_LIBCCL})
    # This test is only enabled if we built libccl internally,
    # because it adds libccl.so.2 as dependency of libccl.so.1
    add_test(NAME both_apis COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/both_apis.sh ${CMAKE_CXX_COMPILER} CONFIGURATIONS unit_tests_host)

    set(ENV_MODIFICATION "${ENV_MODIFICATION}LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_null>;")
    set_property(TEST both_apis PROPERTY ENVIRONMENT_MODIFICATION "${ENV_MODIFICATION}")
endif()