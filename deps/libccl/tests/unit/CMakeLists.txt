cmake_minimum_required(VERSION 3.20)

# Unit tests for pieces of the runtime. Targets link against the `ccl`
# library so tests exercise the code as it ships (matching the functional
# tests' approach).

project(oneCCL_unit_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GoogleTest is vendored alongside the other tests.
set(GTEST_DIR ${PROJECT_SOURCE_DIR}/../googletest-release-1.8.1)
if(NOT TARGET gtest)
    add_subdirectory(${GTEST_DIR} gtest_build)
endif()

# Standalone build (unit tests invoked without the parent libccl build):
# fall back to the installed shared library, mirroring tests/functional.
if(NOT TARGET ccl)
    add_library(ccl INTERFACE)
    target_link_libraries(ccl INTERFACE -l:libccl.so.1)
endif()

set(POLICY_JSON_PATH "${PROJECT_SOURCE_DIR}/../../policies/collectives_policy.json")

set(LIBCCL_SRC_DIR "${PROJECT_SOURCE_DIR}/../../src")

add_executable(mini_json_test mini_json_test.cpp)
target_include_directories(mini_json_test PRIVATE
    ${LIBCCL_SRC_DIR}
    ${GTEST_DIR}/include)
target_compile_definitions(mini_json_test PRIVATE
    MINI_JSON_POLICY_PATH="${POLICY_JSON_PATH}")
target_link_libraries(mini_json_test PRIVATE gtest gtest_main pthread ccl)

enable_testing()
add_test(NAME mini_json_test COMMAND mini_json_test)

install(TARGETS mini_json_test RUNTIME DESTINATION tests/unit OPTIONAL)
