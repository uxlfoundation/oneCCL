cmake_minimum_required (VERSION 3.20)

set(PROJECT_NAME "oneCCL regression tests")
project(${PROJECT_NAME})

message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

if(NOT TARGET ccl)
	add_library(ccl INTERFACE)
	target_link_libraries(ccl INTERFACE -l:libccl.so.1)
endif()

# include helpers if we build in standalone mode
if (${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set(COMMON_CMAKE_DIR ${PROJECT_SOURCE_DIR}/../../cmake)
    include(${COMMON_CMAKE_DIR}/helpers.cmake)
    define_compute_backend()
endif()

if (NOT PV_ENVIRONMENT)
    set(EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/../../examples)
else()
    set(EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/../examples)
endif()

set(EXAMPLES_INC_DIRS ${EXAMPLES_DIR}/include)
message(STATUS "EXAMPLES_INC_DIRS: ${EXAMPLES_INC_DIRS}")
message(STATUS "EXAMPLES_LIB_DIRS: ${EXAMPLES_LIB_DIRS}")
include_directories(${EXAMPLES_INC_DIRS})
link_directories(${EXAMPLES_LIB_DIRS})

set(REG_TESTS_DIR ${PROJECT_SOURCE_DIR})
set(REG_TESTS_INC_DIR ${REG_TESTS_DIR}/include)
message(STATUS "REG_TESTS_INC_DIR: ${REG_TESTS_INC_DIR}")
include_directories(${REG_TESTS_INC_DIR})

if (DEFINED ENV{I_MPI_ROOT})
    set(I_MPI_ROOT "$ENV{I_MPI_ROOT}")
endif()

if (DEFINED ENV{CCL_ROOT})
    set(CCL_ROOT "$ENV{CCL_ROOT}")
endif()
message(STATUS "CCL_ROOT: ${CCL_ROOT}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${CMAKE_VERSION} VERSION_LESS 3.1)
#cmake version below 3.1 does not support CMAKE_C[XX}_STANDARD flags
#set manually
    set(CXX_COMPILER_FLAGS "-std=gnu++11")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COMPILER_FLAGS}")
if ("${COMPUTE_BACKEND}" STREQUAL "dpcpp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
endif()


if (${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" OR(${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM"))
    set(CMAKE_CXX_STANDARD 17)
endif()

add_subdirectory(common)

add_subdirectory(cpu)
add_subdirectory(cpu/pt2pt)

message(STATUS "COMPUTE_BACKEND: ${COMPUTE_BACKEND}")
if ("${COMPUTE_BACKEND}" STREQUAL "dpcpp")
    add_subdirectory(sycl)
    add_subdirectory(sycl/pt2pt)
    add_subdirectory(sycl/inference)
endif()
