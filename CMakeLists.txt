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
#common CMakeList.txt to build CCL, ATL, tests

cmake_minimum_required (VERSION 3.20)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(PROJECT_NAME "oneCCL")
set(PROJECT_FULL_NAME "oneAPI Collective Communications Library")

project(${PROJECT_NAME})

include(${PROJECT_SOURCE_DIR}/cmake/helpers.cmake)

check_compiler_version()

#set default build type
#available build types are: Debug, Release, RelWithDebInfo and MinSizeRel
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

#make build variable case insensitive
string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_CASE_INSENSITIVE)

#predefined options
option(USE_SECURITY_FLAGS "Use security flags" TRUE)
if (${CMAKE_BUILD_TYPE_CASE_INSENSITIVE} STREQUAL "debug")
    set(USE_SECURITY_FLAGS FALSE)
endif()

if ("${DRM_INCLUDE_DIR}" STREQUAL "")
    find_path(DRM_INCLUDE_DIR "i915_drm.h" PATH_SUFFIXES "drm" "libdrm")
else()
    message(STATUS "DRM_INCLUDE_DIR is set by user: ${DRM_INCLUDE_DIR}")
endif()

if (DRM_INCLUDE_DIR)
    if (EXISTS ${DRM_INCLUDE_DIR}/i915_drm.h)
        set(ENABLE_DRM TRUE)
    else()
        set(ENABLE_DRM FALSE)
    endif()
else()
    set(ENABLE_DRM FALSE)
endif()

option(BUILD_EXAMPLES "Build examples" TRUE)
option(BUILD_FT "Build functional tests" TRUE)
option(BUILD_CONFIG "Build cmake configs" TRUE)
option(ENABLE_MPI "Enable MPI support" TRUE)
option(ENABLE_MPI_TESTS "Enable MPI tests support" TRUE)
option(ENABLE_SYCL_INTEROP_EVENT "Enable SYCL interop event support" TRUE)
option(ENABLE_OFI_HMEM "Enable OFI HMEM support" TRUE)
option(ENABLE_OFI_OOT_PROV "Enable OFI out-of-tree providers support" FALSE)
option(ENABLE_ITT "Enable ITT profiling support" TRUE)
option(ENABLE_PROFILING "Enable profiling support" TRUE)
option(ENABLE_PMIX "Enable PMIX support" TRUE)
option(ENABLE_UMF "Enable UMF support" TRUE)
option(ENABLE_STUB_BACKEND "Enable stub backend" TRUE)
option(ENABLE_LINKER_RUNPATH "Enable linker runpath flags" FALSE)
option(ENABLE_OMP "Enable openMP extension for intra-node collectives" TRUE)

option(USE_CODECOV_FLAGS "Calculate code coverage" FALSE)
option(WITH_ASAN "Use address sanitizer, can only be used in Debug build" FALSE)

#installation path variables
include(GNUInstallDirs)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_install" CACHE PATH "Default install path" FORCE)
endif()

# show build info
message(STATUS "Installation directory: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE_CASE_INSENSITIVE}")
message(STATUS "C compiler : ${CMAKE_C_COMPILER}")
message(STATUS "CXX compiler : ${CMAKE_CXX_COMPILER}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Build functional tests: ${BUILD_FT}")
message(STATUS "Build cmake configs: ${BUILD_CONFIG}")
message(STATUS "Enable MPI support: ${ENABLE_MPI}")
message(STATUS "Enable MPI tests support: ${ENABLE_MPI_TESTS}")
message(STATUS "Enable SYCL interop event support: ${ENABLE_SYCL_INTEROP_EVENT}")
message(STATUS "Enable OFI HMEM support: ${ENABLE_OFI_HMEM}")
message(STATUS "Enable OFI out-of-tree providers support: ${ENABLE_OFI_OOT_PROV}")
message(STATUS "Enable ITT profiling support: ${ENABLE_ITT}")
message(STATUS "Enable CCL profiling support: ${ENABLE_PROFILING}")
message(STATUS "Enable PMIX support: ${ENABLE_PMIX}")
message(STATUS "Enable UMF support: ${ENABLE_UMF}")
message(STATUS "Enable DRM support: ${ENABLE_DRM}")
message(STATUS "Enable stub backend: ${ENABLE_STUB_BACKEND}")
message(STATUS "Enable linker rpath flags: ${ENABLE_LINKER_RUNPATH}")
message(STATUS "Enable openMP extension for intra-node collectives: ${ENABLE_OMP}")

add_definitions(-DCCL_C_COMPILER="${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
add_definitions(-DCCL_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

SET(CCL_ENABLE_ZE OFF CACHE BOOL "Enable Level Zero support")

set(CCL_COMMON_INSTALL_PREFIX "intel64")
set(CMAKE_INSTALL_LIBDIR "lib")
set(CCL_INSTALL_LIB "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(CCL_INSTALL_INCLUDE "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(CCL_INSTALL_DOC "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR}")
set(CCL_INSTALL_BIN "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
set(CCL_INSTALL_ENV "${CMAKE_INSTALL_PREFIX}/env")
set(CCL_INSTALL_ETC "${CMAKE_INSTALL_PREFIX}/etc")
set(CCL_INSTALL_SHARE "${CMAKE_INSTALL_PREFIX}/share")
set(CCL_INSTALL_LICENSE "${CCL_INSTALL_SHARE}/doc/ccl/licensing")
set(CCL_INSTALL_MODULE "${CMAKE_INSTALL_PREFIX}/modulefiles")
set(CCL_INSTALL_EXAMPLES "${CMAKE_INSTALL_PREFIX}/examples")
set(CCL_INSTALL_TESTS "${CMAKE_INSTALL_PREFIX}/tests")
set(CCL_INSTALL_KERNELS "${CMAKE_INSTALL_PREFIX}/lib/ccl/kernels")

# setup dependency directories
set(DEPS_DIR "${PROJECT_SOURCE_DIR}/deps")

if ("${MPI_DIR}" STREQUAL "")
    set(MPI_INCLUDE_DIR "${DEPS_DIR}/mpi/include/")
    set(MPI_LIB_DIR "${DEPS_DIR}/mpi/lib/")
else()
    set(MPI_INCLUDE_DIR "${MPI_DIR}/include/")
    set(MPI_LIB_DIR "${MPI_DIR}/lib/")
endif()
message(STATUS "MPI_INCLUDE_DIR: ${MPI_INCLUDE_DIR}")
message(STATUS "MPI_LIB_DIR: ${MPI_LIB_DIR}")

if ("${LIBFABRIC_DIR}" STREQUAL "")
    set(LIBFABRIC_INCLUDE_DIR "${DEPS_DIR}/ofi/include")
    set(LIBFABRIC_LIB_DIR "${DEPS_DIR}/ofi/lib/")
else()
    set(LIBFABRIC_INCLUDE_DIR "${LIBFABRIC_DIR}/include/")
    set(LIBFABRIC_LIB_DIR "${LIBFABRIC_DIR}/lib")
endif()
message(STATUS "LIBFABRIC_LIB_DIR: ${LIBFABRIC_LIB_DIR}")
message(STATUS "LIBFABRIC_INCLUDE_DIR: ${LIBFABRIC_INCLUDE_DIR}")

set(HWLOC_INCLUDE_DIR "${DEPS_DIR}/hwloc/include/")
set(HWLOC_LIB_DIR "${DEPS_DIR}/hwloc/lib/")
message(STATUS "HWLOC_INCLUDE_DIR: ${HWLOC_INCLUDE_DIR}")
message(STATUS "HWLOC_LIB_DIR: ${HWLOC_LIB_DIR}")

set(ITT_INCLUDE_DIR "${DEPS_DIR}/itt/include")
set(ITT_LIB_DIR "${DEPS_DIR}/itt/lib64")
message(STATUS "ITT_INCLUDE_DIR: ${ITT_INCLUDE_DIR}")
message(STATUS "ITT_LIB_DIR: ${ITT_LIB_DIR}")

set(LEVEL_ZERO_INCLUDE_DIR "${DEPS_DIR}/level_zero/include/")
message(STATUS "LEVEL_ZERO_INCLUDE_DIR: ${LEVEL_ZERO_INCLUDE_DIR}")

message(STATUS "DRM_INCLUDE_DIR: ${DRM_INCLUDE_DIR}")

set(PMIX_INCLUDE_DIR "${DEPS_DIR}/pmix/include/")
message(STATUS "PMIX_INCLUDE_DIR: ${PMIX_INCLUDE_DIR}")

set(UMF_INCLUDE_DIR "${DEPS_DIR}/umf/include/")
message(STATUS "UMF_INCLUDE_DIR: ${UMF_INCLUDE_DIR}")

set(CMAKE_SKIP_INSTALL_RPATH TRUE)
set(CMAKE_SKIP_RPATH TRUE)

if (${CMAKE_VERSION} VERSION_LESS 3.1)
#cmake version below 3.1 does not support CMAKE_C[XX}_STANDARD flags
#set manually
    set(CXX_COMPILER_FLAGS "-std=gnu++11")
    set(C_COMPILER_FLAGS "-std=gnu99")
endif()

if (${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(EXTRA_WARN_FLAGS "-Wshadow")
endif()

set(COMPILER_WARN_FLAGS "-Wall -Wextra -Wno-unused-parameter -Werror ${EXTRA_WARN_FLAGS}")

# common release/debug compilation settings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_COMPILER_FLAGS} ${COMPILER_WARN_FLAGS} -D_GNU_SOURCE -fvisibility=internal")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${C_COMPILER_FLAGS} -O0 -g -DENABLE_DEBUG")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${C_COMPILER_FLAGS} -O3")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${C_COMPILER_FLAGS} -O2 -g")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COMPILER_FLAGS} ${COMPILER_WARN_FLAGS} -D_GNU_SOURCE -fvisibility=internal")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_COMPILER_FLAGS} -O0 -g -DENABLE_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_COMPILER_FLAGS} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CXX_COMPILER_FLAGS} -O2 -g")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-implicit-fallthrough")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-fallthrough")
endif()

set(COMMON_CMAKE_DIR ${PROJECT_SOURCE_DIR}/cmake)

define_compute_backend()

if (COMPUTE_BACKEND)
    precheck_compute_backend()
    message(STATUS "COMPUTE_BACKEND: ${COMPUTE_BACKEND}")
    set_compute_backend(${COMMON_CMAKE_DIR})
    # Disable openMP extension for dpcpp
    if (${COMPUTE_BACKEND} STREQUAL "dpcpp")
        set(ENABLE_OMP FALSE)
        message(STATUS "Disable openMP extension for backend dpcpp, set ENABLE_OMP = ${ENABLE_OMP}")
    endif()
    if (${COMPUTE_BACKEND} STREQUAL "dpcpp" AND ENABLE_OFI_HMEM)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCL_ENABLE_OFI_HMEM=1")
        message(STATUS "Enable OFI HMEM support for compute backend ${COMPUTE_BACKEND}")
    endif()
    if (${COMPUTE_BACKEND} STREQUAL "dpcpp" AND ${CMAKE_CXX_COMPILER} MATCHES ".*icpx")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    endif()
endif()

if (ENABLE_OFI_OOT_PROV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCL_ENABLE_OFI_OOT_PROV=1")
    message(STATUS "Enable OFI out-of-tree providers support")
endif()

if (ENABLE_ITT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCL_ENABLE_ITT=1")
    message(STATUS "Enable ITT profiling support")
endif()

if (ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCL_ENABLE_PROFILING=1")
    message(STATUS "Enable CCL profiling support")
endif()


if (ENABLE_PMIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCL_ENABLE_PMIX=1")
    message(STATUS "Enable PMIX support")
endif()

if (ENABLE_UMF)
    set(CCL_ENABLE_UMF 1)  # Set CCL_ENABLE_UMF as a CMake variable
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCL_ENABLE_UMF=1") # Define it for the compiler
    message(STATUS "Enable UMF support")
endif()

if (ENABLE_DRM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCL_ENABLE_DRM=1")
    message(STATUS "Enable DRM support")
endif()

if (ENABLE_STUB_BACKEND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCL_ENABLE_STUB_BACKEND=1")
    message(STATUS "Enable stub backend")
endif()

if (ENABLE_OMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCL_ENABLE_OMP=1")
        message(STATUS "Enable openMP extension")
    endif()
endif(ENABLE_OMP)

if (${CMAKE_C_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
        #c++17 introduces algined new operator, use it
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
    endif()
endif()

# This is a temporal workaround until we fully switch to a new version of the compiler
# that supports the functionality
if (${ENABLE_SYCL_INTEROP_EVENT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCL_ENABLE_SYCL_INTEROP_EVENT=1")
endif()

# Clang doesn't automatically detects ninja processes as supporting colored output
# due to the way they are spawned. In order to fix the issue we need to use the option
# to force colored output
if (${CMAKE_GENERATOR} STREQUAL "Ninja")
    if (${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
    endif()
endif()

if (WITH_ASAN AND ${CMAKE_BUILD_TYPE_CASE_INSENSITIVE} STREQUAL "debug")
    message(STATUS "Compiling with address sanitizer")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lasan")
endif()

set_lp_env()
set_avx_env()
set_sycl_env()

set(CCL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/src)

enable_testing()

set(EXTERNAL_LIBS "")

set(EXAMPLES_INC_DIRS ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples/include ${MPI_INCLUDE_DIR})
set(EXAMPLES_LIB_DIRS ${MPI_LIB_DIR})

# allow `deprecated`
set(CMAKE_CLANG_FLAGS "${CMAKE_CLANG_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# generate & install vars.sh
configure_file(cmake/vars.sh.in ${CMAKE_CURRENT_BINARY_DIR}/vars.sh @ONLY)
configure_file(cmake/setvars.sh.in ${CMAKE_CURRENT_BINARY_DIR}/setvars.sh @ONLY)
configure_file(cmake/ccl ${CMAKE_CURRENT_BINARY_DIR}/ccl @ONLY)
configure_file(third-party-programs.txt ${CMAKE_CURRENT_BINARY_DIR}/third-party-programs.txt COPYONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/vars.sh DESTINATION ${CCL_INSTALL_ENV})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/setvars.sh DESTINATION ${CCL_INSTALL_ENV})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/ccl DESTINATION ${CCL_INSTALL_MODULE})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/third-party-programs.txt DESTINATION ${CCL_INSTALL_LICENSE})
install(PROGRAMS ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${CCL_INSTALL_LICENSE})

# copy kernels
if (COMPUTE_BACKEND AND EXISTS "${PROJECT_SOURCE_DIR}/src/kernels")
file(GLOB spv_kernels "${PROJECT_SOURCE_DIR}/src/kernels/kernels.spv")
    install(PROGRAMS ${spv_kernels}
            DESTINATION ${CCL_INSTALL_KERNELS}
            PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)
endif()

set(CCL_MAJOR_VERSION     "2021")
set(CCL_MINOR_VERSION     "16")
set(CCL_UPDATE_VERSION    "0")
set(CCL_PRODUCT_STATUS    "Gold")
string(TIMESTAMP CCL_PRODUCT_BUILD_DATE "%Y-%m-%dT %H:%M:%SZ")
get_vcs_properties("git")
set(CCL_PRODUCT_FULL "${CCL_PRODUCT_STATUS}-${CCL_MAJOR_VERSION}.${CCL_MINOR_VERSION}.${CCL_UPDATE_VERSION} ${CCL_PRODUCT_BUILD_DATE} ${VCS_INFO}")
configure_file(${PROJECT_SOURCE_DIR}/include/oneapi/ccl/config.h.in "${CMAKE_CURRENT_BINARY_DIR}/include/oneapi/ccl/config.h")
file(COPY "${CMAKE_CURRENT_BINARY_DIR}/include/oneapi/ccl/config.h" DESTINATION ${PROJECT_SOURCE_DIR}/include/oneapi/ccl)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

set(PROJECT_VERSION "${CCL_MAJOR_VERSION}.${CCL_MINOR_VERSION}.${CCL_UPDATE_VERSION}")

if (BUILD_CONFIG)
    configure_file("cmake/templates/oneCCLConfig.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/oneCCLConfig.cmake"
                   COPYONLY)
    configure_file("cmake/templates/oneCCLConfigVersion.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/oneCCLConfigVersion.cmake"
                   @ONLY)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/oneCCLConfig.cmake DESTINATION ${CCL_INSTALL_LIB}/cmake/oneCCL/)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/oneCCLConfigVersion.cmake DESTINATION ${CCL_INSTALL_LIB}/cmake/oneCCL/)

endif()

# include other CMakeLists

add_subdirectory(src)

if (ENABLE_MPI_TESTS)
    if (BUILD_EXAMPLES)
        add_subdirectory(examples/benchmark)
        add_subdirectory(examples/common)
        add_subdirectory(examples/cpu)
        add_subdirectory(examples/pt2pt)
	if (BUILD_CONFIG)
            add_subdirectory(examples/external_launcher)
        endif()
        if (CCL_ENABLE_SYCL)
            add_subdirectory(examples/sycl)
        endif()
    endif()
    if (BUILD_FT)
        add_subdirectory(tests/functional)
    endif()
endif()
