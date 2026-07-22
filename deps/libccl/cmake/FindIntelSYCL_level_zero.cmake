#===============================================================================
# Copyright 2019 Intel Corporation
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
#===============================================================================
# Prioritize L0_ROOT
file(GLOB sycl_headers "lib/clang/*/include")

list(APPEND dpcpp_root_hints
            ${DPCPP_ROOT}
            $ENV{DPCPP_ROOT})
set(original_cmake_prefix_path ${CMAKE_PREFIX_PATH})
if(dpcpp_root_hints)
    list(INSERT CMAKE_PREFIX_PATH 0 ${dpcpp_root_hints})
else()
    message("DPCPP_ROOT prefix path hint is not defiend")
endif()

if (NOT COMPUTE_BACKEND_NAME)
    message("Not OpenCL or L0")
endif()

include(CheckCXXCompilerFlag)
include(FindPackageHandleStandardArgs)

unset(INTEL_SYCL_SUPPORTED CACHE)
check_cxx_compiler_flag("-fsycl" INTEL_SYCL_SUPPORTED)

get_filename_component(INTEL_SYCL_BINARY_DIR ${CMAKE_CXX_COMPILER} PATH)

# Try to find Intel SYCL version.hpp header
find_path(INTEL_SYCL_INCLUDE_DIRS
    NAMES CL/sycl/version.hpp sycl/version.hpp
    PATHS
      ${sycl_root_hints}
      "${INTEL_SYCL_BINARY_DIR}/.."
      "${INTEL_SYCL_BINARY_DIR}/../opt/compiler"
    PATH_SUFFIXES
        include
        include/sycl
		"${sycl_headers}"
    NO_DEFAULT_PATH)

find_library(INTEL_SYCL_LIBRARIES
    NAMES "sycl"
    PATHS
        ${sycl_root_hints}
        "${INTEL_SYCL_BINARY_DIR}/.."
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH)

find_package_handle_standard_args(IntelSYCL_level_zero
    FOUND_VAR IntelSYCL_level_zero_FOUND
    REQUIRED_VARS
        INTEL_SYCL_LIBRARIES
        INTEL_SYCL_INCLUDE_DIRS
        INTEL_SYCL_SUPPORTED)

if(IntelSYCL_level_zero_FOUND AND NOT TARGET Intel::SYCL_level_zero)
    add_library(Intel::SYCL_level_zero UNKNOWN IMPORTED)
    message(STATUS "IntelSYCL_level_zero_FOUND: ${LEVEL_ZERO_INCLUDE_DIR}")
    list(APPEND SYCL_LEVEL_ZERO_INCLUDE_DIRS "${LEVEL_ZERO_INCLUDE_DIR}")
    list(APPEND SYCL_LEVEL_ZERO_INCLUDE_DIRS "${INTEL_SYCL_INCLUDE_DIRS}")

    message(STATUS "SYCL_LEVEL_ZERO_INCLUDE_DIRS: ${SYCL_LEVEL_ZERO_INCLUDE_DIRS}")
    set(imp_libs
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:-fsycl>
        ${COMPUTE_BACKEND_NAME})
    set_target_properties(Intel::SYCL_level_zero PROPERTIES
        INTERFACE_LINK_LIBRARIES "${imp_libs}"
        INTERFACE_INCLUDE_DIRECTORIES "${SYCL_LEVEL_ZERO_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${INTEL_SYCL_LIBRARIES}")
    set(INTEL_SYCL_FLAGS "-fsycl")
    mark_as_advanced(
        INTEL_SYCL_FLAGS
        INTEL_SYCL_LIBRARIES
        INTEL_SYCL_INCLUDE_DIRS)
endif()
