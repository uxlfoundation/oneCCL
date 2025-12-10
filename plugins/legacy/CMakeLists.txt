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

# CMakeLists.txt for ccl_legacy plugin

# Add a shared library for the plugin
add_library(ccl_legacy SHARED ccl_legacy.cpp)

# Find the MPI package
find_package(MPI REQUIRED)

# Find the SYCL package
find_package(IntelSYCL REQUIRED)

# Add SYCL
add_sycl_to_target(TARGET ccl_legacy SOURCES ccl_legacy.cpp)

# Link pluginA with oneCCL2 shared library and also with oneCCL and MPI libraries
target_link_libraries(ccl_legacy PRIVATE oneCCL2 ccl MPI::MPI_CXX)
target_link_libraries(ccl_legacy PRIVATE ittnotify)

# Make sure the oneCCL2 headers and oneCCL, MPI headers can be included by the plugin
target_include_directories(ccl_legacy PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(ccl_legacy PRIVATE "${PROJECT_SOURCE_DIR}/include")

# Add include directories for the MPI headers
target_include_directories(ccl_legacy PRIVATE ${MPI_CXX_INCLUDE_PATH})

# Add coverage
target_compile_options(ccl_legacy PRIVATE "${COVERAGE_CXX_FLAGS}")
target_link_libraries(ccl_legacy PRIVATE "${COVERAGE_CXX_FLAGS}")

# Add performance flags
target_compile_options(ccl_legacy PRIVATE "${EXTRA_CXX_FLAGS}")
target_link_options(ccl_legacy PRIVATE "${EXTRA_CXX_FLAGS}")

install(TARGETS ccl_legacy LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
