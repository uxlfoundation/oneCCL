# Copyright 2016-2026 Intel Corporation
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

# Create a shared library called oneCCL2
#set(CMAKE_CXX_COMPILER g++)
add_library(oneCCL2 SHARED api.cpp debug.cpp)

target_link_libraries(oneCCL2 PUBLIC ${CMAKE_DL_LIBS})

# Specify the public headers for the oneCCL2 library
target_include_directories(oneCCL2 PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
    "$<INSTALL_INTERFACE:include>" # Adjusts path for installed environments
)

target_include_directories(oneCCL2 PRIVATE
    "${ittapi_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
)

# Add flags for coverage
target_compile_options(oneCCL2 PRIVATE "${COVERAGE_CXX_FLAGS}")
target_link_libraries(oneCCL2 PRIVATE "${COVERAGE_CXX_FLAGS}")

# Add common compatiblity flags
target_compile_options(oneCCL2 PRIVATE "${COMMON_CXX_FLAGS}")

# Add additional performance flags
target_compile_options(oneCCL2 PRIVATE "${EXTRA_CXX_FLAGS}")
target_link_options(oneCCL2 PRIVATE "${EXTRA_CXX_FLAGS}")

if(COMPILER_SUPPORTS_STATIC_INTEL_LIB)
    target_link_options(oneCCL2 PRIVATE "-static-intel")
endif()

# Add flags for build identification
# --- Git commit ---
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
else()
  set(GIT_COMMIT "unknown")
endif()

# --- Build timestamp ---
string(TIMESTAMP BUILD_TIME_UTC "%Y-%m-%dT%H:%M:%SZ" UTC)

# --- Compiler info ---
set(COMPILER_ID   "${CMAKE_CXX_COMPILER_ID}")
set(COMPILER_VER  "${CMAKE_CXX_COMPILER_VERSION}")
set(COMPILER_PATH "${CMAKE_CXX_COMPILER}")

# --- Platform / config ---
set(BUILD_TYPE    "${CMAKE_BUILD_TYPE}")
set(SYSTEM_NAME   "${CMAKE_SYSTEM_NAME}")
set(SYSTEM_PROC   "${CMAKE_SYSTEM_PROCESSOR}")

target_compile_definitions(oneCCL2 PRIVATE
  BUILD_GIT_COMMIT="${GIT_COMMIT}"
  BUILD_TIME_UTC="${BUILD_TIME_UTC}"
  BUILD_COMPILER_ID="${COMPILER_ID}"
  BUILD_COMPILER_VERSION="${COMPILER_VER}"
  BUILD_COMPILER_PATH="${COMPILER_PATH}"
  BUILD_TYPE="${BUILD_TYPE}"
  BUILD_SYSTEM_NAME="${SYSTEM_NAME}"
  BUILD_SYSTEM_PROCESSOR="${SYSTEM_PROC}"
)

set_target_properties(oneCCL2 PROPERTIES OUTPUT_NAME ccl)
set_target_properties(oneCCL2 PROPERTIES VERSION 2.0)
set_target_properties(oneCCL2 PROPERTIES SOVERSION 2)

install(TARGETS oneCCL2 LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
