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
add_library(ccl_null SHARED null_comm_impl.cpp null_hook.cpp)

# Make sure the oneCCL2 headers are present
target_include_directories(ccl_null PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(ccl_null PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(ccl_null PRIVATE "${ittapi_SOURCE_DIR}/include")

# If targets link directly to ccl_null in cmake (does not affect plugin discovery)
target_include_directories(ccl_null PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(ccl_null PRIVATE "${COVERAGE_CXX_FLAGS}")
target_link_libraries(ccl_null PRIVATE "${COVERAGE_CXX_FLAGS}" ittnotify)
