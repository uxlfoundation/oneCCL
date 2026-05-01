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
add_library(ccl_openmp SHARED openmp_ext.cpp)

if(NOT TARGET ccl)
	add_library(ccl INTERFACE)
	target_link_libraries(ccl INTERFACE -l:libccl.so.1)
endif()

target_include_directories(ccl_openmp PUBLIC ${SRC_INCLUDE_DIRS})
target_link_libraries(ccl_openmp PUBLIC OpenMP::OpenMP_CXX ccl)

set_target_properties(ccl_openmp PROPERTIES VERSION 0)
set_target_properties(ccl_openmp PROPERTIES SOVERSION 0.1)

install(TARGETS ccl_openmp LIBRARY DESTINATION ${CCL_INSTALL_LIB} OPTIONAL)
