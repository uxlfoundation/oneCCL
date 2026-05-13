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

# Add an executable for the example
add_executable(sycl sycl.cpp)
add_executable(sycl_memory_api sycl_memory_api.cpp)

# Find threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Find the MPI package
find_package(MPI REQUIRED)

# Link simple with MPI shared library
target_link_libraries(sycl PRIVATE MPI::MPI_CXX)
target_link_libraries(sycl_memory_api PRIVATE MPI::MPI_CXX)

# Link simple with oneCCL2 shared library
target_link_libraries(sycl PRIVATE oneCCL2)
target_link_libraries(sycl_memory_api PRIVATE oneCCL2)

# Add threads, W/A for sycl issues with linking
target_link_libraries(sycl PRIVATE Threads::Threads)
target_link_libraries(sycl_memory_api PRIVATE Threads::Threads)

# Find the SYCL package
find_package(IntelSYCL REQUIRED)

# Add SYCL
add_sycl_to_target(TARGET sycl SOURCES sycl.cpp)
add_sycl_to_target(TARGET sycl_memory_api SOURCES sycl_memory_api.cpp)

if(${ONECCL_BUILD_TESTS})
    add_test(NAME example_sycl COMMAND mpiexec -n 2 $<TARGET_FILE:sycl> CONFIGURATIONS examples_sycl)
    set_property(TEST example_sycl PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>")
    set_tests_properties(example_sycl PROPERTIES TIMEOUT 600)

    add_test(NAME example_sycl_memory_api COMMAND mpiexec -n 2 $<TARGET_FILE:sycl_memory_api> CONFIGURATIONS examples_sycl)
    set_property(TEST example_sycl_memory_api PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>")
    set_tests_properties(example_sycl_memory_api PROPERTIES TIMEOUT 600)
endif()
