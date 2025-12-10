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

# Add an executable for the example
add_executable(simple simple.cpp)

# Find threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Link simple with oneCCL2 shared library
target_link_libraries(simple PRIVATE oneCCL2)

# Add threads, W/A for sycl issues with linking
target_link_libraries(simple PRIVATE Threads::Threads)

add_test(NAME example_simple COMMAND $<TARGET_FILE:simple> CONFIGURATIONS examples)
set_property(TEST example_simple PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_mpi>")
set_tests_properties(example_simple PROPERTIES TIMEOUT 30)
