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
add_executable(launcher launcher.cpp)

# Find threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)


# Link simple with oneCCL2 shared library
target_link_libraries(launcher PRIVATE oneCCL2)

# Add threads, W/A for launcher issues with linking
target_link_libraries(launcher PRIVATE Threads::Threads)

# Find the SYCL package
find_package(IntelSYCL REQUIRED)

# Add SYCL
add_sycl_to_target(TARGET launcher SOURCES launcher.cpp)

if(${ONECCL_BUILD_TESTS})
    add_test(NAME example_launcher COMMAND $<TARGET_FILE:launcher> 4 CONFIGURATIONS examples_sycl)
    set_property(TEST example_launcher PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>")
    set_tests_properties(example_launcher PROPERTIES TIMEOUT 300)
endif()
