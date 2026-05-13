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

# Define the executable target for AllReduce benchmark
add_executable(allreduce_perf
    all_reduce.cpp
    common.cpp
    timer.cpp
)

# Define the executable target for Broadcast benchmark
add_executable(broadcast_perf
    broadcast.cpp
    common.cpp
    timer.cpp
)

# Define the executable target for AllGather benchmark
add_executable(allgather_perf
    allgather.cpp
    common.cpp
    timer.cpp
)

# Define the executable target for ReduceScatter benchmark
add_executable(reduce_scatter_perf
    reduce_scatter.cpp
    common.cpp
    timer.cpp
)

# Define the executable target for p2p benchmark
add_executable(p2p_perf
    p2p.cpp
    common.cpp
    timer.cpp
)

# Find necessary packages
find_package(MPI REQUIRED)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Set the include directory to access oneCCL headers from the parent 'src'
target_include_directories(allreduce_perf PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)

target_include_directories(broadcast_perf PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)

target_include_directories(allgather_perf PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)

target_include_directories(reduce_scatter_perf PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)

target_include_directories(p2p_perf PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)

# Link the executables against oneCCL, MPI, and pthreads
target_link_libraries(allreduce_perf PRIVATE
    oneCCL2
    MPI::MPI_CXX
    Threads::Threads
)

target_link_libraries(broadcast_perf PRIVATE
    oneCCL2
    MPI::MPI_CXX
    Threads::Threads
)

target_link_libraries(allgather_perf PRIVATE
    oneCCL2
    MPI::MPI_CXX
    Threads::Threads
)

target_link_libraries(reduce_scatter_perf PRIVATE
    oneCCL2
    MPI::MPI_CXX
    Threads::Threads
)

target_link_libraries(p2p_perf PRIVATE
    oneCCL2
    MPI::MPI_CXX
    Threads::Threads
)

# Add SYCL-specific compilation properties to the targets.
# This assumes 'add_sycl_to_target' is available from your top-level CMake setup.
add_sycl_to_target(TARGET allreduce_perf SOURCES
    all_reduce.cpp
    common.cpp
    timer.cpp
)

add_sycl_to_target(TARGET broadcast_perf SOURCES
    broadcast.cpp
    common.cpp
    timer.cpp
)

add_sycl_to_target(TARGET allgather_perf SOURCES
    allgather.cpp
    common.cpp
    timer.cpp
)

add_sycl_to_target(TARGET reduce_scatter_perf SOURCES
    reduce_scatter.cpp
    common.cpp
    timer.cpp
)

add_sycl_to_target(TARGET p2p_perf SOURCES
    p2p.cpp
    common.cpp
    timer.cpp
)

# Optional: Add a custom test entry to run the benchmark via ctest
if(${ONECCL_BUILD_TESTS})
    add_test(
        NAME allreduce_perf_test
        COMMAND mpiexec -n 2 $<TARGET_FILE:allreduce_perf> -b 1K -e 1M -g 1
        CONFIGURATIONS benchmarks
    )
    set_tests_properties(allreduce_perf_test PROPERTIES
        TIMEOUT 600
    )
    
    add_test(
        NAME broadcast_perf_test
        COMMAND mpiexec -n 2 $<TARGET_FILE:broadcast_perf> -b 1K -e 1M -g 1
        CONFIGURATIONS benchmarks
    )
    set_tests_properties(broadcast_perf_test PROPERTIES
        TIMEOUT 600
    )
    
    add_test(
        NAME allgather_perf_test
        COMMAND mpiexec -n 2 $<TARGET_FILE:allgather_perf> -b 1K -e 1M -g 1
        CONFIGURATIONS benchmarks
    )
    set_tests_properties(allgather_perf_test PROPERTIES
        TIMEOUT 600
    )

    add_test(
        NAME reduce_scatter_perf_test
        COMMAND mpiexec -n 2 $<TARGET_FILE:reduce_scatter_perf> -b 1K -e 1M -g 1
        CONFIGURATIONS benchmarks
    )
    set_tests_properties(reduce_scatter_perf_test PROPERTIES
        TIMEOUT 600
    )
    add_test(
        NAME p2p_perf_test
        COMMAND mpiexec -n 2 $<TARGET_FILE:p2p_perf> -b 1K -e 1M -g 1
        CONFIGURATIONS benchmarks
    )
    set_tests_properties(p2p_perf_test PROPERTIES
        TIMEOUT 600
    )
    
endif()
