cmake_minimum_required(VERSION 3.20)
project(ExampleApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add an executable for the example using sycl example
add_executable(example ../sycl/sycl.cpp)

# Find threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Find the MPI package
find_package(MPI REQUIRED)

# Link simple with MPI shared library
target_link_libraries(example PRIVATE MPI::MPI_CXX)

# Add threads, W/A for example issues with linking
target_link_libraries(example PRIVATE Threads::Threads)


# Add SYCL
add_sycl_to_target(TARGET example SOURCES ../sycl/sycl.cpp)

# Find oneCCL2 package
find_package(oneCCL2 REQUIRED)

# Add oneCCL2 libraries and headers
target_link_libraries(example PRIVATE oneCCL2)
