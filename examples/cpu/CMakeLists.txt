# Add an executable for the example
#set(CMAKE_CXX_COMPILER g++)
add_executable(cpu cpu.cpp)

# Find threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Find the MPI package
find_package(MPI REQUIRED)

# Link simple with MPI shared library
target_link_libraries(cpu PRIVATE MPI::MPI_CXX)

# Link simple with oneCCL2 shared library
target_link_libraries(cpu PRIVATE oneCCL2)

# Add threads, W/A for sycl issues with linking
target_link_libraries(cpu PRIVATE Threads::Threads)

if(${ONECCL_BUILD_TESTS} AND TARGET ccl_legacy_cpu)
    add_test(NAME example_cpu COMMAND mpiexec -n 2 ${PROJECT_SOURCE_DIR}/utils/debug_hook.sh $<TARGET_FILE:cpu> CONFIGURATIONS examples)
    set_property(TEST example_cpu PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy_cpu>")
    set_tests_properties(example_cpu PROPERTIES TIMEOUT 300)
endif()
