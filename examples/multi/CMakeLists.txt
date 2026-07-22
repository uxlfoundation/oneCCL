# Add an executable for the example
add_executable(multi multi.cpp)

# Find threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Find the MPI package
find_package(MPI REQUIRED)

# Link simple with MPI shared library
target_link_libraries(multi PRIVATE MPI::MPI_CXX)

# Link simple with oneCCL2 shared library
target_link_libraries(multi PRIVATE oneCCL2)

# Add threads, W/A for sycl issues with linking
target_link_libraries(multi PRIVATE Threads::Threads)


# Add SYCL
add_sycl_to_target(TARGET multi SOURCES multi.cpp)

if(${ONECCL_BUILD_TESTS})
    add_test(NAME example_multi COMMAND mpiexec -n 2 $<TARGET_FILE:multi> CONFIGURATIONS examples_sycl)
    set_property(TEST example_multi PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>")
    set_tests_properties(example_multi PROPERTIES TIMEOUT 600)
endif()
