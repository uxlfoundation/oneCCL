# Add an executable for the example
add_executable(sycl sycl.cpp)
add_executable(sycl_memory_api sycl_memory_api.cpp)
add_executable(sycl_finalize sycl_finalize.cpp)
add_executable(sycl_graph sycl_graph.cpp)
add_executable(sycl_graph_finalize sycl_graph_finalize.cpp)

# Find threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Find the MPI package
find_package(MPI REQUIRED)

# Link simple with MPI shared library
target_link_libraries(sycl PRIVATE MPI::MPI_CXX)
target_link_libraries(sycl_memory_api PRIVATE MPI::MPI_CXX)
target_link_libraries(sycl_finalize PRIVATE MPI::MPI_CXX)
target_link_libraries(sycl_graph PRIVATE MPI::MPI_CXX)
target_link_libraries(sycl_graph_finalize PRIVATE MPI::MPI_CXX)

# Link simple with oneCCL2 shared library
target_link_libraries(sycl PRIVATE oneCCL2)
target_link_libraries(sycl_memory_api PRIVATE oneCCL2)
target_link_libraries(sycl_finalize PRIVATE oneCCL2)
target_link_libraries(sycl_graph PRIVATE oneCCL2)
target_link_libraries(sycl_graph_finalize PRIVATE oneCCL2)

# Add threads, W/A for sycl issues with linking
target_link_libraries(sycl PRIVATE Threads::Threads)
target_link_libraries(sycl_memory_api PRIVATE Threads::Threads)
target_link_libraries(sycl_finalize PRIVATE Threads::Threads)
target_link_libraries(sycl_graph PRIVATE Threads::Threads)
target_link_libraries(sycl_graph_finalize PRIVATE Threads::Threads)

# Find the SYCL package

# Add SYCL
add_sycl_to_target(TARGET sycl SOURCES sycl.cpp)
add_sycl_to_target(TARGET sycl_memory_api SOURCES sycl_memory_api.cpp)
add_sycl_to_target(TARGET sycl_finalize SOURCES sycl_finalize.cpp)
add_sycl_to_target(TARGET sycl_graph SOURCES sycl_graph.cpp)
add_sycl_to_target(TARGET sycl_graph_finalize SOURCES sycl_graph_finalize.cpp)

if(${ONECCL_BUILD_TESTS})
    add_test(NAME example_sycl COMMAND mpiexec -n 2 $<TARGET_FILE:sycl> CONFIGURATIONS examples_sycl)
    set_property(TEST example_sycl PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>")
    set_tests_properties(example_sycl PROPERTIES TIMEOUT 600)

    add_test(NAME example_sycl_memory_api COMMAND mpiexec -n 2 $<TARGET_FILE:sycl_memory_api> CONFIGURATIONS examples_sycl)
    set_property(TEST example_sycl_memory_api PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>")
    set_tests_properties(example_sycl_memory_api PROPERTIES TIMEOUT 600)

    add_test(NAME example_sycl_finalize COMMAND mpiexec -n 2 $<TARGET_FILE:sycl_finalize> CONFIGURATIONS examples_sycl)
    set_property(TEST example_sycl_finalize PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>")
    set_tests_properties(example_sycl_finalize PROPERTIES TIMEOUT 600)

    add_test(NAME example_sycl_graph COMMAND mpiexec -n 2 $<TARGET_FILE:sycl_graph> CONFIGURATIONS examples_sycl)
    set_property(TEST example_sycl_graph PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>")
    set_tests_properties(example_sycl_graph PROPERTIES TIMEOUT 600)

    add_test(NAME example_sycl_graph_finalize COMMAND mpiexec -n 2 $<TARGET_FILE:sycl_graph_finalize> CONFIGURATIONS examples_sycl)
    set_property(TEST example_sycl_graph_finalize PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>")
    set_tests_properties(example_sycl_graph_finalize PROPERTIES TIMEOUT 600)
endif()
