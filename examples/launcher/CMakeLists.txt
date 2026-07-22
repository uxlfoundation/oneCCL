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


# Add SYCL
add_sycl_to_target(TARGET launcher SOURCES launcher.cpp)

if(${ONECCL_BUILD_TESTS})
    add_test(NAME example_launcher COMMAND $<TARGET_FILE:launcher> 2 CONFIGURATIONS examples_sycl)
    set_property(TEST example_launcher PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>")
    set_tests_properties(example_launcher PROPERTIES TIMEOUT 300)
endif()
