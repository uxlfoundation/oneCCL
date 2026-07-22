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
set_property(TEST example_simple PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_null>")
set_tests_properties(example_simple PROPERTIES TIMEOUT 30)
