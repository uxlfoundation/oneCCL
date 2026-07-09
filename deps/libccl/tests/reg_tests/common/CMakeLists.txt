cmake_minimum_required(VERSION 3.20)

# pkg-config test

# only enable package tests tests if we build in standalone mode
# as they require ccl package
if (${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    add_subdirectory(package_tests)
endif()

set(CCL_INSTALL_TESTS "${CMAKE_CURRENT_SOURCE_DIR}")

# abi-neutrality tests
# TODO extend for whole API tests - compile benchmark/whole samples directory ?
function (add_abi_test executable_name sources pre_abi)
    add_executable("${executable_name}" "${sources}")
    target_include_directories("${executable_name}" PRIVATE ${EXAMPLES_INC_DIRS})
    target_link_libraries("${executable_name}" PRIVATE ccl)
    target_link_libraries("${executable_name}" PUBLIC -L${I_MPI_ROOT}/lib/release/)
    target_link_libraries("${executable_name}" PUBLIC mpi)
    target_link_libraries("${executable_name}" PUBLIC ${COMPUTE_BACKEND_TARGET_NAME})
    install(TARGETS "${executable_name}" RUNTIME DESTINATION ${CCL_INSTALL_TESTS} OPTIONAL)
    string(REGEX MATCH ".*(sycl|SYCL).*" SYCL_USED "${COMPUTE_BACKEND_TARGET_NAME}")
    if (SYCL_USED)
        set_target_properties(${executable_name} PROPERTIES COMPILE_FLAGS "-fpreview-breaking-changes -D_GLIBCXX_USE_CXX11_ABI=${pre_abi}")
        set_target_properties(${executable_name} PROPERTIES LINK_FLAGS "-fpreview-breaking-changes -D_GLIBCXX_USE_CXX11_ABI=${pre_abi}")
    else ()
        set_target_properties(${executable_name} PROPERTIES COMPILE_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${pre_abi}")
        set_target_properties(${executable_name} PROPERTIES LINK_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${pre_abi}")
   endif ()
endfunction ()

add_abi_test(abi_test abi_tests.cpp 0)
add_abi_test(abi_test_pre abi_tests.cpp 1)
