if(CCL_ENABLE_SYCL_RATE_LIMITER)
    find_package(IntelSYCL REQUIRED)
    find_package(MPI REQUIRED)

    set(RATE_LIMITER_TESTS
        sycl_rate_limiter_stats_test
        sycl_rate_limiter_e2e_test)

    foreach(test_name ${RATE_LIMITER_TESTS})
        add_executable(${test_name} ${test_name}.cpp)

        # Internal libccl headers (rate limiter implementation)
        target_include_directories(${test_name} PRIVATE
            ${PROJECT_SOURCE_DIR}/deps/libccl/src
            ${PROJECT_SOURCE_DIR}/deps/libccl/include
            ${PROJECT_SOURCE_DIR}/deps/libccl/examples/include
        )

        target_link_libraries(${test_name} PRIVATE ccl)
        target_link_libraries(${test_name} PRIVATE MPI::MPI_CXX)
        target_link_libraries(${test_name} PUBLIC rt m)

        add_sycl_to_target(TARGET ${test_name} SOURCES ${test_name}.cpp)
    endforeach()
endif()
