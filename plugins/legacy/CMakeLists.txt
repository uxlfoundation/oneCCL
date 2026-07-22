# CMakeLists.txt for ccl_legacy plugin

# Add a shared library for the plugin
add_library(ccl_legacy SHARED ccl_legacy.cpp)


# Add SYCL
add_sycl_to_target(TARGET ccl_legacy SOURCES ccl_legacy.cpp)

# Link pluginA with oneCCL2 shared library and also with oneCCL and MPI libraries
target_link_libraries(ccl_legacy PRIVATE oneCCL2 ccl)
if(ONECCL_ENABLE_ITT)
  target_link_libraries(ccl_legacy PRIVATE ittnotify)
endif()

# Make sure the oneCCL2 headers and oneCCL, MPI headers can be included by the plugin
target_include_directories(ccl_legacy PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(ccl_legacy PRIVATE "${PROJECT_SOURCE_DIR}/include")

# Add include directories for the MPI headers
target_include_directories(ccl_legacy PRIVATE ${MPI_CXX_INCLUDE_PATH})

# Add coverage
target_compile_options(ccl_legacy PRIVATE "${COVERAGE_CXX_FLAGS}")
target_link_libraries(ccl_legacy PRIVATE "${COVERAGE_CXX_FLAGS}")

# Add performance flags
target_compile_options(ccl_legacy PRIVATE "${EXTRA_CXX_FLAGS}")
target_link_options(ccl_legacy PRIVATE "${EXTRA_CXX_FLAGS}")

install(TARGETS ccl_legacy LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
