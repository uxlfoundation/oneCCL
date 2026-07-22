# CMakeLists.txt for ccl_legacy plugin

# Add a shared library for the plugin
add_library(ccl_null SHARED null_comm_impl.cpp null_hook.cpp)

# Make sure the oneCCL2 headers are present
target_include_directories(ccl_null PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(ccl_null PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(ONECCL_ENABLE_ITT)
  target_include_directories(ccl_null PRIVATE "${ittapi_SOURCE_DIR}/include")
endif()

# If targets link directly to ccl_null in cmake (does not affect plugin discovery)
target_include_directories(ccl_null PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(ccl_null PRIVATE "${COVERAGE_CXX_FLAGS}")
if(ONECCL_ENABLE_ITT)
  target_link_libraries(ccl_null PRIVATE "${COVERAGE_CXX_FLAGS}" ittnotify)
endif()

install(TARGETS ccl_null LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)