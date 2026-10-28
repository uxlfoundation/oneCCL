# CMakeLists.txt for ccl_legacy_cpu plugin
#set(CMAKE_CXX_COMPILER g++)
# Add a shared library for the plugin
add_library(ccl_legacy_cpu SHARED ccl_legacy_cpu.cpp)

# Link pluginA with oneCCL2 shared library and also with oneCCL and MPI libraries
target_link_libraries(ccl_legacy_cpu PUBLIC oneCCL2)

# Add itt
if(ONECCL_ENABLE_ITT)
  target_link_libraries(ccl_legacy_cpu PRIVATE ittnotify)
endif()

# Make sure the oneCCL2 headers can be included by the plugin
target_include_directories(ccl_legacy_cpu PRIVATE "${PROJECT_SOURCE_DIR}/include")

# Private headers
target_include_directories(ccl_legacy_cpu PRIVATE "${PROJECT_SOURCE_DIR}/src")

# Add coverage
target_compile_options(ccl_legacy_cpu PRIVATE "${COVERAGE_CXX_FLAGS}")
target_link_libraries(ccl_legacy_cpu PRIVATE "${COVERAGE_CXX_FLAGS}")

# Add performance flags
target_compile_options(ccl_legacy_cpu PRIVATE "${EXTRA_CXX_FLAGS}")
target_link_options(ccl_legacy_cpu PRIVATE "${EXTRA_CXX_FLAGS}")

# Set the RPATH for the install tree: prefer the CPU-only libccl.so.1 from
# ccl/cpu/lib, fall back to the one next to the plugin (flat wheel layout)
set_target_properties(ccl_legacy_cpu PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH "\$ORIGIN/ccl/cpu/lib/;\$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Use DT_RPATH instead of DT_RUNPATH so the CPU-only libccl.so.1 wins over a
# GPU libccl.so.1 found through LD_LIBRARY_PATH
target_link_options(ccl_legacy_cpu PRIVATE "-Wl,--disable-new-dtags")

# Set the RPATH for the build tree if using system libccl.so
if(NOT "${_oneccl_root}" STREQUAL "")
    set_target_properties(ccl_legacy_cpu PROPERTIES
        BUILD_RPATH "${_oneccl_root}/lib/ccl/cpu/lib/"
    )

    # Add include directories for oneCCL headers
    target_include_directories(ccl_legacy_cpu PRIVATE "${_oneccl_root}/include")

    # Link with libccl, enable linking using $ORIGIN
    target_link_options(ccl_legacy_cpu PRIVATE "-lccl" "-Wl,-z,origin")
else()
    target_link_libraries(ccl_legacy_cpu PUBLIC ccl_cpu)
endif()

if(COMPILER_SUPPORTS_STATIC_INTEL_LIB)
    target_link_options(ccl_legacy_cpu PRIVATE "-static-intel")
endif()

install(TARGETS ccl_legacy_cpu LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
