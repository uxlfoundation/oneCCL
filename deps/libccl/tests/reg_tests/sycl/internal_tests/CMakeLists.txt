# Internal tests that link against the shared ccl target and use internal oneCCL headers.
# Only added when building in the main tree where the ccl target is available.

cmake_minimum_required(VERSION 3.20)

set(CCL_INSTALL_TESTS "${CMAKE_CURRENT_SOURCE_DIR}")

# Same include paths as ccl-objects (see src/CMakeLists.txt)
set(INTERNAL_TEST_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/../../include
    ${PROJECT_SOURCE_DIR}/../../src
    ${PROJECT_SOURCE_DIR}/../../src/atl
    ${PROJECT_SOURCE_DIR}/../../src/occp
    ${PROJECT_SOURCE_DIR}/../../src/occp/inc
    ${PROJECT_SOURCE_DIR}/../../src/occp/infra
    ${PROJECT_SOURCE_DIR}/../../src/occp/protocol
    ${LEVEL_ZERO_INCLUDE_DIR}
    ${LIBFABRIC_INCLUDE_DIR}
    ${HWLOC_INCLUDE_DIR}
    ${ITT_INCLUDE_DIR})

if (ENABLE_DRM)
    list(APPEND INTERNAL_TEST_INCLUDE_DIRS ${DRM_INCLUDE_DIR})
endif()
if (ENABLE_PMIX)
    list(APPEND INTERNAL_TEST_INCLUDE_DIRS ${PMIX_INCLUDE_DIR})
endif()
if (ENABLE_UMF)
    list(APPEND INTERNAL_TEST_INCLUDE_DIRS ${UMF_INCLUDE_DIR})
endif()

file(GLOB sources "*.cpp")

# Bring in bundled googletest. tests/functional/CMakeLists.txt also adds the
# same subdir when BUILD_FT=ON, so guard to avoid double-registration.
if (NOT TARGET gtest)
    set(GTEST_DIR ${PROJECT_SOURCE_DIR}/../googletest-release-1.8.1/)
    add_subdirectory(${GTEST_DIR} gtest_internal_build)
endif()

# Build-tree path where libccl.so is emitted; used together with the
# install-tree lib/ dir to populate the test binary's DT_RPATH so it finds
# the project's own libccl.so without requiring the user to source
# env/setvars.sh first. See the linker-options comment below for the
# DT_RPATH vs DT_RUNPATH reasoning.
get_target_property(CCL_SHARED_LIB_DIR ccl LIBRARY_OUTPUT_DIRECTORY)
if (NOT CCL_SHARED_LIB_DIR)
    set(CCL_SHARED_LIB_DIR "${CMAKE_BINARY_DIR}/src")
endif()

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src})
    target_include_directories(${executable} PRIVATE ${INTERNAL_TEST_INCLUDE_DIRS})
    # Force the vendored deps/level_zero/include/ze_api.h to be parsed before
    # any toolchain-provided ze_api.h. Background:
    #
    # The linked ccl target inherits Intel::SYCL_level_zero's
    # INTERFACE_INCLUDE_DIRECTORIES, which contains BOTH the vendored
    # LEVEL_ZERO_INCLUDE_DIR and the SYCL toolchain include dir. Both are
    # propagated as -isystem (CMake marks IMPORTED-interface includes
    # SYSTEM by default). On some toolchains (e.g. icpx 2026.1 in a conda
    # env) the toolchain include dir ships an OLDER ze_api.h that lacks
    # symbols this PR's tests reference transitively via comm/comm.hpp ->
    # ze_api_wrapper.hpp.
    #
    # CMake de-duplicates target_include_directories by string equality, so
    # re-adding LEVEL_ZERO_INCLUDE_DIR PRIVATE BEFORE collapses against the
    # SYSTEM-inherited copy and we lose the priority -I form. Instead use
    # `-include <abspath>/ze_api.h` to pre-parse the vendored header. Its
    # `#ifndef _ZE_API_H` guard then makes any later `#include <ze_api.h>`
    # (the toolchain's stale copy) a no-op, regardless of -isystem ordering.
    target_compile_options(${executable} PRIVATE
        "SHELL:-include ${LEVEL_ZERO_INCLUDE_DIR}ze_api.h")
    # Link against the shared ccl target (built in the main tree) — this pulls
    # in its private deps (hwloc, itt, ...) via CMake. All oneCCL symbols we
    # reference (invoke_p2p_barrier_pair, ccl_comm methods) are either exported
    # or inline in the headers. gtest supplies the test framework (asserts,
    # RUN_ALL_TESTS, ::testing::Environment, etc.); the test provides its own
    # main() so gtest_main is not linked.
    target_link_libraries(${executable} PRIVATE ccl gtest)
    target_link_libraries(${executable} PUBLIC -L${I_MPI_ROOT}/lib/release/)
    target_link_libraries(${executable} PUBLIC mpi)
    target_link_libraries(${executable} PUBLIC ${COMPUTE_BACKEND_TARGET_NAME})
    # Bake the project's own libccl.so location into the binary so it wins
    # over any system-installed libccl at runtime. The top-level CMakeLists
    # sets CMAKE_SKIP_RPATH, which overrides CMake target properties, so we
    # pass -Wl,-rpath,... directly to the linker instead.
    # --disable-new-dtags emits the older DT_RPATH tag (which takes precedence
    # over LD_LIBRARY_PATH) rather than DT_RUNPATH (which is overridden by
    # LD_LIBRARY_PATH). Needed so the test picks our freshly-built libccl even
    # when a system-installed oneCCL is on LD_LIBRARY_PATH.
    target_link_options(${executable} PRIVATE
        "-Wl,--disable-new-dtags"
        "-Wl,-rpath,${CCL_SHARED_LIB_DIR}"
        "-Wl,-rpath,${CMAKE_INSTALL_PREFIX}/lib")
    install(TARGETS ${executable} RUNTIME DESTINATION ${CCL_INSTALL_TESTS} OPTIONAL)
endforeach(src ${sources})
