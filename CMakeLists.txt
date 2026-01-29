# Copyright 2016-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.19)

# Define the project
project(oneCCL2 VERSION 1.0 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops -fomit-frame-pointer")
    set(EXTRA_CXX_FLAGS "-ffast-math" "-flto")
endif()

option(ONECCL_HELGRIND_TESTING "Run tests through valgrind --tool=helgrind" OFF)
option(ONECCL_VALGRIND_TESTING "Run tests through valgrind --tool=memcheck" OFF)
option(ONECCL_ENABLE_COVERAGE "Compile with coverage enabled" OFF)
option(ONECCL_USE_SYSTEM_LIBCCL "Use system-installed libccl instead of subdirectory" OFF)
option(ONECCL_BUILD_TESTS "Build tests for oneCCL" ON)

set(ONECCL_LIBSTDCXX_PATH "" CACHE STRING "Specify the path to the custom libstdc++ file")

include(cmake/compat.cmake)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_install" CACHE PATH "Default install path" FORCE)
endif()

find_package(IntelSYCL QUIET)

# Check if the compiler requires -static-intel for CPU only targets
include(CheckLinkerFlag)
check_linker_flag(CXX "-static-intel" COMPILER_SUPPORTS_STATIC_INTEL_LIB)

# Check if compiler support SYCL
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" COMPILER_SUPPORTS_SYCL)

# Check if we can disable stdlib
check_linker_flag(CXX "-nostdlib++" COMPILER_SUPPORTS_NOSTDLIBXX)

# Build libccl.so.2
add_subdirectory(deps)
add_subdirectory(src)

if(NOT "${ONECCL_LIBSTDCXX_PATH}" STREQUAL "" AND COMPILER_SUPPORTS_NOSTDLIBXX)
    message(STATUS "Using specified libstdc++: ${ONECCL_LIBSTDCXX_PATH}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -nostdlib++ -l:${ONECCL_LIBSTDCXX_PATH}")
else()
    if(NOT COMPILER_SUPPORTS_NOSTDLIBXX)
        message(WARNING "Cannot use custom libstdc++ because -nostdlib++ is not supported by the compiler.")
    endif()
endif()

# Setup legacy libccl
if(NOT WIN32)
    set(DEST_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
    set(DEST_ENV_DIR "${CMAKE_INSTALL_PREFIX}/env")
    set(DEST_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
    set(DEST_OPT_DIR "${CMAKE_INSTALL_PREFIX}/opt")
    set(DEST_SHARE_DIR "${CMAKE_INSTALL_PREFIX}/share")

    if(${ONECCL_USE_SYSTEM_LIBCCL})
        # Attempt to find the system-installed libccl
        find_package(oneCCL REQUIRED)

        # Create an interface target based on the found package, so it's compatible with ccl target
        # originating from `add_subdirectory(libccl)`
        if(TARGET oneCCL)
            add_library(ccl INTERFACE IMPORTED)
            set_target_properties(ccl PROPERTIES
                INTERFACE_LINK_LIBRARIES oneCCL
            )

            # Install files from the system libccl root to the current install tree
            install(DIRECTORY "${_oneccl_root}/lib/" DESTINATION "${DEST_LIB_DIR}" USE_SOURCE_PERMISSIONS)
            install(DIRECTORY "${_oneccl_root}/env/" DESTINATION "${DEST_ENV_DIR}" USE_SOURCE_PERMISSIONS)
            install(DIRECTORY "${_oneccl_root}/include/" DESTINATION "${DEST_INCLUDE_DIR}" USE_SOURCE_PERMISSIONS)
            install(DIRECTORY "${_oneccl_root}/opt/" DESTINATION "${DEST_OPT_DIR}" USE_SOURCE_PERMISSIONS OPTIONAL)
            install(DIRECTORY "${_oneccl_root}/share/" DESTINATION "${DEST_SHARE_DIR}" USE_SOURCE_PERMISSIONS OPTIONAL)

            if(NOT EXISTS "${_oneccl_root}/lib/ccl/cpu/lib/libccl.so.1.0")
                message(WARNING "The file ${_oneccl_root}/lib/ccl/cpu/lib/libccl.so.1.0 does not exist. Please ensure your oneCCL install is complete and contains both CPU and GPU version of the library.")
            endif()
        else()
            message(FATAL_ERROR "System Installed libccl not found. Ensure it is installed and correctly configured.")
        endif()
    else()
        include(ExternalProject)
        include(ProcessorCount)

        ProcessorCount(NPROC)

        # Find directory of libccl.so.2 needed to provide joint API
        # inside libccl.so.1
        set(ONECCL2_LIB_DIR "$<PATH:GET_PARENT_PATH,$<TARGET_FILE:oneCCL2>>")

        # Use the local subdirectory version of `libccl`
        if(COMPILER_SUPPORTS_SYCL)
            set(LIBCCL_CMAKE_ARGS
                "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                "-DCOMPUTE_BACKEND=dpcpp"
                "-DBUILD_EXAMPLES=OFF"
                "-DBUILD_FT=OFF"
                "-DBUILD_REG_TESTS=OFF"
            )
            set(SHARED_LINKER_FLAGS "-L${ONECCL2_LIB_DIR} -l:libccl.so.2")

            if(NOT "${ONECCL_LIBSTDCXX_PATH}" STREQUAL "" AND COMPILER_SUPPORTS_NOSTDLIBXX)
                set(SHARED_LINKER_FLAGS "${SHARED_LINKER_FLAGS} -nostdlib++ -l:${ONECCL_LIBSTDCXX_PATH}")
            endif()
            list(APPEND LIBCCL_CMAKE_ARGS "-DCMAKE_SHARED_LINKER_FLAGS=${SHARED_LINKER_FLAGS}")

            if (DEFINED CMAKE_C_COMPILER)
                list(APPEND LIBCCL_CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
            endif()
            list(JOIN LIBCCL_CMAKE_ARGS ";" LIBCCL_CMAKE_ARGS_LIST)

            ExternalProject_Add(
                oneCCL
                PREFIX ${CMAKE_BINARY_DIR}/libccl
                SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/libccl
                BINARY_DIR ${CMAKE_BINARY_DIR}/libccl
                INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
                CMAKE_ARGS ${LIBCCL_CMAKE_ARGS_LIST}
                BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/libccl --parallel=${NPROC}
                INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} -j -C ${CMAKE_BINARY_DIR}/libccl install
                BUILD_BYPRODUCTS ${CMAKE_INSTALL_PREFIX}/libccl/src/libccl.so;${CMAKE_INSTALL_PREFIX}/libccl/src/libccl.so.1;${CMAKE_INSTALL_PREFIX}/libccl/src/libccl.so.1.0
                LOG_OUTPUT_ON_FAILURE TRUE
                LOG_BUILD ON
                LOG_INSTALL ON
            )

            add_library(ccl SHARED IMPORTED)
            add_dependencies(ccl oneCCL)
            set_target_properties(ccl PROPERTIES
                IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libccl/_install/lib/libccl.so"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/deps/libccl/include"
                
            )
            install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl/_install/lib/" DESTINATION "${DEST_LIB_DIR}" USE_SOURCE_PERMISSIONS)
            install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl/_install/env/" DESTINATION "${DEST_ENV_DIR}" USE_SOURCE_PERMISSIONS)
            install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl/_install/opt/" DESTINATION "${DEST_OPT_DIR}" USE_SOURCE_PERMISSIONS OPTIONAL)
            install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl/_install/share/" DESTINATION "${DEST_SHARE_DIR}" USE_SOURCE_PERMISSIONS OPTIONAL)
        endif()

        # Link intel libs statically
        if(COMPILER_SUPPORTS_STATIC_INTEL_LIB)
            set(STATIC_INTEL_FLAG -static-intel)
            message("The falg ${STATIC_INTEL_FLAG}")
        endif()

        set(LIBCCL_CPU_CMAKE_ARGS
            "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DBUILD_EXAMPLES=OFF"
            "-DBUILD_FT=OFF"
            "-DBUILD_REG_TESTS=OFF"
        )
        set(SHARED_LINKER_FLAGS "-L${ONECCL2_LIB_DIR} -l:libccl.so.2 ${STATIC_INTEL_FLAG}")

        if(NOT "${ONECCL_LIBSTDCXX_PATH}" STREQUAL "" AND COMPILER_SUPPORTS_NOSTDLIBXX)
            set(SHARED_LINKER_FLAGS "${SHARED_LINKER_FLAGS} -nostdlib++ -l:${ONECCL_LIBSTDCXX_PATH}")
        endif()
        list(APPEND LIBCCL_CPU_CMAKE_ARGS "-DCMAKE_SHARED_LINKER_FLAGS=${SHARED_LINKER_FLAGS}")

        if (DEFINED CMAKE_C_COMPILER)
            list(APPEND LIBCCL_CPU_CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
        endif()
        list(JOIN LIBCCL_CPU_CMAKE_ARGS ";" LIBCCL_CPU_CMAKE_ARGS_LIST)

        ExternalProject_Add(
            oneCCLCPU
            PREFIX ${CMAKE_BINARY_DIR}/libccl_cpu
            SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/libccl
            BINARY_DIR ${CMAKE_BINARY_DIR}/libccl_cpu
            INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
            CMAKE_ARGS ${LIBCCL_CPU_CMAKE_ARGS_LIST}
            BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/libccl_cpu --parallel=${NPROC}
            INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} -j -C ${CMAKE_BINARY_DIR}/libccl_cpu install
            BUILD_BYPRODUCTS ${CMAKE_INSTALL_PREFIX}/libccl_cpu/src/libccl.so;${CMAKE_INSTALL_PREFIX}/libccl_cpu/src/libccl.so.1;${CMAKE_INSTALL_PREFIX}/libccl_cpu/src/libccl.so.1.0
            LOG_OUTPUT_ON_FAILURE TRUE
            LOG_BUILD ON
            LOG_INSTALL ON
        )

        add_library(ccl_cpu SHARED IMPORTED)
        add_dependencies(ccl_cpu oneCCLCPU)
        set_target_properties(ccl_cpu PROPERTIES
            IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libccl_cpu/_install/lib/libccl.so"
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/deps/libccl/include"
        )
        install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl_cpu/_install/lib/" DESTINATION "${DEST_LIB_DIR}/ccl/cpu/lib/" USE_SOURCE_PERMISSIONS)
        install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl_cpu/_install/env/" DESTINATION "${DEST_ENV_DIR}" USE_SOURCE_PERMISSIONS)
        install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl_cpu/_install/opt/" DESTINATION "${DEST_OPT_DIR}" USE_SOURCE_PERMISSIONS OPTIONAL)
        install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl_cpu/_install/share/" DESTINATION "${DEST_SHARE_DIR}" USE_SOURCE_PERMISSIONS OPTIONAL)
    endif()
endif() # NOT WIN32

# Add code coverage flags
if(${ONECCL_ENABLE_COVERAGE})
    set(COVERAGE_CXX_FLAGS -fprofile-instr-generate -fcoverage-mapping)
endif()

# Enable tests in top-level cmake for easy usage
enable_testing()

# Build everything except API library
add_subdirectory(plugins)
add_subdirectory(examples)
add_subdirectory(tests)

# Install public headers
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

# Install legacy ccl public headers
install(DIRECTORY deps/libccl/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include OPTIONAL)

# Install cmake config files
install(FILES cmake/oneCCL2Config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/oneCCL2)
install(FILES cmake/oneCCL2ConfigVersion.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/oneCCL2)

if(COMPILER_SUPPORTS_SYCL)
    install_symlink(libccl.so.1 ${CMAKE_INSTALL_PREFIX}/lib/libccl.so)
else()
    install_symlink(./ccl/cpu/lib/libccl.so.1 ${CMAKE_INSTALL_PREFIX}/lib/libccl.so)
endif()
