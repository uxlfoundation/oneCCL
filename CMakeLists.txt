cmake_minimum_required(VERSION 3.24)

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
option(ONECCL_LIBCCL_BUILD_FT         "Build libccl functional tests"   ON)
option(ONECCL_LIBCCL_BUILD_EXAMPLES   "Build libccl examples"            ON)
option(ONECCL_LIBCCL_BUILD_REG_TESTS  "Build libccl regression tests"    ON)
option(ONECCL_BUILD_CPU_PLUGIN        "Build CPU-only libccl + ccl_legacy_cpu plugin" ON)
set(ONECCL_LIBCCL_SYCL_DEVICE_CODE_SPLIT "" CACHE STRING
    "SYCL device-code-split mode for libccl: per_kernel | per_source | off (empty = use the historical default: per_kernel in Release, driver default otherwise)")
set_property(CACHE ONECCL_LIBCCL_SYCL_DEVICE_CODE_SPLIT
             PROPERTY STRINGS "" per_kernel per_source off)
option(ONECCL_LIBCCL_EXTERN_TEMPLATE_INST
       "Emit extern template declarations + per-dtype inst files in libccl (Release iteration win, Debug link-phase cost)" ON)
option(ONECCL_ENABLE_MPI "Enable MPI support" ON)
option(CCL_ENABLE_ARCA "Enable ARCA for embedded libccl builds" OFF)
option(CCL_ENABLE_ARCB "Enable ARCB for embedded libccl builds" OFF)

# Exactly one ARC mode must be selected.
if(CCL_ENABLE_ARCA AND CCL_ENABLE_ARCB)
    message(FATAL_ERROR "Exactly one option can be enabled: set either CCL_ENABLE_ARCA=ON or CCL_ENABLE_ARCB=ON")
endif()

option(ONECCL_ENABLE_ITT "Enable ITT profiling support" ON)

set(ONECCL_LIBSTDCXX_PATH "" CACHE STRING "Specify the path to the custom libstdc++ file")
option(ONECCL_BUILD_LIBS "Build oneCCL libraries" ON)
set(ONECCL_FORTIFY_SOURCE "1" CACHE STRING "Security hardening level: 0 disables, 1-4 progressively adds hardening flags")
set_property(CACHE ONECCL_FORTIFY_SOURCE PROPERTY STRINGS 0 1 2 3 4)

if(NOT ONECCL_FORTIFY_SOURCE MATCHES "^[0-4]$")
    message(FATAL_ERROR "ONECCL_FORTIFY_SOURCE must be one of 0, 1, 2, 3, 4")
endif()

include(CheckLinkerFlag)
include(CheckCXXCompilerFlag)

set(ONECCL_SECURITY_COMPILE_FLAGS "")
set(ONECCL_SECURITY_LINK_FLAGS "")
set(ONECCL_SECURITY_COMPILE_DEFINITIONS "")

string(TOUPPER "${CMAKE_BUILD_TYPE}" ONECCL_BUILD_TYPE_UPPER)
set(ONECCL_IS_RELEASE_BUILD OFF)
if(ONECCL_BUILD_TYPE_UPPER STREQUAL "RELEASE" OR
    ONECCL_BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO" OR
    ONECCL_BUILD_TYPE_UPPER STREQUAL "MINSIZEREL")
    set(ONECCL_IS_RELEASE_BUILD ON)
endif()

function(oneccl_append_supported_compile_flag output_list flag)
    string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _flag_var "${flag}")
    check_cxx_compiler_flag("${flag}" "ONECCL_HAS_COMPILE_FLAG_${_flag_var}")
    if(ONECCL_HAS_COMPILE_FLAG_${_flag_var})
        set(_updated_list ${${output_list}})
        list(APPEND _updated_list "${flag}")
        set(${output_list} "${_updated_list}" PARENT_SCOPE)
    endif()
endfunction()

function(oneccl_append_supported_link_flag output_list flag)
    string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _flag_var "${flag}")
    check_linker_flag(CXX "${flag}" "ONECCL_HAS_LINK_FLAG_${_flag_var}")
    if(ONECCL_HAS_LINK_FLAG_${_flag_var})
        set(_updated_list ${${output_list}})
        list(APPEND _updated_list "${flag}")
        set(${output_list} "${_updated_list}" PARENT_SCOPE)
    endif()
endfunction()

function(oneccl_append_compile_flag_for_min_fortify output_list min_level flag)
    if(ONECCL_FORTIFY_SOURCE GREATER_EQUAL ${min_level})
        oneccl_append_supported_compile_flag(${output_list} "${flag}")
    endif()
endfunction()

function(oneccl_append_link_flag_for_min_fortify output_list min_level flag)
    if(ONECCL_FORTIFY_SOURCE GREATER_EQUAL ${min_level})
        oneccl_append_supported_link_flag(${output_list} "${flag}")
    endif()
endfunction()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
    oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 1 "-Wall")
    oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 1 "-Wextra")
    oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 1 "-Wformat")
    oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 1 "-Wformat-security")
    oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 1 "-fstack-protector")
    oneccl_append_link_flag_for_min_fortify(ONECCL_SECURITY_LINK_FLAGS 1 "-Wl,-z,noexecstack")

    if(ONECCL_IS_RELEASE_BUILD)
        oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 1 "-Werror")
        oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 1 "-Wconversion")
        oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 1 "-Wimplicit-fallthrough")
    endif()

    oneccl_append_link_flag_for_min_fortify(ONECCL_SECURITY_LINK_FLAGS 2 "-Wl,-z,relro")

    oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 3 "-fstack-protector-strong")
    oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 3 "-Werror=format-security")
    oneccl_append_link_flag_for_min_fortify(ONECCL_SECURITY_LINK_FLAGS 3 "-Wl,-z,now")

    oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 4 "-fstack-clash-protection")
    oneccl_append_compile_flag_for_min_fortify(ONECCL_SECURITY_COMPILE_FLAGS 4 "-fcf-protection=full")

    if(ONECCL_FORTIFY_SOURCE GREATER 0 AND ONECCL_IS_RELEASE_BUILD)
        list(APPEND ONECCL_SECURITY_COMPILE_DEFINITIONS "_FORTIFY_SOURCE=${ONECCL_FORTIFY_SOURCE}")
    elseif(ONECCL_FORTIFY_SOURCE GREATER 0)
        message(WARNING "ONECCL_FORTIFY_SOURCE is set but CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'. Fortify checks typically require optimization (for example -O2).")
    endif()

    if(ONECCL_FORTIFY_SOURCE GREATER_EQUAL 2)
        list(APPEND ONECCL_SECURITY_COMPILE_DEFINITIONS "_GLIBCXX_ASSERTIONS")
    endif()
endif()

include(cmake/compat.cmake)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_install" CACHE PATH "Default install path" FORCE)
endif()

find_package(IntelSYCL QUIET)

# Check if compiler support SYCL
check_cxx_compiler_flag("-fsycl" COMPILER_SUPPORTS_SYCL)

# Fallback for open-source SYCL compilers that support -fsycl but don't provide IntelSYCL package
if(NOT COMMAND add_sycl_to_target AND COMPILER_SUPPORTS_SYCL)
    message(STATUS "add_sycl_to_target not found, using fallback for -fsycl support")
    include(${CMAKE_SOURCE_DIR}/cmake/IntelSYCLFallback.cmake)
endif()

# Check if the compiler requires -static-intel for CPU only targets
check_linker_flag(CXX "-static-intel" COMPILER_SUPPORTS_STATIC_INTEL_LIB)

# Check if we can disable stdlib
check_linker_flag(CXX "-nostdlib++" COMPILER_SUPPORTS_NOSTDLIBXX)

# Build libccl.so.2
add_subdirectory(deps)

if(ONECCL_SECURITY_COMPILE_FLAGS)
    add_compile_options(${ONECCL_SECURITY_COMPILE_FLAGS})
endif()

if(ONECCL_SECURITY_LINK_FLAGS)
    add_link_options(${ONECCL_SECURITY_LINK_FLAGS})
endif()

if(ONECCL_SECURITY_COMPILE_DEFINITIONS)
    add_compile_definitions(${ONECCL_SECURITY_COMPILE_DEFINITIONS})
endif()

if(ONECCL_BUILD_LIBS)
    # Build libccl.so.2
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
            if(NPROC GREATER 16)
                set(SYCL_LINK_JOBS 16)
            else()
                set(SYCL_LINK_JOBS ${NPROC})
            endif()

            # Find directory of libccl.so.2 needed to provide joint API
            # inside libccl.so.1
            set(ONECCL2_LIB_DIR "$<PATH:GET_PARENT_PATH,$<TARGET_FILE:oneCCL2>>")

            # Use the local subdirectory version of `libccl`
            if(COMPILER_SUPPORTS_SYCL)
                set(LIBCCL_CMAKE_ARGS
                    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                    "-DCOMPUTE_BACKEND=dpcpp"
                    "-DBUILD_EXAMPLES=${ONECCL_LIBCCL_BUILD_EXAMPLES}"
                    "-DBUILD_FT=${ONECCL_LIBCCL_BUILD_FT}"
                    "-DBUILD_REG_TESTS=${ONECCL_LIBCCL_BUILD_REG_TESTS}"
                    "-DSYCL_DEVICE_CODE_SPLIT=${ONECCL_LIBCCL_SYCL_DEVICE_CODE_SPLIT}"
                    "-DENABLE_EXTERN_TEMPLATE_INST=${ONECCL_LIBCCL_EXTERN_TEMPLATE_INST}"
                    "-DENABLE_ITT=${ONECCL_ENABLE_ITT}"
                    "-DONECCL_ENABLE_MPI=${ONECCL_ENABLE_MPI}"
                )
                set(SHARED_LINKER_FLAGS "-L${ONECCL2_LIB_DIR} -l:libccl.so.2 -fsycl-max-parallel-link-jobs=${SYCL_LINK_JOBS}")

                if(NOT "${ONECCL_LIBSTDCXX_PATH}" STREQUAL "" AND COMPILER_SUPPORTS_NOSTDLIBXX)
                    set(SHARED_LINKER_FLAGS "${SHARED_LINKER_FLAGS} -nostdlib++ -l:${ONECCL_LIBSTDCXX_PATH}")
                endif()
                if(CCL_ENABLE_ARCA)
                    list(APPEND LIBCCL_CMAKE_ARGS "-DCCL_ENABLE_ARCA=ON")
                endif()
                if(CCL_ENABLE_ARCB)
                    list(APPEND LIBCCL_CMAKE_ARGS "-DCCL_ENABLE_ARCB=ON")
                endif()
                list(APPEND LIBCCL_CMAKE_ARGS "-DCMAKE_SHARED_LINKER_FLAGS=${SHARED_LINKER_FLAGS}")

                if(DEFINED CMAKE_C_COMPILER)
                    list(APPEND LIBCCL_CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
                endif()
                list(JOIN LIBCCL_CMAKE_ARGS ";" LIBCCL_CMAKE_ARGS_LIST)

                externalproject_add(
                    oneCCL
                    PREFIX ${CMAKE_BINARY_DIR}/libccl
                    SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/libccl
                    BINARY_DIR ${CMAKE_BINARY_DIR}/libccl
                    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
                    CMAKE_ARGS ${LIBCCL_CMAKE_ARGS_LIST}
                    BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/libccl --parallel=${NPROC} --verbose
                    INSTALL_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/libccl --target install --verbose
                    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/libccl/_install/lib/libccl.so;${CMAKE_INSTALL_PREFIX}/libccl/src/libccl.so;${CMAKE_INSTALL_PREFIX}/libccl/src/libccl.so.1;${CMAKE_INSTALL_PREFIX}/libccl/src/libccl.so.1.0
                    BUILD_ALWAYS 1
                    LOG_OUTPUT_ON_FAILURE TRUE
                )

                add_library(ccl SHARED IMPORTED)
                set_target_properties(ccl PROPERTIES IMPORTED_NO_SYSTEM ON)

                add_dependencies(ccl oneCCL)
                set_target_properties(ccl PROPERTIES
                    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libccl/_install/lib/libccl.so"

                )
                target_include_directories(ccl BEFORE INTERFACE "${CMAKE_SOURCE_DIR}/deps/libccl/include")
                install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl/_install/lib/" DESTINATION "${DEST_LIB_DIR}" USE_SOURCE_PERMISSIONS)
                install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl/_install/env/" DESTINATION "${DEST_ENV_DIR}" USE_SOURCE_PERMISSIONS)
                install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl/_install/opt/" DESTINATION "${DEST_OPT_DIR}" USE_SOURCE_PERMISSIONS OPTIONAL)
                install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl/_install/share/" DESTINATION "${DEST_SHARE_DIR}" USE_SOURCE_PERMISSIONS OPTIONAL)
            endif()

            if(ONECCL_BUILD_CPU_PLUGIN)
                # Link intel libs statically
                if(COMPILER_SUPPORTS_STATIC_INTEL_LIB)
                    set(STATIC_INTEL_FLAG -static-intel)
                endif()

                set(LIBCCL_CPU_CMAKE_ARGS
                    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                    "-DCOMPUTE_BACKEND="
                    "-DBUILD_EXAMPLES=${ONECCL_LIBCCL_BUILD_EXAMPLES}"
                    "-DBUILD_FT=${ONECCL_LIBCCL_BUILD_FT}"
                    "-DBUILD_REG_TESTS=${ONECCL_LIBCCL_BUILD_REG_TESTS}"
                    "-DENABLE_EXTERN_TEMPLATE_INST=${ONECCL_LIBCCL_EXTERN_TEMPLATE_INST}"
                    "-DENABLE_ITT=${ONECCL_ENABLE_ITT}"
                    "-DONECCL_ENABLE_MPI=${ONECCL_ENABLE_MPI}"
                )
                set(SHARED_LINKER_FLAGS "-L${ONECCL2_LIB_DIR} -l:libccl.so.2 ${STATIC_INTEL_FLAG}")

                if(NOT "${ONECCL_LIBSTDCXX_PATH}" STREQUAL "" AND COMPILER_SUPPORTS_NOSTDLIBXX)
                    set(SHARED_LINKER_FLAGS "${SHARED_LINKER_FLAGS} -nostdlib++ -l:${ONECCL_LIBSTDCXX_PATH}")
                endif()

                if(CCL_ENABLE_ARCA)
                    list(APPEND LIBCCL_CPU_CMAKE_ARGS "-DCCL_ENABLE_ARCA=ON")
                endif()

                if(CCL_ENABLE_ARCB)
                    list(APPEND LIBCCL_CPU_CMAKE_ARGS "-DCCL_ENABLE_ARCB=ON")
                endif()

                list(APPEND LIBCCL_CPU_CMAKE_ARGS "-DCMAKE_SHARED_LINKER_FLAGS=${SHARED_LINKER_FLAGS}")

                if(DEFINED CMAKE_C_COMPILER)
                    list(APPEND LIBCCL_CPU_CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
                endif()
                list(JOIN LIBCCL_CPU_CMAKE_ARGS ";" LIBCCL_CPU_CMAKE_ARGS_LIST)

                externalproject_add(
                    oneCCLCPU
                    PREFIX ${CMAKE_BINARY_DIR}/libccl_cpu
                    SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/libccl
                    BINARY_DIR ${CMAKE_BINARY_DIR}/libccl_cpu
                    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
                    CMAKE_ARGS ${LIBCCL_CPU_CMAKE_ARGS_LIST}
                    BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/libccl_cpu --parallel=${NPROC} --verbose
                    INSTALL_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/libccl_cpu --target install --verbose
                    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/libccl_cpu/_install/lib/libccl.so;${CMAKE_INSTALL_PREFIX}/libccl_cpu/src/libccl.so;${CMAKE_INSTALL_PREFIX}/libccl_cpu/src/libccl.so.1;${CMAKE_INSTALL_PREFIX}/libccl_cpu/src/libccl.so.1.0
                    BUILD_ALWAYS 1
                    LOG_OUTPUT_ON_FAILURE TRUE
                )

                add_library(ccl_cpu SHARED IMPORTED)
                set_target_properties(ccl_cpu PROPERTIES IMPORTED_NO_SYSTEM ON)
                add_dependencies(ccl_cpu oneCCLCPU)
                set_target_properties(ccl_cpu PROPERTIES
                    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libccl_cpu/_install/lib/libccl.so"
                )
                target_include_directories(ccl_cpu BEFORE INTERFACE "${CMAKE_SOURCE_DIR}/deps/libccl/include")
                install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl_cpu/_install/lib/" DESTINATION "${DEST_LIB_DIR}/ccl/cpu/lib/" USE_SOURCE_PERMISSIONS)
                install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl_cpu/_install/env/" DESTINATION "${DEST_ENV_DIR}" USE_SOURCE_PERMISSIONS)
                install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl_cpu/_install/opt/" DESTINATION "${DEST_OPT_DIR}" USE_SOURCE_PERMISSIONS OPTIONAL)
                install(DIRECTORY "${CMAKE_BINARY_DIR}/libccl_cpu/_install/share/" DESTINATION "${DEST_SHARE_DIR}" USE_SOURCE_PERMISSIONS OPTIONAL)
            endif() # ONECCL_BUILD_CPU_PLUGIN
        endif()
    endif() # NOT WIN32
else() # Don't build libccl.so.1 and libccl.so.2, require presence of CCL_ROOT
    add_library(ccl SHARED IMPORTED)
    set_target_properties(ccl PROPERTIES
        IMPORTED_LOCATION "$ENV{CCL_ROOT}/lib/libccl.so.1"
        INTERFACE_INCLUDE_DIRECTORIES "$ENV{CCL_ROOT}/include"
    )

    add_library(ccl_cpu SHARED IMPORTED)
    set_target_properties(ccl_cpu PROPERTIES
        IMPORTED_LOCATION "$ENV{CCL_ROOT}/lib/ccl/cpu/lib/libccl.so.1"
        INTERFACE_INCLUDE_DIRECTORIES "$ENV{CCL_ROOT}/include"
    )

    add_library(oneCCL2 SHARED IMPORTED)
    set_target_properties(oneCCL2 PROPERTIES
        IMPORTED_LOCATION "$ENV{CCL_ROOT}/lib/libccl.so.2"
        INTERFACE_INCLUDE_DIRECTORIES "$ENV{CCL_ROOT}/include"
    )
endif() # ONECCL_BUILD_LIBS

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
add_subdirectory(utils)
add_subdirectory(benchmarks)

# Install public headers
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

# Install legacy ccl public headers
install(DIRECTORY deps/libccl/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include OPTIONAL)

# Install cmake config files
install(FILES cmake/oneCCL2Config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/oneCCL2)
install(FILES cmake/oneCCL2ConfigVersion.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/oneCCL2)

# Install vars.sh (override vars.sh from v1)
install(FILES etc/vars.sh DESTINATION ${CMAKE_INSTALL_PREFIX}/env)

# Install pkgconfig
install(DIRECTORY pkgconfig DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install_symlink(libccl.so.2 ${CMAKE_INSTALL_PREFIX}/lib/libccl.so)

message(STATUS "")
message(STATUS "")
message(STATUS "================= ONECCL BUILD SUMMARY =================")

foreach(var IN ITEMS
    ONECCL_HELGRIND_TESTING
    ONECCL_ASAN_TESTING
    ONECCL_UBSAN_TESTING
    ONECCL_THREAD_SANITIZER
    ONECCL_VALGRIND_TESTING
    CCL_ENABLE_ARCA
    CCL_ENABLE_ARCB
    ONECCL_ENABLE_TESTS
    ONECCL_ENABLE_BENCHMARKS
    ONECCL_FORTIFY_SOURCE
    ONECCL_ENABLE_ITT
)
    if(DEFINED ${var})
        message(STATUS "${var} = ${${var}}")
    else()
        message(STATUS "${var} = (not set)")
    endif()
endforeach()

message(STATUS "")
message(STATUS "Targets:")
foreach(tgt IN ITEMS ccl_legacy ccl_legacy_cpu ccl_null)
    if(TARGET ${tgt})
        message(STATUS "  ${tgt} (CONFIGURED)")
    else()
        message(STATUS "  ${tgt} (NOT PRESENT)")
    endif()
endforeach()

message(STATUS "========================================================")
message(STATUS "")
