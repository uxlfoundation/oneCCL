# different functions, etc

function(set_lp_env)

    set(GCC_BF16_MIN_SUPPORTED "4.9.0")
    set(GCC_BF16_AVX512BF_MIN_SUPPORTED "10.0.0")
    set(GCC_BF16_AVX512BF_BINUTILS_MIN_SUPPORTED "2.33")

    set(ICC_BF16_AVX512BF_MIN_SUPPORTED "19.1.0")

    set(CLANG_BF16_MIN_SUPPORTED "9.0.0")
    set(CLANG_BF16_AVX512BF_MIN_SUPPORTED "9.3.0")

    if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel"
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM")
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "Clang"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${CLANG_BF16_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_BF16_MIN_SUPPORTED})
        )
        add_definitions(-DCCL_BF16_COMPILER)
        set(CCL_BF16_COMPILER ON)
    else()
        set(CCL_BF16_COMPILER OFF)
    endif()
    message(STATUS "BF16 AVX512F compiler: ${CCL_BF16_COMPILER}")

    execute_process(COMMAND ld -v
            OUTPUT_VARIABLE BINUTILS_VERSION_RAW
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)" BINUTILS_VERSION ${BINUTILS_VERSION_RAW})
    message(STATUS "binutils version: " "${BINUTILS_VERSION}")

    if (((${CMAKE_C_COMPILER_ID} STREQUAL "Intel"
        OR ${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM")
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${ICC_BF16_AVX512BF_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "Clang"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${CLANG_BF16_AVX512BF_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_BF16_AVX512BF_MIN_SUPPORTED}
            AND NOT ${BINUTILS_VERSION} VERSION_LESS ${GCC_BF16_AVX512BF_BINUTILS_MIN_SUPPORTED})
        )
        add_definitions(-DCCL_BF16_AVX512BF_COMPILER)
        set(CCL_BF16_AVX512BF_COMPILER ON)
    else()
        set(CCL_BF16_AVX512BF_COMPILER OFF)
    endif()
    message(STATUS "BF16 AVX512BF compiler: ${CCL_BF16_AVX512BF_COMPILER}")

    if (CCL_BF16_COMPILER)
        if ((${CMAKE_C_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM" OR  ${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
            add_definitions(-DCCL_BF16_TARGET_ATTRIBUTES)
            set(CCL_BF16_TARGET_ATTRIBUTES ON)
        else()
            set(CCL_BF16_TARGET_ATTRIBUTES OFF)
        endif()
        message(STATUS "BF16 target attributes: ${CCL_BF16_TARGET_ATTRIBUTES}")
    endif()

    option(CCL_BF16_GPU_TRUNCATE "Truncate BF16 in GPU operations" OFF)
    if (CCL_BF16_GPU_TRUNCATE)
        add_definitions(-DCCL_BF16_GPU_TRUNCATE)
    endif()
    message(STATUS "BF16 GPU truncate: ${CCL_BF16_GPU_TRUNCATE}")

    set(GCC_FP16_MIN_SUPPORTED "4.9.0")
    set(GCC_FP16_AVX512FP16_MIN_SUPPORTED "12.0.0")
    set(GCC_FP16_AVX512FP16_BINUTILS_MIN_SUPPORTED "2.38")

    set(ICX_FP16_AVX512FP16_MIN_SUPPORTED "2021.4.0")

    set(CLANG_FP16_MIN_SUPPORTED "9.0.0")
    set(CLANG_FP16_AVX512FP16_MIN_SUPPORTED "14.0.0")

    if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel"
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM")
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "Clang"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${CLANG_FP16_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_FP16_MIN_SUPPORTED})
        )
        add_definitions(-DCCL_FP16_COMPILER)
        set(CCL_FP16_COMPILER ON)
    else()
        set(CCL_FP16_COMPILER OFF)
    endif()
    message(STATUS "FP16 compiler: ${CCL_FP16_COMPILER}")

    if ((${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${ICX_FP16_AVX512FP16_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "Clang"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${CLANG_FP16_AVX512FP16_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_FP16_AVX512FP16_MIN_SUPPORTED}
	    AND NOT ${BINUTILS_VERSION} VERSION_LESS ${GCC_FP16_AVX512FP16_BINUTILS_MIN_SUPPORTED})
        )
        add_definitions(-DCCL_FP16_AVX512FP16_COMPILER)
        set(CCL_FP16_AVX512FP16_COMPILER ON)
    else()
        set(CCL_FP16_AVX512FP16_COMPILER OFF)
    endif()
    message(STATUS "FP16 AVX512FP16 compiler: ${CCL_FP16_AVX512FP16_COMPILER}")

    if (CCL_FP16_COMPILER)
        if ((${CMAKE_C_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM"
        OR ${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
            add_definitions(-DCCL_FP16_TARGET_ATTRIBUTES)
            set(CCL_FP16_TARGET_ATTRIBUTES ON)
        else()
            set(CCL_FP16_TARGET_ATTRIBUTES OFF)
        endif()
        message(STATUS "FP16 target attributes: ${CCL_FP16_TARGET_ATTRIBUTES}")
    endif()

    option(CCL_FP16_GPU_TRUNCATE "Truncate FP16 in GPU operations" OFF)
    if (CCL_FP16_GPU_TRUNCATE)
        add_definitions(-DCCL_FP16_GPU_TRUNCATE)
    endif()
    message(STATUS "FP16 GPU truncate: ${CCL_FP16_GPU_TRUNCATE}")

    set(LP_ENV_DEFINED 1 PARENT_SCOPE)

endfunction(set_lp_env)

function(set_sycl_env)
    set(ICX_SYCL_VEC_BF16_MIN_SUPPORTED "2024.2.0")

    if (CCL_ENABLE_SYCL
    AND ${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM"
    AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${ICX_SYCL_VEC_BF16_MIN_SUPPORTED})
        add_definitions(-DCCL_SYCL_VEC_SUPPORT_BF16)
	set(CCL_SYCL_VEC_SUPPORT_BF16 ON)
    else()
        set(CCL_SYCL_VEC_SUPPORT_BF16 OFF)
    endif()

    if (CCL_ENABLE_SYCL
    AND ${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
        set(CCL_SYCL_VEC_SUPPORT_FP16 OFF)
    else()
        add_definitions(-DCCL_SYCL_VEC_SUPPORT_FP16)
	set(CCL_SYCL_VEC_SUPPORT_FP16 ON)
    endif()

    if (CCL_ENABLE_SYCL)
        if (CCL_ENABLE_ARCA)
            add_definitions(-DCCL_SYCL_ENABLE_ARCA)
	    message(STATUS "Enable ARC A-series LL256 support")
        elseif (CCL_ENABLE_ARCB)
            add_definitions(-DCCL_SYCL_ENABLE_ARCB)
	    message(STATUS "Enable ARC B-series LL256 support")
        else()
            add_definitions(-DCCL_SYCL_ENABLE_PVC)
	    message(STATUS "Enable PVC LL256 support")
        endif()
    endif()
endfunction(set_sycl_env)

function(set_avx_env)

    set(GCC_AVX_MIN_SUPPORTED "4.9.0")
    set(CLANG_AVX_MIN_SUPPORTED "9.0.0")

    if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel"
        OR ${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM"
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "Clang"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${CLANG_AVX_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_AVX_MIN_SUPPORTED})
        )
        add_definitions(-DCCL_AVX_COMPILER)
        set(CCL_AVX_COMPILER ON)
    else()
        set(CCL_AVX_COMPILER OFF)
    endif()
    message(STATUS "AVX compiler: ${CCL_AVX_COMPILER}")

    if (CCL_AVX_COMPILER)
        if ((${CMAKE_C_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM" OR ${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
            add_definitions(-DCCL_AVX_TARGET_ATTRIBUTES)
            set(CCL_AVX_TARGET_ATTRIBUTES ON)
        else()
            set(CCL_AVX_TARGET_ATTRIBUTES OFF)
        endif()
        message(STATUS "AVX target attributes: ${CCL_AVX_TARGET_ATTRIBUTES}")
    endif()

    set(AVX_ENV_DEFINED 1 PARENT_SCOPE)

endfunction(set_avx_env)

function(check_compiler_version)

    set(GCC_MIN_SUPPORTED "4.8")
    set(ICC_MIN_SUPPORTED "15.0")
    set(CLANG_MIN_SUPPORTED "9.0")

    if(${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
        if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_MIN_SUPPORTED})
            message(FATAL_ERROR "gcc min supported version is ${GCC_MIN_SUPPORTED}, current version ${CMAKE_C_COMPILER_VERSION}")
        endif()
    elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
        if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${ICC_MIN_SUPPORTED})
            message(FATAL_ERROR "icc min supported version is ${ICC_MIN_SUPPORTED}, current version ${CMAKE_C_COMPILER_VERSION}")
        endif()
    elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
        if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${CLANG_MIN_SUPPORTED})
            message(FATAL_ERROR "clang min supported version is ${CLANG_MIN_SUPPORTED}, current version ${CMAKE_C_COMPILER_VERSION}")
        endif()
    elseif(${CMAKE_C_COMPILER_ID} STREQUAL "IntelLLVM")
        if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${ICC_MIN_SUPPORTED})
            message(FATAL_ERROR "icc min supported version is ${ICC_MIN_SUPPORTED}, current version ${CMAKE_C_COMPILER_VERSION}")
        endif()
    else()
        message(WARNING "Compilation with ${CMAKE_C_COMPILER_ID} was not tested, no warranty")
    endif()

endfunction(check_compiler_version)


function(get_vcs_properties VCS)

    if(${VCS} STREQUAL "git")
        # Get the current working branch
        execute_process(COMMAND git rev-parse --abbrev-ref HEAD
                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                        OUTPUT_VARIABLE GIT_BRANCH
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        )

        # Get the latest abbreviated commit hash of the working branch
        execute_process(COMMAND git log -1 --format=%h
                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                        OUTPUT_VARIABLE GIT_COMMIT_HASH
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        )
        message("-- Git branch: ${GIT_BRANCH}, commit: ${GIT_COMMIT_HASH}")
        set(VCS_INFO "(${GIT_BRANCH}/${GIT_COMMIT_HASH})" PARENT_SCOPE)
    endif()
endfunction(get_vcs_properties)


function(activate_compute_backend MODULES_PATH COMPUTE_BACKEND)
    string( TOLOWER "${COMPUTE_BACKEND}" COMPUTE_BACKEND)

    message("Search Compute Runtime by MODULES_PATH: ${MODULES_PATH}")
    list(APPEND CMAKE_MODULE_PATH "${MODULES_PATH}")

    if(COMPUTE_BACKEND STREQUAL "dpcpp")
        message ("COMPUTE_BACKEND=${COMPUTE_BACKEND} requested. Using DPC++ provider")
        SET (COMPUTE_BACKEND_LOAD_MODULE "IntelSYCL_level_zero"
                CACHE STRING
             "COMPUTE_BACKEND=${COMPUTE_BACKEND} requested. Using DPC++ provider")

        find_package(${COMPUTE_BACKEND_LOAD_MODULE} REQUIRED)

        if(NOT IntelSYCL_level_zero_FOUND)
            message(FATAL_ERROR "Failed to find IntelSYCL_level_zero")
        endif()

        # remember compilation flags, because flag required for OBJECTS target
        # but if we use `target_link_libraries`, then these flags applied to all compiler options
        # for c & cxx. But we need special flags for cxx only
        # So set it manually
        set (COMPUTE_BACKEND_CXXFLAGS_LOCAL "${COMPUTE_BACKEND_CXXFLAGS_LOCAL} ${INTEL_SYCL_FLAGS}")

        # remember current target for `target_link_libraries` in ccl
        set (COMPUTE_BACKEND_TARGET_NAME Intel::SYCL_level_zero)
        set (COMPUTE_BACKEND_TARGET_NAME Intel::SYCL_level_zero PARENT_SCOPE)
        message (STATUS "COMPUTE_BACKEND_TARGET_NAME: ${COMPUTE_BACKEND_TARGET_NAME} requested. Using DPC++ provider")
    endif()

    # extract target properties
    get_target_property(COMPUTE_BACKEND_INCLUDE_DIRS_LOCAL
                        ${COMPUTE_BACKEND_TARGET_NAME} INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(COMPUTE_BACKEND_LIBRARIES_LOCAL
                        ${COMPUTE_BACKEND_TARGET_NAME} INTERFACE_LINK_LIBRARIES)

    # When we use dpcpp compiler(dpcpp backends), use c++17 to be aligned with compiler
    if (${COMPUTE_BACKEND_TARGET_NAME} MATCHES "^Intel::SYCL.*")
        set(CMAKE_CXX_STANDARD 17 PARENT_SCOPE)
    # And use c++11 for all other cases
    else()
        set(CMAKE_CXX_STANDARD 11 PARENT_SCOPE)
    endif()

    # set output variables in the parent scope:
    # Only `COMPUTE_BACKEND_FLAGS` is actually required, because  the other flags are derived from
    # 'target_link_libraries'.
    # For simplicity, set all variables
    set(COMPUTE_BACKEND_FLAGS        ${COMPUTE_BACKEND_CXXFLAGS_LOCAL}      PARENT_SCOPE)
    set(COMPUTE_BACKEND_LIBRARIES    ${COMPUTE_BACKEND_LIBRARIES_LOCAL}     PARENT_SCOPE)
    set(COMPUTE_BACKEND_INCLUDE_DIRS ${COMPUTE_BACKEND_INCLUDE_DIRS_LOCAL}  PARENT_SCOPE)

endfunction(activate_compute_backend)

function(define_compute_backend)
    if (NOT DEFINED COMPUTE_BACKEND)
        message(STATUS "COMPUTE_BACKEND is not defined")
        if (${CMAKE_CXX_COMPILER} MATCHES ".*dpcpp")
            set(COMPUTE_BACKEND "dpcpp" CACHE STRING "compute backend value")
            message(STATUS "COMPUTE_BACKEND: ${COMPUTE_BACKEND} (set by default)")
        endif()
    else()
        message(STATUS "COMPUTE_BACKEND: ${COMPUTE_BACKEND} (set by user)")
    endif()
endfunction(define_compute_backend)

function(set_compute_backend COMMON_CMAKE_DIR)
    activate_compute_backend("${COMMON_CMAKE_DIR}" ${COMPUTE_BACKEND})

    # When we use dpcpp compiler(dpcpp backends), use c++17 to be aligned with compiler
    # Although the same thing is done in activate_compute_backend we need to set the variable here as
    # well bacause both set_compute_backend and activate_compute_backend can be called directly
    if (${COMPUTE_BACKEND_TARGET_NAME} MATCHES "^Intel::SYCL.*")
        set(CMAKE_CXX_STANDARD 17 PARENT_SCOPE)
    # And use c++11 for all other cases
    else()
        set(CMAKE_CXX_STANDARD 11 PARENT_SCOPE)
    endif()

    if (NOT COMPUTE_BACKEND_TARGET_NAME)
        message(FATAL_ERROR "Failed to find requested compute runtime: ${COMPUTE_BACKEND} in ${COMMON_CMAKE_DIR}")
    endif()
    message(STATUS "COMPUTE_BACKEND_TARGET_NAME: ${COMPUTE_BACKEND_TARGET_NAME}")

    if (${COMPUTE_BACKEND_TARGET_NAME} STREQUAL "Intel::SYCL" OR ${COMPUTE_BACKEND_TARGET_NAME} STREQUAL "Intel::SYCL_level_zero")

        set(CCL_ENABLE_SYCL ON PARENT_SCOPE)
        message(STATUS "Enable CCL SYCL support")

        set(CCL_ENABLE_ZE ON PARENT_SCOPE)
        message(STATUS "Enable CCL Level Zero support")

        set (CMAKE_CXX_FLAGS "-Wno-c++20-extensions" PARENT_SCOPE)

        execute_process(COMMAND icpx -v
            OUTPUT_VARIABLE ICPX_VERSION
            ERROR_VARIABLE ICPX_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        message(STATUS "DPC++ compiler version:\n" "${ICPX_VERSION}")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPUTE_BACKEND_FLAGS}")

    # need to pass these variables to overlying function
    set (COMPUTE_BACKEND_TARGET_NAME ${COMPUTE_BACKEND_TARGET_NAME} PARENT_SCOPE)
    set (COMPUTE_BACKEND_FLAGS ${COMPUTE_BACKEND_FLAGS} PARENT_SCOPE)
    set (COMPUTE_BACKEND_LIBRARIES ${COMPUTE_BACKEND_LIBRARIES} PARENT_SCOPE)
    set (COMPUTE_BACKEND_FLAGS ${COMPUTE_BACKEND_FLAGS} PARENT_SCOPE)
endfunction(set_compute_backend)

function(precheck_compute_backend)
    if (${COMPUTE_BACKEND} STREQUAL "dpcpp_level_zero")
        message(WARNING "Deprecated value \"dpcpp_level_zero\" is used for COMPUTE_BACKEND, switching to \"dpcpp\", please use it instead")
        set(COMPUTE_BACKEND "dpcpp")
        set(COMPUTE_BACKEND "dpcpp" PARENT_SCOPE)
    endif()

    if (COMPUTE_BACKEND)
        if (NOT ${COMPUTE_BACKEND} STREQUAL "dpcpp")
            message(FATAL_ERROR "Invalid value is used for COMPUTE_BACKEND: ${COMPUTE_BACKEND}\n"
                    "Possible values: <empty>(default), dpcpp")
        endif()
    endif()

endfunction(precheck_compute_backend)
