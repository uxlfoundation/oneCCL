add_subdirectory(null)

if(NOT WIN32 AND ONECCL_BUILD_CPU_PLUGIN)
    add_subdirectory(legacy_cpu)
endif()

if(COMPILER_SUPPORTS_SYCL)
    if(NOT WIN32)
        add_subdirectory(legacy)
    endif()
endif()
