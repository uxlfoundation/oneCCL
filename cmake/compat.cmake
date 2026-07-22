if(WIN32)
    # For easier exectuion of examples and tests in build directory
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # Enable exceptions by default
    set(COMMON_CXX_FLAGS "-EHsc" "-DNOGDI")

    # Force gtest to use correct version of runtime libraries
    set(gtest_force_shared_crt ON CACHE BOOL on)

    macro(install_symlink filepath sympath)
    endmacro(install_symlink)
else()
    macro(install_symlink filepath sympath)
        install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${filepath} ${sympath})")
        install(CODE "message(\"-- Created symlink: ${sympath} -> ${filepath}\")")
    endmacro(install_symlink)
endif()
