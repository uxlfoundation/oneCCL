# Default installation path: <oneccl2_root>/lib/cmake/oneCCL2/
get_filename_component(_oneccl2_root "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(_oneccl2_root "${_oneccl2_root}/../../../" ABSOLUTE)


get_filename_component(_oneccl2_headers "${_oneccl2_root}/include" ABSOLUTE)
get_filename_component(_oneccl2_lib "${_oneccl2_root}/lib/libccl.so.2.0" ABSOLUTE)

if (EXISTS "${_oneccl2_headers}" AND EXISTS "${_oneccl2_lib}")
    if (NOT TARGET oneCCL)
        add_library(oneCCL2 SHARED IMPORTED)
        set_target_properties(oneCCL2 PROPERTIES
                             INTERFACE_INCLUDE_DIRECTORIES "${_oneccl2_headers}"
                             IMPORTED_LOCATION "${_oneccl2_lib}")
        unset(_oneccl2_headers)
        unset(_oneccl2_lib)

    endif()
else()
    if (NOT EXISTS "${_oneccl2_headers}")
        message(STATUS "oneCCL2: headers do not exist - ${_oneccl2_headers}")
    endif()
    if (NOT EXISTS "${_oneccl2_lib}")
        message(STATUS "oneCCL2: lib do not exist - ${_oneccl2_lib}")
    endif()
    set(oneCCL2_FOUND FALSE)
endif()
