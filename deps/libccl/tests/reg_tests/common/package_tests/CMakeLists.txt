cmake_minimum_required(VERSION 3.20)

# pkg-config test

set(pkg_sources
    pkgconfig_support_checker.cpp)

set(CCL_INSTALL_TESTS "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(PkgConfig)

set(PKG_CCL_LIB "ccl")

pkg_check_modules(IMPI_PKG IMPORTED_TARGET impi)
pkg_check_modules(CCL_PKG IMPORTED_TARGET ccl)

if ("${IMPI_PKG_FOUND}" AND "${CCL_PKG_FOUND}")
    message(STATUS "CCL_PKG_LDFLAGS = ${CCL_PKG_LDFLAGS}")
    message(STATUS "CCL_PKG_CFLAGS = ${CCL_PKG_CFLAGS}")

    foreach(src ${pkg_sources})
       get_filename_component(executable ${src} NAME_WE)
       add_executable(${executable} ${src})
       target_link_libraries(${executable} PRIVATE ${CCL_PKG_LDFLAGS})
       target_include_directories(${executable} PRIVATE ${CCL_PKG_CFLAGS})
       install(TARGETS ${executable} RUNTIME DESTINATION ${CCL_INSTALL_TESTS} OPTIONAL)
    endforeach(src ${pkg_sources})
else()
    message(WARNING "pkg-config not found")
endif()

