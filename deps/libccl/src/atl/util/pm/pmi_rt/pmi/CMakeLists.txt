#builds pmi

set(PMI_SRC
    simple_pmiutil.c
    simple_pmi.c)

set(COMMON_PMI_INC_DIRS
    ${PROJECT_SOURCE_DIR}/src/atl/util/pm/pmi_rt/pmi)

#special library that holds objects only
add_library(pmi-objects OBJECT ${PMI_SRC})
set_target_properties(pmi-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_include_directories(pmi-objects PUBLIC ${COMMON_PMI_INC_DIRS})
target_compile_definitions(pmi-objects PRIVATE HAVE_UNISTD_H HAVE_STDLIB_H HAVE_STRING_H HAVE_STRINGS_H)

#shared lib
add_library(pmi SHARED $<TARGET_OBJECTS:pmi-objects>)
target_include_directories(pmi PUBLIC INTERFACE ${COMMON_PMI_INC_DIRS})
if (NOT LIB_PMI_SO_VERSION AND NOT LIB_PMI_MAJOR_VERSION)
        set_target_properties(pmi PROPERTIES VERSION 1 SOVERSION 1.0)
else()
        set_target_properties(pmi PROPERTIES VERSION ${LIB_PMI_SO_VERSION} SOVERSION ${LIB_PMI_MAJOR_VERSION})
endif()
set_target_properties(pmi PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})

install(TARGETS pmi LIBRARY DESTINATION ${CCL_INSTALL_LIB})

#static lib
add_library(pmi-static STATIC $<TARGET_OBJECTS:pmi-objects>)
set_target_properties(pmi-static PROPERTIES OUTPUT_NAME pmi)
set_target_properties(pmi-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS pmi-static ARCHIVE DESTINATION ${CCL_INSTALL_LIB})
