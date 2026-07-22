set(ONECCL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

if (NOT OUTPUT_DIR)
    set(OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}")
endif()

set(PROJECT_VERSION "2022.1.0")

configure_file("${ONECCL_ROOT}/cmake/templates/oneCCLConfig.cmake.in"
               "${OUTPUT_DIR}/oneCCLConfig.cmake"
               COPYONLY)
configure_file("${ONECCL_ROOT}/cmake/templates/oneCCLConfigVersion.cmake.in"
               "${OUTPUT_DIR}/oneCCLConfigVersion.cmake"
               @ONLY)

message(STATUS "oneCCL ${PROJECT_VERSION} configuration files were created in '${OUTPUT_DIR}'")

