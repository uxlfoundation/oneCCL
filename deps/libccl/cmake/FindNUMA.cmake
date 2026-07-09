# Find the NUMA library and includes
#
# NUMA_INCLUDE_DIR - where to find numa.h
# NUMA_LIBRARIES - list of libraries when using NUMA
# NUMA_FOUND - true if NUMA found

find_path(NUMA_INCLUDE_DIR
  NAMES numa.h numaif.h
  HINTS ${NUMA_ROOT_DIR}/include)

find_library(NUMA_LIBRARIES
  NAMES numa
  HINTS ${NUMA_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_LIBRARIES NUMA_INCLUDE_DIR)

if (NUMA_FOUND)
    message(STATUS "NUMA was found, include_dir: ${NUMA_INCLUDE_DIR}, libraries: ${NUMA_LIBRARIES}")
else()
    message(STATUS "NUMA was not found")
endif()
