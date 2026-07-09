add_library(ccl_openmp SHARED openmp_ext.cpp)

if(NOT TARGET ccl)
	add_library(ccl INTERFACE)
	target_link_libraries(ccl INTERFACE -l:libccl.so.1)
endif()

target_include_directories(ccl_openmp PUBLIC ${SRC_INCLUDE_DIRS})
target_link_libraries(ccl_openmp PUBLIC OpenMP::OpenMP_CXX ccl)

set_target_properties(ccl_openmp PROPERTIES VERSION 0)
set_target_properties(ccl_openmp PROPERTIES SOVERSION 0.1)

install(TARGETS ccl_openmp LIBRARY DESTINATION ${CCL_INSTALL_LIB} OPTIONAL)
