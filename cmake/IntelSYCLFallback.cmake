# Fallback implementation for add_sycl_to_target when IntelSYCL package is not available
# This provides basic SYCL support for open-source compilers that support -fsycl

function(add_sycl_to_target)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(SYCL "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT SYCL_TARGET)
    message(FATAL_ERROR "add_sycl_to_target requires TARGET argument")
  endif()

  # Add -fsycl to specified sources or whole target
  if(SYCL_SOURCES)
    foreach(source ${SYCL_SOURCES})
      set_source_files_properties(${source} PROPERTIES COMPILE_OPTIONS "-fsycl")
    endforeach()
  else()
    target_compile_options(${SYCL_TARGET} PUBLIC "-fsycl")
  endif()

  # Add -fsycl to link options
  target_link_options(${SYCL_TARGET} PUBLIC "-fsycl")
endfunction()

# Create IntelSYCL::SYCL_CXX target for compatibility
if(NOT TARGET IntelSYCL::SYCL_CXX)
  add_library(IntelSYCL::SYCL_CXX INTERFACE IMPORTED)
  set_property(TARGET IntelSYCL::SYCL_CXX PROPERTY INTERFACE_COMPILE_OPTIONS "-fsycl")
  set_property(TARGET IntelSYCL::SYCL_CXX PROPERTY INTERFACE_LINK_OPTIONS "-fsycl")
endif()
