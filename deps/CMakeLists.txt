
# --- Helper to debug missing submodules ---
function(require_submodule subdir expected_file)
  set(_path "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/${expected_file}")
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR
      "Missing files for git submodule '${subdir}'.\n"
      "Expected to find: ${_path}\n"
      "Submodules are likely not initialized. Please run:\n"
      "  git submodule update --init --recursive\n"
      "If you downloaded a source archive without submodules, fetch dependencies accordingly."
    )
  endif()
endfunction()

if(ONECCL_ENABLE_ITT)
  require_submodule(ittapi CMakeLists.txt)
  add_subdirectory(ittapi EXCLUDE_FROM_ALL)
endif()

# --- GoogleTest ---
set(INSTALL_GMOCK OFF CACHE BOOL "Don't install GoogleMock")
set(INSTALL_GTEST OFF CACHE BOOL "Don't install GoogleTest")
set(gtest_force_shared_crt OFF CACHE BOOL "Use static runtime for gtest on MSVC")

require_submodule(googletest CMakeLists.txt)
add_subdirectory(googletest EXCLUDE_FROM_ALL)

# --- Google Benchmark ---
set(BENCHMARK_ENABLE_TESTING  OFF CACHE BOOL "Disable Google Benchmark tests")
set(BENCHMARK_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Google Benchmark examples")
set(BENCHMARK_ENABLE_INSTALL  OFF CACHE BOOL "Disable Google Benchmark install")
set(BENCHMARK_INSTALL_DOCS    OFF CACHE BOOL "Disable Google Benchmark docs")
set(BENCHMARK_ENABLE_WERROR   OFF CACHE BOOL "Disable Google Benchmark docs")

set(THREADS_PREFER_PTHREAD_FLAG ON)
# TODO: Check when https://github.com/google/benchmark/issues/2057 is fixed
# require_submodule(benchmark CMakeLists.txt)
# add_subdirectory(benchmark EXCLUDE_FROM_ALL)

require_submodule(libccl CMakeLists.txt)

find_package(Threads REQUIRED)

