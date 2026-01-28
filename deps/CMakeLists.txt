# Copyright 2016-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# --- ITT API ---
add_subdirectory(ittapi EXCLUDE_FROM_ALL)

# --- GoogleTest ---
set(INSTALL_GMOCK OFF CACHE BOOL "Don't install GoogleMock")
set(INSTALL_GTEST OFF CACHE BOOL "Don't install GoogleTest")
set(gtest_force_shared_crt OFF CACHE BOOL "Use static runtime for gtest on MSVC")

add_subdirectory(googletest EXCLUDE_FROM_ALL)
