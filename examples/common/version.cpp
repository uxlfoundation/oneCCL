/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#include <stdio.h>

#include "base.hpp"

int main() {
    auto version = ccl::get_library_version();

    printf("\nCompile-time CCL library version:\nmajor: %d\nminor: %d\nupdate: %d\n"
           "Product: %s\nBuild date: %s\nFull: %s\n",
           CCL_MAJOR_VERSION,
           CCL_MINOR_VERSION,
           CCL_UPDATE_VERSION,
           CCL_PRODUCT_STATUS,
           CCL_PRODUCT_BUILD_DATE,
           CCL_PRODUCT_FULL);

    printf("\nRuntime CCL library version:\nmajor: %d\nminor: %d\nupdate: %d\n"
           "Product: %s\nBuild date: %s\nFull: %s\ncl_backend name: %s\n",
           version.major,
           version.minor,
           version.update,
           version.product_status,
           version.build_date,
           version.full,
           version.cl_backend_name.c_str());

    printf("\noneCCL specification version: %s\n", ONECCL_SPEC_VERSION);

    if (static_cast<int>(CCL_MAJOR_VERSION) == static_cast<int>(version.major) &&
        static_cast<int>(CCL_MINOR_VERSION) <= static_cast<int>(version.minor)) {
        printf("\nVersions are compatible\n");
    }
    else {
        perror("\nVersions are not compatible!\n");
        return -1;
    }

    printf("\nPASSED\n");

    return 0;
}
