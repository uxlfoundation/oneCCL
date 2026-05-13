#!/bin/bash

# Copyright 2016-2026 Intel Corporation
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


# Step 1: Accept the CXX compiler as an argument
CXX_COMPILER=$1

# Check if the compiler argument is provided
if [ -z "$CXX_COMPILER" ]; then
    echo "Error: No CXX compiler specified."
    echo "Usage: $0 <CXX_COMPILER>"
    exit 1
fi

# Step 2: Create a temporary C++ source file
SOURCE_FILE=$(mktemp /tmp/test_XXXXXXXX.cpp)
cat << 'EOF' > $SOURCE_FILE
#include <oneapi/ccl.hpp>
#include <oneapi/ccl.h>

int main() {
    int version;
    ccl::init();
    onecclGetVersion(&version);

    return 0;
}
EOF

# Step 3: Compile the source file into a temporary executable
EXECUTABLE=$(mktemp /tmp/test_exec_XXXXXXXX)

# Compile the binary with regular settings
COMPILE_COMMAND="$CXX_COMPILER $SOURCE_FILE -lccl -o $EXECUTABLE"
echo "Compiling with command: $COMPILE_COMMAND"
$COMPILE_COMMAND
COMPILE_STATUS=$?

# Check for compilation success, exit early if failed
if [ $COMPILE_STATUS -ne 0 ]; then
    echo "Compilation failed with status $COMPILE_STATUS."
    rm -f $SOURCE_FILE
    exit $COMPILE_STATUS
fi

echo "Compilation successful."

# Step 4: Execute the binary
echo "Running the binary..."
CCL_LOG_LEVEL=info $EXECUTABLE
EXECUTE_STATUS=$?

echo "Execution successful."

# Step 5: Clean up
rm -f $SOURCE_FILE $EXECUTABLE

# Exit with a success status
exit 0
