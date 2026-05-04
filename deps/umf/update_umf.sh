#!/bin/bash
#
# Copyright 2016-2020 Intel Corporation
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
#

REPO=https://github.com/oneapi-src/unified-memory-framework.git

install_path_root=$(pwd)
repo_dir=unified-memory-framework

target_version="v1.0.x"

git clone $REPO
pushd $repo_dir
git checkout ${target_version}

# copy all umf headers to local include/umf directory
mkdir -p $install_path_root/include/umf
cp include/umf/*.h $install_path_root/include/umf/
cp include/umf/*.hpp $install_path_root/include/umf/ 2>/dev/null || true
cp include/umf/pools/*.h $install_path_root/include/umf/pools/ 2>/dev/null || true
cp include/umf/pools/*.hpp $install_path_root/include/umf/pools/ 2>/dev/null || true
cp include/umf/providers/*.h $install_path_root/include/umf/providers/ 2>/dev/null || true
cp include/umf/providers/*.hpp $install_path_root/include/umf/providers/ 2>/dev/null || true
cp include/umf.h $install_path_root/include/

popd

git add $install_path_root/include

git commit -m "deps: updated unified-memory-framework to ${target_version}"

rm -rf $install_path_root/$repo_dir
