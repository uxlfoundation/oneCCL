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


SCRIPT_DIR=`cd $(dirname "$BASH_SOURCE") && pwd -P`
requirements_txt_filename=${SCRIPT_DIR}/requirements.txt

echo "creating temp directory"
temp_dir=$(mktemp -d)
echo

echo "creating venv"
python3 -m venv ${temp_dir}
echo

echo activating venv
source ${temp_dir}/bin/activate
echo

echo "installing requirements.txt"
pip install -r ${requirements_txt_filename}
echo

python_code="req_filename=\"${requirements_txt_filename}\""'
import sys
import re


replacements={}

for line in sys.stdin:
    package, from_version, to_version = re.findall("([^\\s]*) *([^\\s]*) *([^\\s]*)", line)[0]
    replacements[f"{package}=={from_version}"]=f"{package}=={to_version}"

requirements_txt_str = open(req_filename, "r").read()
for original, replacement in replacements.items():
    requirements_txt_str = requirements_txt_str.replace(original, replacement)
open(req_filename, "w").write(requirements_txt_str)
'

echo "updating requirements.txt"
pip list --outdated | tail -n +3 | python3 -c "${python_code}"
echo

echo "removing temporary directory with venv"
rm -rf "${temp_dir}"
echo

echo done
echo

echo "Please generate docs manually to assure requirements.txt compatibility."

