# oneAPI Collective Communications Library (oneCCL) <!-- omit in toc --> <img align="right" width="200" height="100" src="https://raw.githubusercontent.com/uxlfoundation/artwork/e98f1a7a3d305c582d02c5f532e41487b710d470/foundation/uxl-foundation-logo-horizontal-color.svg">

[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Usage](#usage)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-collective-communication-library-ccl-release-notes.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://oneapi-src.github.io/oneCCL/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](CONTRIBUTING.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[License](LICENSE)

oneAPI Collective Communications Library (oneCCL) provides an efficient implementation of communication patterns used in deep learning.

oneCCL is governed by the [UXL Foundation](http://www.uxlfoundation.org) and is an implementation of the [oneAPI specification](https://spec.oneapi.io).

## Table of Contents <!-- omit in toc -->

- [oneAPI Collective Communications Library (oneCCL)  ](#oneapi-collective-communications-library-oneccl--)
  - [Installation](#installation)
    - [Linux build:](#linux-build)
  - [Usage](#usage)
    - [Launching Example Application](#launching-example-application)
    - [Common problems](#common-problems)
  - [Governance](#governance)


### SYCL support <!-- omit in toc -->

oneCCL supports SYCL. In order to enable it, compile with `icpx`/`icx-cl` compiler or any compiler supporting `find_package(IntelSYCL)` in cmake. 

To install Level Zero, refer to the instructions in [Intel(R) Graphics Compute Runtime repository](https://github.com/intel/compute-runtime/releases) or to the [installation guide](https://dgpu-docs.intel.com/installation-guides/index.html) for oneAPI users.

## Installation

### Linux build:

```sh
git submodule init # The two steps are not required if -DONECCL_USE_SYSTEM_LIBCCL=ON
git submodule update

mkdir build
cd build

cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=debug -DCMAKE_INSTALL_PREFIX=$(pwd)/_install
make -j install
```

## Usage

### Launching Example Application

Use the command:
```bash
$ source <install_dir>/env/setvars.sh
$ CCL_LOG_LEVEL=info mpirun -n 2 <install_dir>/examples/sycl/sycl
```

### Common problems

C API version of oneCCL introduced a new plugin based architecture. Running examples or tests might require adjusting `LD_LIBRARY_PATH`, so it points to at least one directory with correct plugin. It can be done using `vars.sh` script or manually like `export LD_LIBRARY_PATH=<build_dir>/plugins/legacy/:$LD_LIBRARY_PATH`, so test binaries linking to `libccl.so.2` through **RUNPATH** can still discover required plugin. 

You can also use **CCL_PLUGIN** environment variable to set name of the plugin (`ONECCL_LEGACY`/`ONECCL_LEGACY_CPU`/`ONECCL_NULL`) or a whole path to a shared object implementing oneCCL plugin interface (`<build_dir>/plugins/legacy/libccl_legacy.so`).

## Governance

The oneCCL project is governed by the UXL Foundation and you can get involved in this project in multiple ways. It is possible to join the [Special Interest Groups (SIG)](https://github.com/uxlfoundation/foundation) meetings where the group discuss and demonstrates work using the foundation projects. Members can also join the Open Source and Specification Working Group meetings.

You can also join the mailing lists for the [UXL Foundation](https://lists.uxlfoundation.org/g/main/subgroups) to be informed of when meetings are happening and receive the latest information and discussions.

## Contribute <!-- omit in toc -->

See [CONTRIBUTING](CONTRIBUTING.md) for more information.

## License <!-- omit in toc -->

Distributed under the Apache License 2.0 license. See [LICENSE](LICENSE) for more
information.

## Security Policy <!-- omit in toc -->

See [SECURITY](SECURITY.md) for more information.
