# oneAPI Collective Communications Library (oneCCL) <!-- omit in toc --> <img align="right" width="200" height="100" src="https://raw.githubusercontent.com/uxlfoundation/artwork/e98f1a7a3d305c582d02c5f532e41487b710d470/foundation/uxl-foundation-logo-horizontal-color.svg">

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/oneapi-src/oneCCL/badge)](https://scorecard.dev/viewer/?uri=github.com/oneapi-src/oneCCL)
[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Usage](#usage)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-collective-communication-library-ccl-release-notes.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://oneapi-src.github.io/oneCCL/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](CONTRIBUTING.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[License](LICENSE)

oneAPI Collective Communications Library (oneCCL) provides an efficient implementation of communication patterns used in deep learning.

oneCCL is integrated into:
* [Horovod\*](https://github.com/horovod/horovod) (distributed training framework). Refer to [Horovod with oneCCL](https://github.com/horovod/horovod/blob/master/docs/oneccl.rst) for details.
* [PyTorch\*](https://github.com/pytorch/pytorch) (machine learning framework). Refer to [PyTorch bindings for oneCCL](https://github.com/intel/torch-ccl) for details.

oneCCL is governed by the [UXL Foundation](http://www.uxlfoundation.org) and is an implementation of the [oneAPI specification](https://spec.oneapi.io).

## Table of Contents <!-- omit in toc -->

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Launching Example Application](#launching-example-application)
    - [Using external MPI](#using-external-mpi)
  - [Setting workers affinity](#setting-workers-affinity)
    - [Automatic setup](#automatic-setup)
    - [Explicit setup](#explicit-setup)
  - [Using oneCCL package from CMake](#using-oneccl-package-from-cmake)
    - [oneCCLConfig files generation](#onecclconfig-files-generation)
- [Additional Resources](#additional-resources)
  - [Blog Posts](#blog-posts)
  - [Workshop Materials](#workshop-materials)
- [Notice of Deprecation](#notice-of-deprecation)
  
## Prerequisites 

See [System Requirements](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-collective-communication-library-system-requirements.html) to learn about hardware and software requirements before getting started with oneCCL.

## Installation

General installation scenario:

```
cd oneCCL
mkdir build
cd build
cmake ..
make -j install
```

If you need a clean build, create a new build directory and invoke `cmake` within it.

You can also do the following during installation:
- [Specify installation directory](INSTALL.md#specify-installation-directory)
- [Specify the compiler](INSTALL.md#specify-the-compiler)
- [Specify `SYCL` cross-platform abstraction level](INSTALL.md#specify-sycl-cross-platform-abstraction-level)
- [Specify the build type](INSTALL.md#specify-the-build-type)
- [Enable `make` verbose output](INSTALL.md#enable-make-verbose-output)

## Usage

### Launching Example Application

Use the command:
```bash
$ source <install_dir>/env/setvars.sh
$ mpirun -n 2 <install_dir>/examples/benchmark/benchmark
```

#### Using external mpi

The ccl-bundled-mpi flag in vars.sh can take values "yes" or "no" to control if bundled Intel MPI should be used or not. Current default is "yes", which means that oneCCL temporarily overrides the mpi implementation in use.

In order to suppress the behavior and use user-supplied or system-default mpi use the following command *instead* of sourcing `setvars.sh`:

```bash
$ source <install_dir>/env/vars.sh --ccl-bundled-mpi=no
```

The mpi implementation will not be overridden. Please note that, in this case, user needs to assure the system finds all required mpi-related binaries.

### Setting workers affinity

There are two ways to set worker threads (workers) affinity: [automatically](#setting-affinity-automatically) and [explicitly](#setting-affinity-explicitly).

#### Automatic setup

1. Set the `CCL_WORKER_COUNT` environment variable with the desired number of workers per process.
2. Set the `CCL_WORKER_AFFINITY` environment variable with the value `auto`.

Example:
```
export CCL_WORKER_COUNT=4
export CCL_WORKER_AFFINITY=auto
```
With the variables above, oneCCL will create four workers per process and the pinning will depend from process launcher.

If an application has been launched using `mpirun` that is provided by oneCCL distribution package then workers will be automatically pinned to the last four cores available for the launched process. The exact IDs of CPU cores can be controlled by `mpirun` parameters.

Otherwise, workers will be automatically pinned to the last four cores available on the node.

---

#### Explicit setup

1. Set the `CCL_WORKER_COUNT` environment variable with the desired number of workers per process.
2. Set the `CCL_WORKER_AFFINITY` environment variable with the IDs of cores to pin local workers.

Example:
```
export CCL_WORKER_COUNT=4
export CCL_WORKER_AFFINITY=3,4,5,6
```
With the variables above, oneCCL will create four workers per process and pin them to the cores with the IDs of 3, 4, 5, and 6 respectively.

### Using oneCCL package from CMake

`oneCCLConfig.cmake` and `oneCCLConfigVersion.cmake` are included into oneCCL distribution.

With these files, you can integrate oneCCL into a user project with the [find_package](https://cmake.org/cmake/help/latest/command/find_package.html) command. Successful invocation of `find_package(oneCCL <options>)` creates imported target `oneCCL` that can be passed to the [target_link_libraries](https://cmake.org/cmake/help/latest/command/target_link_libraries.html) command.

For example:

```cmake
project(Foo)
add_executable(foo foo.cpp)

# Search for oneCCL
find_package(oneCCL REQUIRED)

# Connect oneCCL to foo
target_link_libraries(foo oneCCL)
```
#### oneCCLConfig files generation

To generate oneCCLConfig files for oneCCL package, use the provided [`cmake/scripts/config_generation.cmake`](/cmake/scripts/config_generation.cmake) file:

```
cmake [-DOUTPUT_DIR=<output_dir>] -P cmake/script/config_generation.cmake
```

### OS File Descriptors 

oneCCL uses [Level Zero IPC handles](https://spec.oneapi.io/level-zero/latest/core/PROG.html#memory-1) so that a process can access a memory allocation done by a different process. 
However, these IPC handles consume OS File Descriptors (FDs). As a result, to avoid running out of OS FDs, we recommend to increase the default limit of FDs in the system for applications running with oneCCL and GPU buffers. 

The number of FDs required is application-dependent, but the recommended limit is ``1048575``. This value can be modified with the ulimit command.  

## Governance

The oneCCL project is governed by the UXL Foundation and you can get involved in this project in multiple ways. It is possible to join the [Special Interest Groups (SIG)](https://github.com/uxlfoundation/foundation) meetings where the group discuss and demonstrates work using the foundation projects. Members can also join the Open Source and Specification Working Group meetings.

You can also join the mailing lists for the [UXL Foundation](https://lists.uxlfoundation.org/g/main/subgroups) to be informed of when meetings are happening and receive the latest information and discussions.

## Additional Resources

### Blog Posts

- [Optimizing DLRM by using PyTorch with oneCCL Backend](https://pytorch.medium.com/optimizing-dlrm-by-using-pytorch-with-oneccl-backend-9f85b8ef6929)
- [Intel MLSL Makes Distributed Training with MXNet Faster](https://medium.com/apache-mxnet/intel-mlsl-makes-distributed-training-with-mxnet-faster-7186ad245e81)

### Workshop Materials

- oneAPI, oneCCL and OFI: Path to Heterogeneous Architecure Programming with Scalable Collective Communications: [recording](https://www.youtube.com/watch?v=ksiZ90EtP98&feature=youtu.be) and [slides](https://www.openfabrics.org/wp-content/uploads/2020-workshop-presentations/502.-OFA-Virtual-Workshop-2020-oneCCL-v5.pdf)

## Notice of Deprecation 

### Deprecation of C++ API
- In oneCCL version 2021.17 included with the 2025.3 oneAPI release, oneCCL will add support for a new C API that closely follows the NVIDIA Collective Communications Libary (NCCL)* API standard. The existing C++ API will remain available and will remain the default API for the 2021.17 release. Details explaining how an application may link against and use the new API will be shared in this release.

See the [oneCCL C API RFC document](https://github.com/uxlfoundation/oneCCL/tree/rfcs/rfcs/20240806-c-api) to view the proposed API and provide any feedback. 

- In oneCCL version 2022.0 included with the 2026.0 oneAPI release, oneCCL will use the new NCCL like C API by default. This is a breaking change. The legacy C++ API will remain available, and details explaining how an application may link against and use the legacy API will be included in the 2022.0 release.

Applications cannot use both the C and C++ APIs simultaneously.

Support for the legacy C++ API shall remain in the release until future notice. The schedule for legacy API removal will be announced here.

## Contribute <!-- omit in toc -->

See [CONTRIBUTING](CONTRIBUTING.md) for more information.

## License <!-- omit in toc -->

Distributed under the Apache License 2.0 license. See [LICENSE](LICENSE) for more
information.

## Security Policy <!-- omit in toc -->

See [SECURITY](SECURITY.md) for more information.

## Notices and Disclaimers

Intel technologies may require enabled hardware, software or service activation.

No product or component can be absolutely secure.

Your costs and results may vary.

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.

No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document.

The products described may contain design defects or errors known as errata which may cause the product to deviate from published specifications. Current characterized errata are available on request.

Intel disclaims all express and implied warranties, including without limitation, the implied warranties of merchantability, fitness for a particular purpose, and non-infringement, as well as any warranty arising from course of performance, course of dealing, or usage in trade.
