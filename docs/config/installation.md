# Getting Started with oneCCL

```{eval-rst}
.. note::
    In oneCCL version 2021.17 included with the 2025.3 oneAPI release, oneCCL has added support for a new **C API** that closely follows the NVIDIA Collective Communications Libary (NCCL)* API standard. Details about the new API, instructions on how to build, and run an example can be found `here <./index.html>`_.

    The existing C++ API will remain the default API for the 2021.17 release and can be found `here <../index.html>`_.
```

## Introduction

Welcome to the getting started guide for oneCCL with C API, the new API for the Intel oneAPI Collective Communications Library. This guide will walk you through the installation process and provide instructions to build and run example code using oneCCL.

## Installation

### Linux build

```sh
git clone https://github.com/uxlfoundation/oneCCL
cd oneCCL

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

C API oneCCL introduced a new plugin based architecture. Running examples or tests might require adjusting `LD_LIBRARY_PATH`, so it points to at least one directory with correct plugin. It can be done using `vars.sh` script or manually like `export LD_LIBRARY_PATH=<build_dir>/plugins/legacy/:$LD_LIBRARY_PATH`, so test binaries linking to `libccl.so.2` through **RUNPATH** can still discover required plugin. 

You can also use **CCL_PLUGIN** environment variable to set name of the plugin (`ONECCL_LEGACY`/`ONECCL_LEGACY_CPU`/`ONECCL_NULL`) or a whole path to a shared object implementing oneCCL plugin interface (`<build_dir>/plugins/legacy/libccl_legacy.so`).
