# Getting Started with oneCCL

```{eval-rst}
.. note::
    This document describes the new C API that closely follows the NVIDIA Collective Communications Library (NCCL)* API standard. Documentation for the legacy C++ API can be found `here <../index.html>`_.
```

## Introduction

Welcome to the getting started guide for oneCCL, the Intel oneAPI Collective Communications Library. This document guides you through the installation process and provides instructions for building and running the example code using our latest NCCL like C API.

## Installation

### Linux build

oneCCL provides support only for Linux and leverages the cmake build system. Different build options are available to suit CPU only and CPU+GPU targets. To build oneCCL with GPU support, specify `-DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx`.

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

To experiment with oneCCL, use examples in the `examples/sycl` directory. The examples are built by default if the compiler supports SYCL (e.g. the `icpx` compiler). Before running the app, initialize the shell environment by sourcing the `vars.sh` script, installed in `<install_dir>/env/vars.sh`:  
`source <install_dir>/env/vars.sh`  

```{warning}
By default the `vars.sh` script will initialize the users environment using the Intel® MPI Library distribution bundled with oneCCL. To avoid overriding the configuration of custom systems (for example, systems using custom MPI impementation)  use `source vars.sh --ccl-bundled-mpi=no`.
```

```bash
$ source <install_dir>/env/vars.sh
$ CCL_LOG_LEVEL=info mpirun -n 2 <install_dir>/examples/sycl/sycl
```
