# Installation <!-- omit in toc -->

## Prerequisites <!-- omit in toc -->

- Ubuntu* 18
- GNU*: C, C++ 4.8.5 or higher.

Refer to [System Requirements](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-collective-communication-library-system-requirements.html) for more details.

### SYCL support <!-- omit in toc -->
Intel(R) oneAPI DPC++/C++ Compiler with L0 v1.0 support


## General installation scenario <!-- omit in toc -->

```
cd oneccl
mkdir build
cd build
cmake ..
make -j install
```

If you need a "clear" build, create a new build directory and invoke `cmake` within it.

You can also do the following during installation:
- [Specify installation directory](#specify-installation-directory)
- [Specify the compiler](#specify-the-compiler)
- [Specify `SYCL` cross-platform abstraction level](#specify-sycl-cross-platform-abstraction-level)
- [Specify the build type](#specify-the-build-type)
- [Enable `make` verbose output](#enable-make-verbose-output)

## Specify installation directory

Modify `cmake` command as follows:

```
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/installation/directory
```

If no `-DCMAKE_INSTALL_PREFIX` is specified, oneCCL will be installed into `_install` subdirectory of the current
build directory, for example, `ccl/build/_install`.

## Specify the compiler

Modify `cmake` command as follows:

```
cmake .. -DCMAKE_C_COMPILER=your_c_compiler -DCMAKE_CXX_COMPILER=your_cxx_compiler
```

## Specify `SYCL` cross-platform abstraction level

If your CXX compiler requires SYCL, it is possible to specify it (DPC++ is supported for now).
Modify `cmake` command as follows:

```
cmake .. -DCMAKE_C_COMPILER=your_c_compiler -DCMAKE_CXX_COMPILER=icpx -DCOMPUTE_BACKEND=dpcpp
```

## Specify the build type

Modify `cmake` command as follows:

```
cmake .. -DCMAKE_BUILD_TYPE=[Debug|Release|RelWithDebInfo|MinSizeRel]
```

## Enable `make` verbose output

To see all parameters used by `make` during compilation
and linkage, modify `make` command as follows:

```
make -j VERBOSE=1
```