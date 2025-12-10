# How to upgrade from C++ API

```{eval-rst}
.. note::
    This document describes the new C API that closely follows the NVIDIA Collective Communications Library (NCCL)* API standard. Documentation for the legacy C++ API can be found `here <../index.html>`_.
```

### How to compile using oneCCL **C API**

* New C API is defined in `oneapi/ccl.h` header file.
* The API is implemented in a new version of the library - `libccl.so.2`, but this new API is not yet the default. This new API shall become the default in our next major release.
* Symbolic link `libccl.so` is still pointing to the C++ API.
* To link with the C API use linker flag such as `-l:libccl.so.2`.
* Sample cmake target:  :
```cmake
add_library(oneCCL INTERFACE)
target_include_directories(oneCCL INTERFACE $ENV{ONECCL_ROOT}/include)
target_link_libraries(oneCCL INTERFACE -l:libccl.so.2)
```

### Key differences between the two APIs

|               | C++ API                                   | C API                        |
|---------------|-------------------------------------------|------------------------------|
| Header file   | oneapi/ccl.hpp                            | oneapi/ccl.h                 |
| Shared object | libccl.so or libccl.so.1 or libccl.so.1.0 | libccl.so.2 or libccl.so.2.0 |
| APIs          | <a href="../api.html">C++ API</a>         | [C API](project:./api.md)    |
