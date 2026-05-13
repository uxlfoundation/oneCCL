# API Documentation

```{eval-rst}
.. note::
    In oneCCL version 2021.17 included with the 2025.3 oneAPI release, oneCCL has added support for a new **C API** that closely follows the NVIDIA Collective Communications Libary (NCCL)* API standard. Details about the new API, instructions on how to build, and run an example can be found `here <./index.html>`_.

    The existing C++ API will remain the default API for the 2021.17 release and can be found `here <../index.html>`_.
```

## Communicator Creation API

This section includes functions related to initializing and managing communicators in oneCCL.
```{eval-rst}
.. doxygengroup:: CommunicatorCreation
   :content-only:
```

## Collective Functions API

This section includes functions related to collective communication operations.
```{eval-rst}
.. doxygengroup:: CollectiveFunctions
   :content-only:
```

## Types API

This section includes types and definitions used throughout the oneCCL API.
```{eval-rst}
.. doxygengroup:: Types
   :content-only:
```
