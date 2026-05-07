# Introduction

```{eval-rst}
.. note::
    In oneCCL version 2021.17 included with the 2025.3 oneAPI release, oneCCL has added support for a new **C API** that closely follows the NVIDIA Collective Communications Libary (NCCL)* API standard. Details about the new API, instructions on how to build, and run an example can be found `here <./index.html>`_.

    The existing C++ API will remain the default API for the 2021.17 release and can be found `here <../index.html>`_.
```

oneAPI Collective Communications Library (oneCCL) provides an efficient implementation of communication patterns used in deep learning. 

oneCCL features include:

- Built on top of lower-level communication middleware - MPI and [libfabric](https://github.com/ofiwg/libfabric).
- Optimized to drive scalability of communication patterns. It supports collectives and point-to-point (send/receive) primitives.
- Works across various interconnects: InfiniBand*, Cornelis Networks*, and Ethernet.
- Provides common API sufficient to support communication workflows within Deep Learning / distributed frameworks (such as [PyTorch](https://github.com/pytorch/pytorch), [Horovod](https://github.com/horovod/horovod)).

oneCCL package comprises the oneCCL Software Development Kit (SDK) and the MPI Runtime components.
