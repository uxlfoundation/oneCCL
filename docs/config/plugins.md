# Plugin architecture

```{eval-rst}
.. note::
    In oneCCL version 2021.17 included with the 2025.3 oneAPI release, oneCCL has added support for a new **C API** that closely follows the NVIDIA Collective Communications Libary (NCCL)* API standard. Details about the new API, instructions on how to build, and run an example can be found `here <./index.html>`_.

    The existing C++ API will remain the default API for the 2021.17 release and can be found `here <../index.html>`_.
```

Alongside introduction of new API for oneCCL we introduced a concept of oneCCL plugins. The plugin interface was designed to provide dynamic selection of backends for different hardware platforms. Currently oneCCL provides two plugins - `onecclLegacy` and `onecclLegacyCpu`, which are based on `libccl.so.1.0` and provide the same functional capabilities. On startup oneCCL will load the most appropriate plugin unless user specifies [`CCL_PLUGIN`](project:./env.md#ccl-plugin) environment variable to override it.

## Plugins and their features
 * `onecclLegacy` - supports both CPU and GPU collectives, selected by default if `libsycl.so.8` is available on the platform. The plugin will select CPU or GPU backend on following conditions:
   * Operations such as the `onecclAllreduce` collective take as argument a `void *stream`. When the collective uses GPU buffers, the `stream` should be a pointer to a SYCL queue. 
   * If the collective uses a host buffer, the `stream` can be a pointer to a SYCL queue or `NULL`. When the `stream` is a SYCL queue, oneCCL will use the stream to submit the new collective after the operations previously submitted to the SYCL queue, even if the collective itself executes in the host. When the `stream` is `NULL`, the collective executes right away in a blocking mode. 
 * `onecclLegacyCPU` - The `stream` argument should be `NULL`, but in any case, the implementation will ignore the value of the argument. 
  
```{eval-rst}
.. hint::
    You can inspect plugin selection process using environment variable CCL_LOG_LEVEL=info.
```
