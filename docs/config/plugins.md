# New features of oneCCL with C API

```{eval-rst}
.. note::
    This document describes the new C API that closely follows the NVIDIA Collective Communications Library (NCCL)* API standard. Documentation for the legacy C++ API can be found `here <../index.html>`_.
```

Alongside the new C API, oneCCL introduces a plugin system for dynamic backend selection. This design allows oneCCL to load the most suitable backend for the hardware platform at runtime.

### Available Plugins
* onecclLegacy
  * Supports both CPU and GPU collectives.
  * Selected by default if SYCL runtime is available.
  * Backend selection depends on the type of buffer and the stream argument:
    * For GPU buffers: stream must point to a SYCL queue.
    * For host buffers:
      * If stream is a SYCL queue, the collective is scheduled after previously submitted SYCL operations.
      * If stream is NULL, the collective executes immediately in blocking mode.

* onecclLegacyCPU
  * Designed for CPU-only collectives.
  * Ignores the stream argument (should be `NULL`).
    
```{hint}
To inspect plugin selection, enable logging using `CCL_LOG_LEVEL=info`.
```
### How Plugin Selection Works

* By default, oneCCL automatically loads the most appropriate plugin based on platform capabilities.
* Users can override this behavior using the [`CCL_PLUGIN`](project:./env.md#ccl-plugin) environment variable:
```c++
export CCL_PLUGIN=ONECCL_LEGACY_CPU
```