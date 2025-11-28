# Environment Variables

```{eval-rst}
.. note::
    This document describes the new C API that closely follows the NVIDIA Collective Communications Library (NCCL)* API standard. Documentation for the legacy C++ API can be found `here <../index.html>`_.
```

## General

```{eval-rst}
.. note::
    Please see C++ API documentation for additional variables used to tune oneCCL's behavior `here <../env-variables.html>`_.
```

### `CCL_PLUGIN`

**Syntax**

Select plugin by name or by path to binary implementing oneCCL's plugin interface:

```bash
CCL_PLUGIN="<plugin_name>"
CCL_PLUGIN="<plugin_path>"
```

Where:

- `<plugin_name>` is selected from the list of available plugins.
- `<plugin_path>` is absolute path to .so/.dll implementing oneCCL plugins interface.

**Example**

```bash
CCL_PLUGIN="ONECCL_LEGACY"
CCL_PLUGIN="/home/user/oneCCL/build/plugins/legacy/libccl_legacy.so"
```

**Arguments**

| `<plugin_name>`     | Description                                                                |
| ------------------- | -------------------------------------------------------------------------- |
| `ONECCL_LEGACY`     | Legacy oneCCL implementation based on SYCL for CPU and GPU workloads       |
| `ONECCL_LEGACY_CPU` | Legacy oneCCL implementation for CPU only workloads                        |
| `ONECCL_NULL`       | Empty implementation of oneCCL plugin, will always return `ONECCL_SUCCESS` |


**Description**

Use this environment variable to specify underlying implementation of all oneCCL functions.
