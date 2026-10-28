<!---

Copyright (C) 2026 Intel Corporation

SPDX-License-Identifier: Apache

-->

File to cover guidelines for OneCCL project.

# C++ usage

* Keep platform/backend logic behind separate files as much as possible. If not, try to use compile-time flags such as CCL_ENABLE_SYCL, CCL_ENABLE_ZE, and CCL_ENABLE_UMF.
* Use static create/init methods returning std::unique_ptr when a constructor would otherwise need to throw.
* Consider grouping multiple function arguments with struct if total number of function arguments is more than 6.
* Use smart pointer when appropriate for explicit ownership, such as std::unique_ptr and std::shared_ptr
* Use constructor initializer list instead of in-class member initialization for member variables
* Use in-class member initialization for structs or data only classes
* Use enum class for scoped, type-safe enums
* Use constexpr for compile-time constants and if constexpr for compile-time if statements when possible
* Use `using` over `typedef` when possible
* Use c++ style casts instead of c style casts.
* Do not use infinite loops or goto statements
* Use `default` for all `switch` statements
* Avoid defining global constants with internal-linkage in header files; use inline variables from C++17 instead
* Use .hpp for C++ headers and .cpp for implementation files
* Use #pragma once as the default header protection style to avoid include-order issues
* Minimize include to what is strictly necessary (include what you use) and use forward declarations in headers
* Ensure self-contained and order-independent includes
* Favor the design of a self-explanatory code and simple comments over the excessive use of comments
  * if comments are needed, use double slash instead of block comments
  * break down complex algorithm into multiple helper or internal methods instead of single method
* Do not use `TODO`s or `FIXME`s in the code. Address them as a part of pull request or file an issue instead.
* For braces and indentation, use linux kernel conventions
  * Opening brace: Same line as if/else/for/function signature
  * Closing brace: On its own line (or paired with else/else if)
  * Single-line statement: Always enclose the body in braces even for single-line loops
  * Body indentation: 4 spaces instead of tab


# Naming conventions
* use snake_case for new files, class, struct, enum, and namespace names
* Use SCREAMING_SNAKE_CASE for constants and macro definitions
* Use the following short identifiers for data type when it is appropriate:
  1. the suffix _t for struct
  2. the prefix m_ for member variable
  3. the suffix _ptr for the pointer of the variable
  4. the suffix _type for template/type parameter sytle names or type descriptors
  5. the prefixes is_, can_, use_, enable_, and has_ for bool type variables and function returning boolean value.
     * is_ for state / identity,
     * can_ for capability, such as can_use_sycl_kernels,
     * use_ for feature / code path selection for,
     * enable_ for feature flags used for enviornment variables, such as enable_ze_cache,
     * has_ for presence check, such as has_error.
* Use short prefixes for subsystems, such as ccl_ for core implementation, and sycl_ for functions and variables used to call sycl APIs

```
bad examples : class BufferCache, constexpr size_t maxSS, struct st_raw_send_load, device *s_dev
good examples : class sycl_buffer_cache, constexpr size_t MAX_SS, struct raw_send_load_t, device *sycl_device_ptr
```

# Namespaces and API structure
* Keep ccl as the top-level namespace
* Keep implementation-only helpers under internal/detail-style namespaces
* Preserve existing PIMPL-related alias style with _impl to separate API declaration from the actual implementation

# Error Handling
* Use `CCL_FATAL`, `CCL_THROW`, and `CCL_ASSERT` instead of `assert` and `throw`:
  * Use `CCL_FATAL` when a failure is found and driver cannot proceed with normal execution. `CCL_FATAL` is implemented in Release and Debug builds and can be used in destructors.
  * Use `CCL_THROW` when a failure should be handled by throw...catch statements. `CCL_THROW` is implemented in Release and Debug builds and cannot be used in destructors.
  * Use `CCL_ASSERT` when a failure must not happen in terms of logic and can be used to check whether the code path falls into an undefined state and unexpected path. `CCL_ASSERT` is only implemented in Debug builds.


# Guideline specific to unit-level testing
* For unit-level tests, follow <method_name>_with_<condition1>_and_<condition2>_then_<expected_result> naming pattern,
indicate what is interesting in the test

bad examples :
```
TEST(ipc_handle_manager_tests, initialize)
```
good examples:
```
TEST(ipc_handle_manager_tests, initialize_with_null_comm_and_null_stream_then_not_initialized)
```

* Test behaviors instead of implementations, do not focus on adding a test per every function in the
class (avoid tests for setters and getters), focus on the functionality you are adding and how it changes
the driver behavior, do not bind tests to implementation.
* Make sure that test is fast, our test suite needs to complete in seconds for efficient development pace, as
a general rule test shouldn't be longer then 2ms in Debug build.

