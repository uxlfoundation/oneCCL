# Contributing guidelines

We welcome community contributions to oneCCL. You can:

- Submit your changes directly with a [pull request](https://github.com/oneapi-src/oneCCL/pulls).
- Log a bug or feedback with an [issue](https://github.com/oneapi-src/oneCCL/issues).

Refer to our guidelines on [pull requests](#pull-requests) and [isssues](#issues) before you proceed.

## Issues

Use [GitHub issues]((https://github.com/oneapi-src/oneCCL/issues)) to:
- report an issue
- provide feedback
- make a feature request

**Note**: To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

## Pull requests

Before you submit a pull request, make sure that:

- You follow our [code contribution guidelines](#code-contribution-guidelines) and our [coding style](#coding-style).
- You provided the [requested details](#rfc-pull-requests) for new primitives or extended the existing [functional tests](#functional-tests) when fixing an issue.

**Note**: This project follows the
[GitHub flow](https://guides.github.com/introduction/flow/index.html). To get started with pull requests, see [GitHub howto](https://help.github.com/en/articles/about-pull-requests).

### RFC pull requests

It is strongly advised to open an RFC (request for comments) pull request when contributing new
primitives. Please provide the following details:

* The definition of the operation as a oneCCL primitive. It should include an interface and semantics. We welcome sketches for the interface, but the semantics should be fairly well-defined.

* A use case, including a model and parallelism scenario.

### Code contribution guidelines

The code must be:

* *Tested*: oneCCL uses `gtests` for lightweight functional testing.

* *Documented*: oneCCL uses `Doxygen` for inline comments in public header
  files that are used to build the API reference and  `reStructuredText` for the Developer Guide. See [oneCCL documentation](https://oneapi-src.github.io/oneCCL/) for reference.

* *Portable*: oneCCL supports CPU and GPU
  architectures as well as different compilers and run-times. The new code should be complaint
  with the [System Requirements](README.md#prerequisites).

### Coding style

The general principle is to follow the style of existing or surrounding code.

### Functional tests

How to run functional testing:

1. [Build and install oneCCL](README.md#Installation)
2. Make sure you are located in `<oneCCL directory>/<build directory>`
3. Source oneCCL: `source <oneCCL install directory>/env/setvars.sh`
4. Enter the test directory: `cd tests/functional`
5. Run tests: `ctest -VV -C default`

The results of the tests, including the pass rate, should be printed on the screen.
