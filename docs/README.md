# Documentation

To generate HTML documentation, run the `generate_docs.py` script from any sub-directory of the repository (most likely `build`).
To display the proper version of oneCCL in the documentation title, set the `CCL_VERSION` variable before running the script.

```bash
cd build
$ CCL_VERSION=<your version, e.g. "2022.0.0"> python ../docs/generate_docs.py
```

Documentation can also be built using the build target 'docs' (see details below).

This script will create `./docs_build` sub-directory, where the intermediate and final files
will be created. HTML docs will be in the `./docs_build/generated/html` directory.

## Make docs

To run documentation generation via build target use CMake commands below.
To enable this target, python executable (in required version) has to be found in the system.

```bash
cmake -B build
cmake --build build --target docs
```

## Requirements

Script to generate HTML docs requires:

* [Doxygen](http://www.doxygen.nl/) at least v1.9.1
* [Python](https://www.python.org/downloads/) at least v3.8
* and python pip requirements, as defined in `docs/requirements.txt`

## Structure

Documentation is based on files inside `docs/config` directory and C headers in `include/oneapi` directory. File `index.rst` is the entrypoint of our site. It contains list of contents and links all the `.md` or `.rst` files together.