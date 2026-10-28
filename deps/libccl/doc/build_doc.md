# Documentation Generation

## Description ##

The documentation is written using the restructured text markup language (also referred to as reST) and can be built with Doxygen and Sphinx.

## How to generate documentation ##

Install docker if absent and invoke

```bash
doc/build_doc_by_docker.sh
```

Generated documentation can be found in: `doc/rst/build/html` directory.

## Configure Doxygen ##

The Doxygen configuration lives in the `doc/Doxyfile` file.
Please refer to the [Doxygen configuration reference](http://www.doxygen.nl/manual/config.html) for more information.


## Configure Sphinx ##

You can create and modify Sphinx settings in the `doc/rst/source/conf.py` file.

For more details, please refer to the [Sphinx configuration reference](https://www.sphinx-doc.org/en/master/usage/configuration.html).
