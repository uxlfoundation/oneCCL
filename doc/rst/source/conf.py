# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'oneCCL'
copyright = '2025'
author = 'Intel'

# The full version, including alpha/beta/rc tags
# release = '2022'

rst_prolog = """
.. |product_full| replace:: oneAPI Collective Communications Library (oneCCL)
.. |product_short| replace:: oneCCL
.. |mpi| replace:: Intel\ |reg|\  MPI Library
.. |reg| unicode:: U+000AE
.. |tm| unicode:: U+2122
.. |copy| unicode:: U+000A9
.. |base_tk| replace:: Intel\ |reg|\  oneAPI Base Toolkit 
.. |c_api| replace:: C API
.. |cpp_api| replace:: C++ API
"""

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_output_encoding = 'ascii'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_context = {
#    'css_files': [
#        '_static/style.css',  # override wide tables in RTD theme
#        ],
#    }


import sys
import os
import subprocess

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

extensions = [
    'sphinx.ext.autosectionlabel',
    'breathe',
#    'exhale',
    'sphinx-prompt',
    'sphinx_design',
    ]

breathe_projects = {
    "oneccl":"../../doxygen/xml"
}
breathe_default_project = "oneccl"

# Setup the exhale extension
#exhale_args = {
#    # These arguments are required
#    "containmentFolder":     "./api",
#    "rootFileName":          "library_root.rst",
#    "rootFileTitle":         "Library API",
#    "doxygenStripFromPath":  "..",
#    "fullApiSubSectionTitle": 'Full API'
#}

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'

html_theme = 'sphinx_book_theme'
html_logo = '_static/oneAPI-rgb-rev-100.png'
html_favicon = '_static/favicons.png'

def get_git_last_updated_date():
    try:
        result = subprocess.check_output(
            ['git', 'log', '-1', '--date=format:%d %B, %Y', '--format=%cd'],
            stderr=subprocess.STDOUT,
            encoding='utf-8'
        )
        return result.strip()
    except Exception as e:
        print(f"Error fetching last updated date: {e}")
        return "Unknown"

# Example usage
last_updated = get_git_last_updated_date()
print("Last updated:", last_updated)

# Theme options
html_theme_options = {
    'repository_url': 'https://github.com/oneapi-src/oneCCL',
    'path_to_docs': 'doc/rst/source',
    'use_issues_button': True,
    'use_edit_page_button': True,
    'repository_branch': 'master',
    'extra_footer': '<p align="right"><a href="https://www.intel.com/content/www/us/en/privacy/intel-cookie-notice.html">Cookies</a></p><br><div>Last updated: ' + last_updated + '</div>'
}

html_theme_options["logo"] = {"text": "oneCCL Documentation 2021.16"}

