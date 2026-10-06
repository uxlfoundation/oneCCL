.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _oneCCL-section:

======================
oneCCL Specification
======================

This section is the oneCCL specification: the requirements an implementation of |product_short|
must meet. It states what each object and communication operation *shall* do, independently of how
this library implements it.

It is the companion to the :doc:`Developer Reference </api>`, which documents the API this
implementation actually provides, generated from the headers. Read the specification for the
contract; read the Developer Reference for the exact signatures, overloads, and the behavior
specific to this implementation.

oneCCL is one element of the oneAPI specification, which the UXL Foundation maintains
dynamically: the shared programming model is described in the `oneAPI technical overview
<https://uxlfoundation.org/specifications/oneapi/technical-overview/>`_, while each
`specification element <https://uxlfoundation.org/specifications/oneapi/specification-elements/>`_
is defined in its own project documentation. These pages are that definition for oneCCL.

Namespaces
==========

The ``oneapi::ccl`` namespace shall contain public identifiers defined by the library.

The alternative ``ccl`` namespace shall be considered an alias for the ``oneapi::ccl`` namespace.

Contents
========

.. toctree::
   :maxdepth: 1

   generic_workflow.rst
   main_objects.rst
   datatypes.rst
   reductions.rst
   collective_operations.rst
   operation_attributes.rst
   operation_progress.rst
   group_calls.rst
   error_handling.rst
