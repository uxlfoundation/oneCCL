.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

==============
Error handling
==============

oneCCL error handling relies on the mechanism of C++ exceptions. If an error
occurs, it shall be propagated at the point of a function call where it is
caught using standard C++ error handling mechanism.

Exception classification
************************

Exception classification in oneCCL is aligned with C++ Standard Library classification. oneCCL introduces class that defines the base class in the hierarchy of oneCCL exception classes. All oneCCL routines throw exceptions inherited from this base class.

In the hierarchy of oneCCL exceptions, ``ccl::exception`` is the base class inherited from ``std::exception`` class. All other oneCCL exception classes are derived from this base class.

This specification does not require implementations to perform error-checking. However, if an implementation does provide error-checking, it shall use the following exception classes. Additional implementation-specific exception classes can be used for exceptional conditions not fitting any of these classes.

Common exceptions
*****************

.. csv-table::
    :header: "Exception class", "Description"
    :widths: 40, 60

    ".. _oneccl_exception:

    ``ccl::exception``", "Reports general unspecified error"
    ".. _oneccl_exception_invalid_argument:

    ``ccl::invalid_argument``", "Reports an error when arguments to the operation were rejected"
    ".. _oneccl_exception_host_bad_alloc:

    ``ccl::host_bad_alloc``", "Reports an error that occurred during memory allocation on the host"
    ".. _oneccl_exception_unimplemented:

    ``ccl::unimplemented``", "Reports an error when the requested operation is not implemented"
    ".. _oneccl_exception_unsupported:

    ``ccl::unsupported``", "Reports an error when the requested operation is not supported"
