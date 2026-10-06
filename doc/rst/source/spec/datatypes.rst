.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _`16-bit/half-precision floating point`: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
.. _`non-standard 16-bit floating point with 7-bit mantissa`: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

=========
Datatypes
=========

oneCCL specification defines the following datatypes that may be used for communication operations:

.. code:: cpp

    enum class datatype : int
    {
        int8            = /* unspecified */,
        uint8           = /* unspecified */,
        int16           = /* unspecified */,
        uint16          = /* unspecified */,
        int32           = /* unspecified */,
        uint32          = /* unspecified */,
        int64           = /* unspecified */,
        uint64          = /* unspecified */,

        float16         = /* unspecified */,
        float32         = /* unspecified */,
        float64         = /* unspecified */,
        bfloat16        = /* unspecified */,

        last_predefined = /* unspecified, equal to the largest of all the values above */
    };

datatype::int8
    8 bits signed integer
datatype::uint8
    8 bits unsigned integer
datatype::int16
    16 bits signed integer
datatype::uint16
    16 bits unsigned integer
datatype::int32
    32 bits signed integer
datatype::uint32
    32 bits unsigned integer
datatype::int64
    64 bits signed integer
datatype::uint64
    64 bits unsigned integer
datatype::float16
    `16-bit/half-precision floating point`_
datatype::float32
    32-bit/single-precision floating point
datatype::float64
    64-bit/double-precision floating point
datatype::bfloat16
    `non-standard 16-bit floating point with 7-bit mantissa`_


.. note::
    Support for ``datatype::float16`` is optional


.. _Custom Datatypes:

Custom Datatypes
****************

oneCCL specification defines the way to register and deregister a custom datatype
using the ``datatype_attr`` attribute object.


The list of identifiers that may be used to fill an attribute object:

.. code:: cpp

    enum class datatype_attr_id {
        size = /* unspecified */
    };

datatype_attr_id::size
    the size of the datatype in bytes


Creating a datatype attribute object, which may used to register custom datatype:

.. code:: cpp

    datatype_attr ccl::create_datatype_attr();

return ``datatype_attr``
    an object containing attributes for the custom datatype


Registering a custom datatype to be used in communication operations:

.. code:: cpp

    datatype ccl::register_datatype(const datatype_attr& attr);

attr
    the datatype's attributes
return ``datatype``
    the handle for the custom datatype


Deregistering a custom datatype:

.. code:: cpp

    void ccl::deregister_datatype(datatype dtype);

dtype
    the handle for the custom datatype


Retrieving a datatype size in bytes:

.. code:: cpp

    size_t ccl::get_datatype_size(datatype dtype);

dtype
    the datatype's handle
return ``size_t``
    datatype size in bytes
