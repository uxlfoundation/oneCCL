.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

====================
Operation Attributes
====================

oneCCL specification defines communication operation attributes that serve as modifiers of an operation's behavior. Optionally, they may be passed to the corresponding communication operations.

oneCCL specification defines the following operation attribute classes:

- ``allgatherv_attr``
- ``allreduce_attr``
- ``alltoallv_attr``
- ``barrier_attr``
- ``broadcast_attr``
- ``reduce_attr``
- ``reduce_scatter_attr``

oneCCL specification defines attribute identifiers that may be used to fill operation attribute objects.

The list of common attribute identifiers that may be used for any communication operation:

.. code:: cpp

    enum class operation_attr_id {
        priority    = /* unspecified */,
        to_cache    = /* unspecified */,
        synchronous = /* unspecified */,
        match_id    = /* unspecified */

        last_value  = /* unspecified, equal to the largest of all the values above */
    };

operation_attr_id::priority
    the priority of the communication operation
operation_attr_id::to_cache
    | persistent/non-persistent communication operation
    | should be used in conjunction with ``match_id``
operation_attr_id::synchronous
    synchronous/asynchronous communication operation
operation_attr_id::match_id
    | the unique identifier of the operation
    | in conjunction with ``to_cache``, it enables the caching of the communication operation

The communication operation specific attribute identifiers may extend the list of common identifiers.

The list of attribute identifiers that may be used for :ref:`Allreduce`, :ref:`Reduce` and :ref:`ReduceScatter` collective operations:

.. code:: cpp

    enum class allreduce_attr_id {
        reduction_fn = /* unspecified */
    };

    enum class reduce_attr_id {
        reduction_fn = /* unspecified */
    };

    enum class reduce_scatter_attr_id {
        reduction_fn = /* unspecified */
    };

allreduce_attr_id::reduction_fn / reduce_attr_id::reduction_fn / reduce_scatter_attr_id::reduction_fn
    a function pointer for the custom reduction operation that follows the signature:

.. code:: cpp

        typedef void (*reduction_fn)
        (
            const void*,      /* in_buf    */
            size_t,           /* in_count  */
            void*,            /* inout_buf */
            size_t*,          /* out_count */
            datatype,         /* datatype  */
            const fn_context* /* context   */
        );

        typedef struct {
            const char* match_id;
            const size_t offset;
        } fn_context;

Creating an operation attribute object, which may be used in a corresponding communication operation:

.. code:: cpp

    template <class OpAttrType>
    OpAttrType ccl::create_operation_attr();

return ``OpAttrType``
    an object to hold attributes for a specific communication operation

The operation attribute classes shall provide ``get`` and ``set`` methods for getting and setting of values with specific attribute identifiers.
