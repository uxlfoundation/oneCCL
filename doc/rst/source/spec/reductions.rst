.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

==========
Reductions
==========

oneCCL specification defines the following reduction operations for :ref:`Allreduce`, :ref:`Reduce` and :ref:`ReduceScatter` collective operations:

.. code:: cpp

    enum class reduction
    {
        sum    = /* unspecified */,
        prod   = /* unspecified */,
        min    = /* unspecified */,
        max    = /* unspecified */,
        avg    = /* unspecified */,  
        custom = /* unspecified */
    };

reduction::sum
    elementwise summation
reduction::prod
    elementwise multiplication
reduction::min
    elementwise min
reduction::max
    elementwise max
reduction::avg
    arithmethic average
reduction::custom
    | specify user-defined reduction operation
    | the actual reduction function must be passed through ``reduction_fn`` operation attribute

:doc:`operation_attributes`
