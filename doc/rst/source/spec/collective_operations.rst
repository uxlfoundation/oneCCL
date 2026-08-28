.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

=====================
Collective Operations
=====================

oneCCL specification defines the following collective communication operations:

- :ref:`Allgather`
- :ref:`Allgatherv`
- :ref:`Allreduce`
- :ref:`Alltoallv`
- :ref:`Barrier`
- :ref:`Broadcast`
- :ref:`Reduce`
- :ref:`ReduceScatter`

These operations are collective, meaning that all participants (ranks) of oneCCL communicator should make a call.
The order of collective operation calls should be the same across all ranks.

``communicator`` shall provide the ability to perform communication operations either on host or device memory buffers depending on the ``device`` used to create the communicator. Additionally, communication operations shall accept an execution context (stream) and may accept a vector of events that the communication operation should depend on, that is, input dependencies. The output ``event`` object shall provide the ability to track the progress of the operation.

.. note::
    Support for handling of input events is optional

``BufferType`` is used below to define the C++ type of elements in data buffers (``buf``, ``send_buf`` and  ``recv_buf``) of communication operations. At least the following types shall be supported: ``[u]int{8/16/32/64}_t``, ``float``, ``double``. The explicit ``datatype`` parameter shall be used to enable data types which cannot be inferred from the function arguments.

.. note::
    See also: :ref:`Custom Datatypes`

The communication operation accepts a ``stream`` object. If a communicator is created from ``native_device_type``, then the stream shall translate to ``native_stream_type`` created from the corresponding device.

The communication operation may accept attribute object. If that parameter is missed, then the default attribute object is used (default_<operation_name>_attr). The default attribute object shall be provided by the library.

.. note::
    See also: :doc:`operation_attributes`

If the arguments provided to a communication operation call do not comply to the requirements of the operation, the behavior is undefined unless it is specified otherwise.


.. _Allgather:

Allgather
*********

Allgather is a collective communication operation that collects the ``send_count`` elements from all the ranks within the communicator and places the results into ``recv_buf``, in such a way that data from rank ``i`` can be found at offset rank ``i * count``. The resulting data in the output ``recv_buf`` buffer is the same for each rank. 


Allgather is in place when ``sendbuff == recvbuff + rank * send_count``. 

.. code:: cpp

     template<class BufferType> 
     event ccl::allgather(const BufferType* send_buf, 
                          BufferType* recv_buf, 
                          size_t send_count, 
                          const communicator& comm, 
                          const stream& stream, 
                          const allgather_attr& attr = default_allgather_attr, 
                          const vector_class<event>& deps = {}); 

     event ccl::allgather(const void* send_buf, 
                          void* recv_buf, 
                          size_t send_count, 
                          datatype dtype,  
                          const communicator& comm, 
                          const stream& stream, 
                          const allgather_attr& attr = default_allgather_attr, 
                          const vector_class<event>& deps = {}); 



send_buf 
    The buffer with send_count elements of BufferType that stores local data to be gathered 

recv_buf [out] 
    The buffer to store gathered result of BufferTuype, must be large enough to hold values from all ranks, i.e., size should be equal do BufferType * send_count 

send_count 
    The number of elements of type BufferType in send_buf 

dtype 
    The datatype of elements in send_buf and recv_buf must be skipped if BufferType can be inferred otherwise must be passed explicitly 

comm 
    The communicator that defines a group of ranks for the operation 

stream 
    The stream associated with the operation  

attr 
    Optional attributes to customize the operation 

deps 
    An optional vector of the events that the operation should depend on 

return event 
    An object to track the progress of the operation 

.. _Allgatherv:

Allgatherv
**********

Allgatherv is a collective communication operation that collects data from all
the ranks within a communicator into a single buffer. Different ranks may
contribute segments of different sizes. The resulting data in the output buffer
must be the same for each rank.

Allgatherv is in place when  ``send_buf == recv_buf + rank_offset``, where
``rank_offset = sum (recv_counts[i])``, for all ``i < rank``.

.. code:: cpp

    template<class BufferType>
    event ccl::allgatherv(const BufferType* send_buf,
                          size_t send_count,
                          BufferType* recv_buf,
                          const vector_class<size_t>& recv_counts,
                          const communicator& comm,
                          const stream& stream,
                          const allgatherv_attr& attr = default_allgatherv_attr,
                          const vector_class<event>& deps = {});

    event ccl::allgatherv(const void* send_buf,
                          size_t send_count,
                          void* recv_buf,
                          const vector_class<size_t>& recv_counts,
                          datatype dtype,
                          const communicator& comm,
                          const stream& stream,
                          const allgatherv_attr& attr = default_allgatherv_attr,
                          const vector_class<event>& deps = {});

send_buf
    the buffer with ``send_count`` elements of ``BufferType`` that stores local data to be gathered
send_count
    the number of elements of type ``BufferType`` in ``send_buf``
recv_buf [out]
    the buffer to store the gathered result, must be large enough to hold values from all ranks
recv_counts
    | an array with the number of elements of type ``BufferType`` to be received from each rank
    | the array's size must be equal to the number of ranks
    | the values in the array are expected to be the same for all ranks
    | the value at the position of the caller's rank must be equal to ``send_count``
dtype
    | the datatype of elements in ``send_buf`` and ``recv_buf``
    | must be skipped if ``BufferType`` can be inferred
    | otherwise must be passed explicitly
comm
    the communicator that defines a group of ranks for the operation
stream
    the stream associated with the operation
attr
    optional attributes to customize the operation
deps
    an optional vector of the events that the operation should depend on
return ``event``
    an object to track the progress of the operation


.. _Allreduce:

Allreduce
*********

Allreduce is a collective communication operation that performs the global
reduction operation on values from all ranks of communicator and distributes
the result back to all ranks.

Allreduce is in-place when ``send_buf == recv_buf``.

.. code:: cpp

    template <class BufferType>
    event ccl::allreduce(const BufferType* send_buf,
                         BufferType* recv_buf,
                         size_t count,
                         reduction rtype,
                         const communicator& comm,
                         const stream& stream,
                         const allreduce_attr& attr = default_allreduce_attr,
                         const vector_class<event>& deps = {});

    event ccl::allreduce(const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         reduction rtype,
                         datatype dtype,
                         const communicator& comm,
                         const stream& stream,
                         const allreduce_attr& attr = default_allreduce_attr,
                         const vector_class<event>& deps = {});

send_buf
    the buffer with ``count`` elements of ``BufferType`` that stores local data to be reduced
recv_buf [out]
    the buffer to store the reduced result, must have the same dimension as ``send_buf``
count
    the number of elements of type ``BufferType`` in ``send_buf`` and ``recv_buf``
rtype
    the type of the reduction operation to be applied
dtype
    | the datatype of elements in ``send_buf`` and ``recv_buf``
    | must be skipped if ``BufferType`` can be inferred
    | otherwise must be passed explicitly
comm
    the communicator that defines a group of ranks for the operation
stream
    the stream associated with the operation
attr
    optional attributes to customize the operation
deps
    an optional vector of the events that the operation should depend on
return ``event``
    an object to track the progress of the operation


.. _Alltoallv:

Alltoallv
*********

Alltoall is a collective communication operation in which each rank
sends separate blocks of data to each rank. Block sizes may differ.
The j-th block of send buffer sent from the i-th rank is received by the j-th rank
and is placed in the i-th block of receive buffer.

.. code:: cpp

    template <class BufferType>
    event ccl::alltoallv(const BufferType* send_buf,
                         const vector_class<size_t>& send_counts,
                         BufferType* recv_buf,
                         const vector_class<size_t>& recv_counts,
                         const communicator& comm,
                         const stream& stream,
                         const alltoallv_attr& attr = default_alltoallv_attr,
                         const vector_class<event>& deps = {});

    event ccl::alltoallv(const void* send_buf,
                         const vector_class<size_t>& send_counts,
                         void* recv_buf,
                         const vector_class<size_t>& recv_counts,
                         datatype dtype,
                         const communicator& comm,
                         const stream& stream,
                         const alltoallv_attr& attr = default_alltoallv_attr,
                         const vector_class<event>& deps = {});

send_buf
    the buffer with elements of ``BufferType`` that stores local blocks to be sent to each rank
send_counts
    | an array with number of elements of type ``BufferType`` in the blocks sent for each rank
    | the array's size must be equal to the number of ranks
    | the values at the position of the caller's rank in ``send_counts`` and ``recv_counts`` must be equal
recv_buf [out]
    the buffer to store the received result, must be large enough to hold blocks from all ranks
recv_counts
    | an array with number of elements of type ``BufferType`` in the blocks received from each rank
    | the array's size must be equal to the number of ranks
    | the values at the position of the caller's rank in ``send_counts`` and ``recv_counts`` must be equal
dtype
    | the datatype of elements in ``send_buf`` and ``recv_buf``
    | must be skipped if ``BufferType`` can be inferred
    | otherwise must be passed explicitly
comm
    the communicator that defines a group of ranks for the operation
stream
    the stream associated with the operation
attr
    optional attributes to customize the operation
deps
    an optional vector of the events that the operation should depend on
return ``event``
    an object to track the progress of the operation


.. _Barrier:

Barrier
*******

Barrier synchronization is performed across all ranks of the communicator
and it is completed only after all the ranks in the communicator have called it.

.. code:: cpp

    event ccl::barrier(const communicator& comm,
                       const stream& stream,
                       const barrier_attr& attr = default_barrier_attr,
                       const vector_class<event>& deps = {});

comm
    the communicator that defines a group of ranks for the operation
stream
    the stream associated with the operation
attr
    optional attributes to customize the operation
deps
    an optional vector of the events that the operation should depend on
return ``event``
    an object to track the progress of the operation


.. _Broadcast:

Broadcast
*********

Broadcast is a collective communication operation that broadcasts data from one rank of communicator (denoted as root) to all other ranks.

Broadcast is in-place if send_buf == recv_buf 

.. code:: cpp

    template <class BufferType> 
    event ccl::broadcast(BufferType*send_buf, 
                         BufferType*recv_buf, 
                         size_t count, 
                         int root, 
                         const communicator& comm, 
                         const stream& stream, 
                         const broadcast_attr& attr = default_broadcast_attr, 
                         const vector_class<event>& deps = {}); 
 
     event ccl::broadcast(void* send_buf, 
                          void* recv_buf 
                          size_t count, 
                          datatype dtype, 
                          int root, 
                          const communicator& comm, 
                          const stream& stream, 
                          const broadcast_attr& attr = default_broadcast_attr, 
                          const vector_class<event>& deps = {}); 
 

send_buf [in,out]
    The buffer with ``count`` elements of ``BufferType`` serves as ``send_buf`` for root and as ``recv_buf`` for other ranks
count
    The number of elements of type ``BufferType`` in ``buf``
root
    The rank that broadcasts ``buf``
dtype
     The datatype of elements in ``buf``
     | must be skipped if ``BufferType`` can be inferred
     | otherwise must be passed explicitly
comm
    The communicator that defines a group of ranks for the operation
stream
    The stream associated with the operation
attr
    Optional attributes to customize the operation
deps
    An optional vector of the events that the operation should depend on
return ``event``
    An object to track the progress of the operation


.. _Reduce:

Reduce
******

Reduce is a collective communication operation that performs the global
reduction operation on values from all ranks of the communicator and returns
the result to the root rank.

Reduce is in-place when ``send_buf == recv_buf``.

.. code:: cpp

    template <class BufferType>
    event ccl::reduce(const BufferType* send_buf,
                      BufferType* recv_buf,
                      size_t count,
                      reduction rtype,
                      int root,
                      const communicator& comm,
                      const stream& stream,
                      const reduce_attr& attr = default_reduce_attr,
                      const vector_class<event>& deps = {});

    event ccl::reduce(const void* send_buf,
                      void* recv_buf,
                      size_t count,
                      datatype dtype,
                      reduction rtype,
                      int root,
                      const communicator& comm,
                      const stream& stream,
                      const reduce_attr& attr = default_reduce_attr,
                      const vector_class<event>& deps = {});

send_buf
    the buffer with ``count`` elements of ``BufferType`` that stores local data to be reduced
recv_buf [out]
    | the buffer to store the reduced result, must have the same dimension as ``send_buf``.
    | Used by the ``root`` rank only, ignored by other ranks.
count
    the number of elements of type ``BufferType`` in ``send_buf`` and ``recv_buf``
rtype
    the type of the reduction operation to be applied
root
    the rank that gets the result of the reduction
dtype
    | the datatype of elements in ``send_buf`` and ``recv_buf``
    | must be skipped if ``BufferType`` can be inferred
    | otherwise must be passed explicitly
comm
    the communicator that defines a group of ranks for the operation
stream
    the stream associated with the operation
attr
    optional attributes to customize the operation
deps
    an optional vector of the events that the operation should depend on
return ``event``
    an object to track the progress of the operation


.. _ReduceScatter:

ReduceScatter
**************

Reduce-scatter is a collective communication operation that performs the global
reduction operation on values from all ranks of the communicator and scatters
the result in blocks back to all ranks.

ReduceScatter is in-place when ``recv_buf == send_buf + rank * recv_count``

.. code:: cpp

    template <class BufferType>
    event ccl::reduce_scatter(const BufferType* send_buf,
                              BufferType* recv_buf,
                              size_t recv_count,
                              reduction rtype,
                              const communicator& comm,
                              const stream& stream,
                              const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                              const vector_class<event>& deps = {});

    event ccl::reduce_scatter(const void* send_buf,
                              void* recv_buf,
                              size_t recv_count,
                              datatype dtype,
                              reduction rtype,
                              const communicator& comm,
                              const stream& stream,
                              const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                              const vector_class<event>& deps = {});

send_buf
    the buffer with ``comm_size`` * ``count`` elements of ``BufferType`` that stores local data to be reduced
recv_buf [out]
    the buffer to store the result block containing ``recv_count`` elements of type ``BufferType``
recv_count
    the number of elements of type ``BufferType`` in the received block
rtype
    the type of the reduction operation to be applied
dtype
    | the datatype of elements in ``send_buf`` and ``recv_buf``
    | must be skipped if ``BufferType`` can be inferred
    | otherwise must be passed explicitly
comm
    the communicator that defines a group of ranks for the operation
stream
    the stream associated with the operation
attr
    optional attributes to customize the operation
deps
    an optional vector of the events that the operation should depend on
return ``event``
    an object to track the progress of the operation

.. note::
    See also:

    - :ref:`Communicator`
    - :ref:`Stream`
    - :ref:`Event`
    - :doc:`operation_progress`

Point-to-point operations (``send`` and ``recv``) are specified separately from the collectives.
See :doc:`Point-To-Point Operations </api/operations/point-to-point>` in the Developer Reference.
