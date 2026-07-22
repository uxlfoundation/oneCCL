.. _point-to-point:

Point-To-Point Operations
**************************

.. note::
   In |product_short| version 2021.17 included with the 2025.3 oneAPI release, oneCCL has added support for a new **C API** that closely follows the NVIDIA Collective Communications Libary (NCCL)* API standard. Details about the new API, instructions on how to build, and run an example can be found `here <./v2/index.html>`_.

   The existing **C++ API** will remain the default API for the 2021.17 release.

Point-to-point operations enable direct communication between two specific entities, facilitating data exchange, synchronization, and coordination within a parallel computing environment. 

The following point-to-point operations are available in oneCCL: 

* ``send``
* ``recv``

send
====

``send`` is a blocking point-to-point communication operation that transfers data from a designated memory buffer (``buf``) to a specific peer rank.

.. code:: cpp

    event CCL_API send(void *buf,   
              size_t count,
              datatype dtype, 
              int peer, 
              const communicator &comm, 
              const stream &stream, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event> &deps = {}); 

**Parameters**

* ``buf`` - A buffer with ``dtype`` count elements that contains the data to be sent.
* ``count`` - The number of ``dtype`` elements in a ``buf``.  
* ``dtype``- The datatype of elements in a ``buf``.  
* ``peer`` - A destination rank.  
* ``comm`` - A communicator for which the operation is performed. 
* ``stream`` - A stream associated with the operation. 
* ``attr`` - Optional attributes to customize the operation. 
* ``deps`` - An optional vector of the events, on which the operation should depend. 

**Returns**
  
``ccl::event`` - An object to track the progress of the operation. 

.. code:: cpp

    event CCL_API send(void* buf, 
              size_t count, 
              datatype dtype, 
              int peer, 
              const communicator &comm, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event> &deps = {}); 

Below you can find an overloaded member function provided for the convenience. It differs from the above function only in what argument(s) it accepts.

.. code:: cpp

    template <class BufferType, 
        class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type> 
    event CCL_API send(BufferType *buf, 
              size_t count, 
              int peer, 
              const communicator &comm, 
              const stream &stream, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event>& deps = {}); 

Below you can find an overloaded member function provided for the convenience. It differs from the above function only in what argument(s) it accepts.: 

.. code:: cpp

    event CCL_API send(BufferType *buf, 
              size_t count, 
              int peer, 
              const communicator &comm, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event> &deps = {}); 

Below you can find an overloaded member function provided for the convenience. It differs from the above function only in what argument(s) it accepts.

.. code:: cpp

    event CCL_API send(BufferObjectType &buf, 
              size_t count, 
              int peer, 
              const communicator &comm, 
              const stream &stream, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event> &deps = {}); 

Below you can find an overloaded member function provided for the convenience. It differs from the above function only in what argument(s) it accepts.

.. code:: cpp

    event CCL_API send(BufferObjectType &buf, 
              size_t count, 
              int peer, 
              const communicator &comm, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event> &deps = {}); 



recv
=====

``recv`` is a blocking point-to-point communication operation that receives data from a peer rank in a memory buffer.  

.. code:: cpp

   event CCL_API recv(void *buf,     
             size_t count,              
             datatype dtype, 
             int peer, 
             const communicator &comm, 
             const stream &stream, 
             const pt2pt_attr &attr = default_pt2pt_attr, 
             const vector_class<event> &deps = {});  

**Parameters**

* ``buf`` - A buffer with ``dtype`` count elements that contains where the data is received.
* ``count`` - The number of ``dtype`` elements in a ``buf``.  
* ``dtype``- The datatype of elements in a ``buf``.  
* ``peer`` - A source rank.  
* ``comm`` - A communicator for which the operation is performed. 
* ``dtream`` - A stream associated with the operation. 
* ``attr`` - Optional attributes to customize the operation. 
* ``deps`` - An optional vector of the events, on which the operation should depend. 


**Returns:**

``ccl::event`` - An object to track the progress of the operation. 

.. code:: cpp

    event CCL_API recv(void *buf, 
              size_t count, 
              datatype dtype, 
              int peer, 
              const communicator &comm, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event>& deps = {}); 


Below you can find an overloaded member function provided for the convenience. It differs from the above function only in what argument(s) it accepts. 

.. code:: cpp

    template <class BufferType, 
        class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type> 
    event CCL_API recv(BufferType *buf, 
              size_t count, 
              int peer, 
              const communicator &comm, 
              const stream &stream, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event> &deps = {}); 

Below you can find an overloaded member function provided for the convenience. It differs from the above function only in what argument(s) it accepts. 

.. code:: cpp

    event CCL_API recv(BufferType *buf, 
              size_t count, 
              int peer, 
              const communicator &comm, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event> &deps = {}); 

Below you can find an overloaded member function provided for the convenience. It differs from the above function only in what argument(s) it accepts.

.. code:: cpp

    event CCL_API recv(BufferObjectType &buf, 
              size_t count, 
              int peer, 
              const communicator &comm, 
              const stream &stream, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event> &deps = {}); 


Below you can find an overloaded member function provided for the convenience. It differs from the above function only in what argument(s) it accepts. 

.. code:: cpp
    
    event CCL_API recv(BufferObjectType &buf, 
              size_t count, 
              int peer, 
              const communicator &comm, 
              const pt2pt_attr &attr = default_pt2pt_attr, 
              const vector_class<event> &deps = {}); 
