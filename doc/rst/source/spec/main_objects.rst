.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

============
Main Objects
============

oneCCL specification defines the following list of concepts:

- `Device`_
- `Context`_
- `Key-Value Store`_
- `Communicator`_
- `Stream`_
- `Event`_

Communication operation behavior may additionally be controlled through
:doc:`operation attributes <operation_attributes>`.


.. _Device:

Device
******

.. note::
    Here and below, a native device/context/stream/event are defined in the scope of SYCL device runtime

.. code:: cpp

    using native_device_type = sycl::device;
    using native_context_type = sycl::context;
    using native_stream_type = sycl::queue;
    using native_event_type = sycl::event;

oneCCL specification defines ``device`` as an abstraction of a computational device: a CPU, a specific GPU card in the system, or any other device participating in a communication operation. ``device`` corresponds to the communicator's rank (addressable entity in a communication operation).

oneCCL specification defines the way to create an instance of the ``device`` class with a native object (``native_device_type``) and without a native object (corresponds to the host).


Creating a new device object:

.. code:: cpp

    device ccl::create_device(native_device_type& native_device);

    device ccl::create_device();

native_device
    the existing native device object
return ``device``
    a device object

``device`` class shall provide ability to retrieve a native object.

Retrieving a native device object:

.. code:: cpp

    native_device_type device::get_native();

return ``native_device_type``
    | a native device object
    | shall throw exception if a ``device`` object does not wrap the native object


.. _Context:

Context
*******

oneCCL specification defines ``context`` as an abstraction of a computational devices context that is responsible for managing resources and for executing of communication operations on one or more devices specified in the context.

oneCCL specification defines the way to create an instance of the ``context`` class with a native object (``native_context_type``) and without a native object.


Creating a new context object:

.. code:: cpp

    context ccl::create_context(native_context_type& native_context);

    context ccl::create_context();

native_context
    the existing native context object
return ``context``
    a context object


``context`` class shall provide ability to retrieve a native object.

Retrieving a native context object:

.. code:: cpp

    native_context_type context::get_native();

return ``native_context_type``
    | a native context object
    | shall throw exception if a ``context`` object does not wrap the native object


Key-Value Store
***************

``kvs_interface`` defines the key-value store (KVS) interface to be used to establish connection between ranks during the creation of oneCCL communicator. The interface shall include blocking ``get`` and ``set`` methods.


Getting a record from the key-value store:

.. code:: cpp

    virtual vector_class<char> kvs_interface::get(
        const string_class& key) = 0;

key
    the key of value to be retrieved
return ``vector_class<char>``
    the value associated with the given key

.. note::
    ``get`` operation with a non-existing key shall return empty result


Saving a record in the key-value store:

.. code:: cpp

    void kvs_interface::set(
        const string_class& key,
        const vector_class<char>& data) = 0;

key
    the key at which the value should be stored
data
    the value that should be associated with the given key

.. note::
    ``set`` operation with empty ``data`` shall remove a record from the key-value store


oneCCL specification defines ``kvs`` class as a built-in KVS provided by oneCCL.

.. code:: cpp

    class kvs : public kvs_interface {
    
    public:

    static constexpr size_t address_max_size = 256;
    using address_type = array_class<char, address_max_size>;

    ~kvs() override;

    address_type get_address() const;

    vector_class<char> get(
        const string_class& key) override;

    void set(
        const string_class& key,
        const vector_class<char>& data) override;

    }


Retrieving an address of built-in key-value store:

.. code:: cpp

    kvs::address_type kvs::get_address() const;

return ``kvs::address_type``
    | the address of the key-value store
    | should be retrieved from the main built-in KVS and distributed to other processes for the built-in KVS creation


Creating a main built-in key-value store. Its address should be distributed using an out-of-band communication mechanism
and be used to create key-value stores on other ranks:

.. code:: cpp

    shared_ptr_class<kvs> ccl::create_main_kvs();

return ``shared_ptr_class<kvs>``
    the main key-value store object


Creating a new key-value store from main kvs address:

.. code:: cpp

    shared_ptr_class<kvs> ccl::create_kvs(const kvs::address_type& addr);

addr
    the address of the main kvs
return ``shared_ptr_class<kvs>``
    key-value store object


.. _Communicator:

Communicator
************

oneCCL specification defines ``communicator`` class that describes a group of communicating ranks, where a rank is an addressable entity in a communication operation and corresponds to single oneCCL device.

``communicator`` defines communication operations on memory buffers between homogeneous oneCCL devices, that is, all oneCCL devices either wrap native device objects of the same type (for example CPUs only or GPUs only) or do not wrap native objects.

Each process may correspond to multiple ranks.

.. note::
    Support for multiple ranks per process is optional

Creating a new communicator(s) with user-supplied communicator size, rank-to-device mapping/rank, context and kvs:

.. note::
  If ``device`` and ``context`` objects are omitted, then they are created with ``ccl::create_device()`` and ``ccl::create_context()`` functions without native objects

.. code:: cpp

    vector_class<communicator> ccl::create_communicators(
        int size,
        const map_class<int, device>& rank_device_map,
        const context& context,
        shared_ptr_class<kvs_interface> kvs);

    communicator ccl::create_communicator(
        int size,
        int rank,
        shared_ptr_class<kvs_interface> kvs);

size
    user-supplied total number of ranks
rank_device_map
    user-supplied mapping of local ranks on devices
rank
    user-supplied local rank
context
    device context
kvs
    key-value store for ranks wire-up
return ``vector_class<communicator>`` / ``communicator``
    a vector of communicator objects  / a communicator object


``communicator`` shall provide methods to retrieve the rank, the device, and the context that correspond to the communicator object as well as the total number of ranks in the communicator.


Retrieving the rank in a communicator:

.. code:: cpp

    int communicator::rank() const;

return ``int``
    the rank that corresponds to the communicator object


Retrieving the total number of ranks in a communicator:

.. code:: cpp

    int communicator::size() const;

return ``int``
    the total number of the ranks


Retrieving an underlying device, which was used as communicator construction argument:

.. code:: cpp

    device communicator::get_device() const;

return ``device``
    the device that corresponds to the communicator object


Retrieving an underlying context, which was used as communicator construction argument:

.. code:: cpp

    context communicator::get_context() const;

return ``context``
    the context that corresponds to the communicator object

.. note::
    See also: :doc:`collective_operations`



.. _Split_communicator:

Split-Communicator
******************

The communicator provides methods to create one or more new communicators from an existing communicator. The new sub-communicators are created based on user-supplied color and key. Each newly formed sub-communicator includes only the ranks that provided the same color. Within each sub-communicator, ranks are reordered by the ascending key. This call is collective over all ranks in the communicator. 

.. code:: cpp

    communicator split_communicator(const communicator& comm, int color, int key); 

``comm``
   An existing communicator handle.

``color``
    A value that defines which ranks will belong to the new sub-communicator. Ranks providing the same color are part of the same sub-communicator.  

``key``
    Used to order the ranks within each sub-communicator. Ranks in the resulting communicator are sorted by the ascending key. 

 

Return Value 

``communicator``
    The handle to the new sub-communicator  

 



.. _Stream:

Stream
******

oneCCL specification defines ``stream`` as an abstraction that encapsulates execution context for ``communicator`` communication operations.

Stream shall be passed to ``communicator`` communication operation.

oneCCL specification defines the way to create an instance of the ``stream`` class with a native object (``native_stream_type``) and without a native object.


Creating a new stream object:

.. code:: cpp

    stream ccl::create_stream(native_stream_type& native_stream);

    stream ccl::create_stream();

native_stream
    the existing native stream object
return ``stream``
    a stream object


``stream`` class shall provide ability to retrieve a native object.

Retrieving a native stream object:

.. code:: cpp

    native_stream_type stream::get_native();

return ``native_stream_type``
    | a native stream object
    | shall throw exception if a ``stream`` object does not wrap the native object


.. _Event:

Event
*****

oneCCL specification defines ``event`` as an abstraction that encapsulates synchronization context for ``communicator`` communication operations.

Each communication operation of oneCCL shall return an event object for tracking the operation's progress. A vector of events may be passed to the ``communicator`` communication operation to designate input dependencies for the operation.

.. note::
    Support for handling of input events is optional

oneCCL specification defines the way to create an instance of the ``event`` class with a native object (``native_event_type``).


Creating a new event object:

.. code:: cpp

    event ccl::create_event(native_event_type& native_event);

native_event
    the existing native event object
return ``event``
    an event object


``event`` class shall provide ability to retrieve a native object.

Retrieving a native event object:

.. code:: cpp

    native_event_type event::get_native();

return ``native_event_type``
    | a native event object
    | shall throw exception if an ``event`` object does not wrap the native object


.. note::
    See also: :doc:`operation_progress`
