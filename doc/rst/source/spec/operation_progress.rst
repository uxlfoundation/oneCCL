.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

===========================
Operation Progress Tracking
===========================

oneCCL communication operation shall return an event object to be used for tracking the operation's progress.

The ``event`` class shall provide the ability to wait for completion of an operation in a blocking manner, the ability to check the completion status in a non-blocking manner, and the ability to retrieve the underlying native object that is signaled when the operation completes.

Event
*****

Waiting for the completion of an operation in a blocking manner:

.. code:: cpp

    void event::wait();


Checking for the completion of an operation in a non-blocking manner:

.. code:: cpp

    bool event::test();

return ``bool``
    ``true`` if the operation has been completed
    ``false`` if the operation has not been completed


Retrieving a native object that is signaled when the operation completes:

.. code:: cpp

    native_event_type event::get_native();

return ``native_event_type``
    | a native object that is signaled when the operation completes
    | shall throw an exception if an ``event`` object does not wrap the native object
