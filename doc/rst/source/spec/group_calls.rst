.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

============
Group Calls
============

oneCCL specification defines the following group calls: 

* group_start() 
* group_end() 


Group calls serve to merge multiple calls into a single group operation. These operations are initiated and finalized using primary functions: group_start() and group_end(). 

Group_start 
***********

.. code:: cpp

    void CCL_API group_start();

group_start() starts a group call. group_start() can be used to initiate a group call operation to indicate that successive operations should not get blocked due to CPU synchronization.  

Group_end 
*********

.. code:: cpp

    void CCL_API group_end();

group_end() ends a group call. The group_end() call returns when all the operations between group_start() and group_end() have been enqueued for execution, but not necessarily completed. 


.. note::
    Currently group calls are supported for point-to-point and collective operations.  

