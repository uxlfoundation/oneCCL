Programming Model
=================

The programming model for oneCCL describes how to:

* Set up distributed computations.

* Perform collective communication operations (for example, ALLREDUCE, BROADCAST, ALLGATHER).

.. seealso:: See the :ref:`oneCCL specification <oneCCL-section>` that oneCCL is based on.


oneCCL supports a single rank/process per GPU device. The current implementation does not yet support a single process opening multiple devices.


Review the oneCCL :doc:`generic workflow </spec/generic_workflow>` in the specification before getting started with the communication operations.

You can quickly get started with:

.. toctree::
   :maxdepth: 1

   programming-model/host-communication.rst
   programming-model/device-communication.rst
   


