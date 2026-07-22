.. _`Communicator`: https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/spec/main_objects#communicator

==================
Host Communication
==================

The oneCCL Communicator defines a group of processes that communicate with each other.

The following example demonstrates communication between two processes on host memory buffers.

Consider a simple oneCCL ``allreduce`` example for CPU.

.. rubric:: Example

#. Create a communicator object and provide the size, rank, and key-value store values:

   .. code:: cpp

      auto comm = ccl::create_communicator(size, rank, kvs);

#. Initialize ``send_buf`` by providing the input content. For example, you can create and initialize the ``send_buf`` parameter as follows:

   .. code:: cpp

      const size_t elem_count = <N>;

      /* initialize send_buf */
      for (idx = 0; idx < elem_count; idx++) {
         send_buf[idx] = rank + 1;
      }

   The ``allreduce`` invocation performs the reduction of values from all the processes and then distributes the result to all the processes. In this case, the result is an array with the ``elem_count`` elements, where all elements are equal to the sum of the arithmetical progression:

   .. math::
      p \cdot (p + 1) / 2

#. Execute the ``allreduce`` operation:

   .. code:: cpp

      ccl::allreduce(send_buf,
                     recv_buf,
                     elem_count,
                     reduction::sum,
                     comm).wait();
                  
   .. note:: The oneCCL operations are asynchronous and the memory buffers must remain intact until all operations are completed. You can free up memory associated with the buffers when an operation completes.

#. Verify that the ``allreduce`` operation is correct:

   .. code:: cpp

      auto comm_size = comm.size();
      auto expected = comm_size * (comm_size + 1) / 2;

      for (idx = 0; idx < elem_count; idx++) {
         if (recv_buf[idx] != expected) {
               std::count << "unexpected value at index " << idx << std::endl;
               break;
         }
      }


If you encounter an error, make sure the oneCCL environment is configured correctly.

Additional Resources
====================

- `OneCCL Communicator <https://uxlfoundation.github.io/oneAPI-spec/spec/elements/oneCCL/source/spec/main_objects.html#communicator>`_
- `OneCCL ALLREDUCE communication pattern <https://uxlfoundation.github.io/oneAPI-spec/spec/elements/oneCCL/source/spec/collective_operations.html#allreduce>`_
