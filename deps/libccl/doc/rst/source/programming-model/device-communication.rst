.. _`Communicator`: https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/spec/main_objects#communicator

====================
Device Communication
====================

The oneCCL Communicator defines a group of processes that communicate with each other.

The following example demonstrates the main concepts of communication on device memory buffers.

Consider a simple oneCCL ``allreduce`` example for GPU.

.. rubric:: Example

#. Create oneCCL communicator objects with user-supplied size, rank <-> SYCL device mapping, SYCL context and key-value store:

   .. code:: cpp

      auto ccl_context = ccl::create_context(sycl_context);
      auto ccl_device = ccl::create_device(sycl_device);

      auto comms = ccl::create_communicators(
         size,
         vector_class<pair_class<size_t, device>>{ { rank, ccl_device } },
         ccl_context,
         kvs);

#. Create oneCCL stream object from user-supplied ``sycl::queue`` object:

   .. code:: cpp

      auto stream = ccl::create_stream(sycl_queue);

#. Initialize ``send_buf`` by providing the input content. For example, you can create and initialize the ``send_buf`` parameter as follows:

   .. code:: cpp

      const size_t elem_count = <N>;

      /* using SYCL buffer and accessor */
      auto send_buf_host_acc = send_buf.get_host_access(h, sycl::write_only);
      for (idx = 0; idx < elem_count; idx++) {
         send_buf_host_acc[idx] = rank;
      }
   .. code:: cpp

      /* or using SYCL USM */
      for (idx = 0; idx < elem_count; idx++) {
         send_buf[idx] = rank;
      }

#. For demonstration purposes, modify the ``send_buf`` on the GPU side:

   .. code:: cpp

      /* using SYCL buffer and accessor */
      sycl_queue.submit([&](sycl::handler& h) {
         auto send_buf_dev_acc = send_buf.get_access<mode::write>(h);
         h.parallel_for(range<1>{elem_count}, [=](item<1> idx) {
               send_buf_dev_acc[idx] += 1;
         });
      });

   .. code:: cpp

      /* or using SYCL USM */
      for (idx = 0; idx < elem_count; idx++) {
         send_buf[idx]+ = 1;
      }

   The ``allreduce`` invocation performs reduction of values from all processes and then distributes the result to all processes. In this case, the result is an array with ``elem_count`` elements, where all elements are equal to the sum of arithmetical progression:

   .. math::
      p \cdot (p + 1) / 2


#. Execute the ``allreduce`` operation:

   .. code:: cpp

      std::vector<event> events;
      for (auto& comm : comms) {
         events.push_back(ccl::allreduce(send_buf,
                                          recv_buf,
                                          elem_count,
                                          reduction::sum,
                                          comm,
                                          streams[comm.rank()]));
      }

      for (auto& e : events) {
         e.wait();
      }

#. Verify that the ``allreduce`` operation is correct:

   .. code:: cpp

      /* using SYCL buffer and accessor */

      auto comm_size = comm.size();
      auto expected = comm_size * (comm_size + 1) / 2;

      sycl_queue.submit([&](handler& h) {
         auto recv_buf_dev_acc = recv_buf.get_access<mode::write>(h);
         h.parallel_for(range<1>{elem_count}, [=](item<1> idx) {
               if (recv_buf_dev_acc[idx] != expected) {
                  recv_buf_dev_acc[idx] = -1;
               }
         });
      });

      ...

      auto recv_buf_host_acc = recv_buf.get_host_access(sycl::read_only);
      for (idx = 0; idx < elem_count; idx++) {
         if (recv_buf_host_acc[idx] == -1) {
               std::count << "unexpected value at index " << idx << std::endl;
               break;
         }
      }

   .. code:: cpp

      /* or using SYCL USM */

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
