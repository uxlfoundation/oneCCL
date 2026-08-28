.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

================
Generic Workflow
================

Below is a generic workflow with oneCCL API

1. Create a main built-in key-value store. Its address should be distributed
using an out-of-band communication mechanism and be used to create key-value stores on other processes:

.. code:: cpp

    using namespace std;

    /* for example use MPI as an out-of-band communication mechanism */

    int mpi_rank, mpi_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type kvs_addr;

    if (mpi_rank == 0) {
        kvs = ccl::create_main_kvs();
        kvs_addr = kvs->get_address();
        MPI_Bcast((void*)kvs_addr.data(), ccl::kvs::address_max_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)kvs_addr.data(), ccl::kvs::address_max_size, MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(kvs_addr);
    }

2. Create communicator(s):

.. code:: cpp

    /* host communications */
    auto comm = ccl::create_communicator(mpi_size, mpi_rank, kvs);

.. code:: cpp

    /* SYCL devices communications, for example with multiple devices per process */

    /* sycl_context -> sycl::context */
    /* sycl_devices -> vector<sycl::device> */
    /* sycl_queues  -> vector<sycl::queue> */

    /* create ccl::context object from sycl::context object */
    auto ccl_context = ccl::create_context(sycl_context);

    /* create ccl::device objects from sycl::device objects */
    vector<ccl::device> ccl_devices;
    for (size_t idx = 0; idx < sycl_devices.size(); idx++) {
        ccl_devices.push_back(ccl::create_device(sycl_devices[idx]));
    }

    map<int, ccl::device> r2d_map;
    for (auto& dev : ccl_devices) {
        int rank = /* generate a globally unique rank for a specific device */
        r2d_map[rank] = dev;
    }

    /* create ccl::stream objects from sycl::queue objects */
    vector<ccl::stream> ccl_streams;
    for (size_t idx = 0; idx < sycl_queues.size(); idx++) {
        ccl_streams.push_back(ccl::create_stream(sycl_queues[idx]));
    }

    auto comms = ccl::create_communicators(mpi_size * r2d_map.size(),
                                           r2d_map,
                                           ccl_context,
                                           kvs);

3. Execute a communication operation of choice on the communicator(s):

.. code:: cpp

    /* host communications */
    allreduce(..., comm).wait();

.. code:: cpp

    /* SYCL devices communications */
    vector<ccl::event> events;
    for (auto& comm : comms) {
        events.push_back(allreduce(..., comm, ccl_streams[comm.rank()]));
    }

    for (auto& e : events) {
        e.wait();
    }
