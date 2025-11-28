oneCCL Benchmark Collective User Guide
======================================

The oneCCL benchmark provides performance measurements for the collective operations in oneCCL, such as:

* ``allreduce``
* ``reduce``
* ``allgather``
* ``allgatherv``
* ``alltoall``
* ``alltoallv``
* ``reduce_scatter``
* ``broadcast``

The benchmark is distributed with the oneCCL package. You can find it in the examples directory within the oneCCL installation path.


Build oneCCL Benchmark
***********************

To build the benchmark, complete the following steps:

.. tab-set::

    .. tab-item:: CPU-Only

      

      #. Configure your environment. Source the installed oneCCL library for the CPU-only support:

         .. code::

            source <oneCCL install dir>/ccl/latest/env/vars.sh --ccl-configuration=cpu

      #. Navigate to ``<oneCCL install dir>/share/doc/ccl/examples``
      #. Build the benchmark with the following command:

         .. code::

            cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$(pwd)/build/_install && cmake --build build -j $(nproc) -t install

    .. tab-item:: CPU-GPU

       #. Configure your environment.

          * Source the SYCL compiler. See the `documentation <https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/get-started-guide/2024-2/overview.html>`_ for the instructions.
          * Source the installed oneCCL library for the CPU-GPU support:

          .. code::

             source <oneCCL install dir>/ccl/latest/env/vars.sh --ccl-configuration=cpu_gpu_dpcpp

       #. Navigate to ``<oneCCL install dir>/share/doc/ccl/examples``.
       #. Build the SYCL benchmark with the following command:

          .. code::

             cmake -S . -B build -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCOMPUTE_BACKEND=dpcpp -DCMAKE_INSTALL_PREFIX=$(pwd)/build/_install && cmake --build build -j $(nproc) -t install


Run oneCCL Benchmark
*********************

To run the benchmark, use the following command:

.. code::

   mpirun -n <N> -ppn <P> benchmark [arguments]

Where:

* ``N`` is the overall number of processes
* ``P`` is the number of processes within a node

The benchmark reports:

* ``#bytes`` - the message size in the number of bytes
* ``elem_count`` - the message size in the number of elements
* ``#repetitions`` - the number of iterations
* ``t_min`` - the average time across iterations of the fastest process in each iteration
* ``t_max`` - the average time across iterations of the slowest rank in each iteration
* ``t_avg`` - the average time across processes and iterations
* ``stddev`` - standard deviation
* ``wait_t_avg`` - the average wait time after the collective call returns and until it completes To enable, use the ``-x`` option.

Notice that ``t_min``, ``t_max``, and ``t_avg`` measure the total collective time. It means the timer starts before calling the collective and ends once the collective completes.
While ``wait_t_avg`` only measures the wait time. It means the timer starts after the collective call returns and ends once the collective completes.
Thus, ``wait_t_avg`` does not measure the time the oneCCL runtime spends preparing the necessary tasks for the collective to execute, while ``t_min``, ``t_max``, and ``t_avg`` include that time. Time is reported in `μsec`.


Benchmark Arguments
^^^^^^^^^^^^^^^^^^^

To see the benchmark arguments, use the ``--help`` argument.

The benchmark accepts the following arguments:

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Option
     - Description
     - Default Value
   * - ``-b``, ``--backend``
     - Specify the backend. The possible values are host and sycl. For a CPU-only build, the backend is automatically set to host, and only the host option is available. 
       For a CPU-GPU build, host and sycl options are available, and sycl is the default value. The host value allocates buffers in the host (CPU) memory, while the sycl value allocates buffers in the host (CPU) or device (GPU) memory.
     -  ``sycl`` for CPU-GPU build and ``host`` for CPU build
   * - ``-i``, ``--iters``
     - Specify the number of iterations executed by the benchmark.
     - ``16``
   * - ``-w``, ``--warmup_iters``
     - Specify the number of the warmup iterations. It means the number of iterations the benchmark runs before starting the timing of the iterations specified with the ``-i`` argument.
     - ``16``
   * - ``-j``, ``--iter_policy``
     - Specify the iteration policy. Possible values are off and auto. When the iteration policy is off, the number of iterations is the same across the message sizes. 
       When the iteration policy is auto, the number of iterations reduces based on the message size of the collective operation.
     - ``auto``
   * - ``-n``, ``--buf_count``
     - Specify the number of collective operations the benchmark calls in each iteration. Each collective uses different ``send`` and ``receive`` buffers.
       The explicit wait calls are placed for each collective after all of them are called.
     - ``1``
   * - ``-f``, ``--min_elem_count``
     - Specify the minimum number of elements used for the collective.
     - ``0``
   * - ``-t``, ``--max_elem_count``
     - Specify the maximum number of elements used for the collective.
     - ``128``
   * - ``-y``, ``-y, --elem_counts``
     - 	Specify a list with the number of elements used for the collective, such as ``[-y 4,8,32,131072]``.
     - ``[-y 0,1,2,4,8,16,32,64,128]``
   * - ``-c``, ``--check``
     - Check for correctness. The possible values are ``off`` (disable checking), ``last`` (check the last iteration), and ``all`` (check all the iterations).
     - ``last``
   * - ``-p``, ``--cache``
     - Specify whether to use persistent collectives (``p=1``) or not (``p=0``).

       .. note:: A collective is persistent when the same collective is called with the same parameters multiple times. OneCCL generates a schedule for each collective it runs and can apply optimizations when persistent collectives are used.
                 It means the schedule is generated once and reused across the subsequent invocations, saving the time to generate the schedule.

     - ``1``
   * - ``-q``, ``--inplace``
     - Specify for oneCCL to use in-place (``1``) or out-of-place (``0``) buffers. With the in-place buffers, the send and receive buffers used by the collective are the same.
       With the out-of-place, the buffers are different.
     - ``0``
   * - ``-a``, ``--sycl_dev_type``
     - Specify the type of the SYCL device. The possible values are ``host``, ``cpu``, and ``gpu``.
     - ``gpu``
   * - ``-g``, ``--sycl_root_dev``
     - Specify to use the root devices (``0``) and sub-devices (``1``).
     - ``0``
   * - ``-m``, ``--sycl_mem_type``
     - Specify the type of SYCL memory. The possible values are ``usm`` (unified shared memory) and ``buf`` (buffers).
     - ``usm``
   * - ``-u``, ``--sycl_usm_type``
     - Specify the type of SYCL device. The possible values are ``device`` or ``shared``.
     - ``device``
   * - ``-e``, ``--sycl_queue_type``
     - Specify the type of the SYCL queue. The possible values are ``in_order`` and ``out_order``.
     - ``out_order``
   * - ``-l``, ``--coll``
     - Specify the collective to run. Accept a comma-separated list, without whitespace characters, of collectives to run. The available collectives are ``allreduce``, ``reduce``, ``alltoallv``, ``alltoall``, ``allgatherv``, ``reduce_scatter``, ``broadcast``.
     - ``allreduce``
   * - ``-d``, ``--dtype``
     - Specify the datatype. Accept a comma-separated list, without whitespace characters, of datatypes to benchmark. The available types are ``int8``, ``int32``, ``int64``, ``uint64``, ``float16``, ``float32``, and ``bfloat16``.
     - ``float32``
   * - ``-r``, ``--reduction``
     - Specify the type of the reduction. Accept a coma-separated list, without whitespace characters, of the reduction operations to run. The available operations are ``sum``, ``prod``, ``min``, and ``max``.
     - ``sum``
   * - ``-o``, ``--csv_filepath``
     - Specify to store the output in the specified CSV file. User specifies the csv_filepath/file_to_store CSV-formatted data. 
     -
   * - ``-x``, ``--ext``
     - Specify to show the additional information. The possible values are ``off``, ``auto``, and ``on``. With ``on``, it also displays the average wait time.
     - ``auto``
   * - ``-h``, ``--help``
     - Show all of the supported options.
     -

.. note::

   The ``-t`` and ``-f`` options specify the count in number of elements, so the total number of bytes is obtained by multiplying the number of elements by the number of bytes of the data type the collective uses.
   For instance, with ``-f 128`` and ``fp32`` datatype, the total amount of bytes is 512 (128 element count * 4 bytes FP32).
   The benchmark runs and reports time for message sizes that correspond to the ``-t`` and ``-f`` arguments and all message sizes that are powers of two in between these two numbers.


Example
********

.. tab-set::

  .. tab-item:: GPU

    The following example shows how to run the benchmark with the GPU buffers:

    .. code::

       mpirun -n <N> -ppn <P> benchmark -a gpu -m usm -u device -l allreduce -i 20 -f 1024 -t 67108864  -j off -d float32 -p 0 -e in_order

    The above command runs:

    * The ``allreduce`` benchmark
    * With a total of ``N`` processes
    * With ``P`` processes per node allocating the memory in the GPU
    * 20 iterations
    * Use SYCL Unified Shared Memory (USM) of the device type
    * With the element count from 1024 to 67108864 (the benchmark runs with all the powers on two in that range) of float32 datatype, assuming the collective is not persistent and using a SYCL in-order queue.


  .. tab-item:: CPU
    
    The following example shows how to run the benchmark with the CPU buffers:

    .. code::

       mpirun -n <N> -ppn <P> benchmark -l allreduce -i 20 -f 1024 -t 67108864  -j off -d float32 -p 0

    The above command specifies to run:

    * The ``allreduce`` benchmark
    * With a total of ``N`` processes
    * With ``P`` processes per node
    * 20 iterations
    * With the element count from 1024 to 67108864 (the benchmark runs with all the powers on two in that range) of float32 datatype, assuming the collective is not persistent


      
