oneCCL Benchmark Point-to-Point User Guide
==========================================

oneCCL provides two benchmarks for performance measurement of the point-to-point operations in oneCCL: 

* ``ccl_latency measures latency``  
* ``ccl_bw measures bandwidth`` 


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
      #. Build the benchmark using the following command:

         .. code::

            cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$(pwd)/build/_install && cmake --build build -j $(nproc) -t install

    .. tab-item:: CPU-GPU
       
      #. Configure your environment.

         * Source the Intel(R) oneAPI DPC++/C++ Compiler. See the `documentation <https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/get-started-guide/2024-2/overview.html>`_ for the instructions.
         * Source the installed oneCCL library for the CPU-GPU support:

           .. code::
          
              source <oneCCL install dir>/ccl/latest/env/vars.sh --ccl-configuration=cpu_gpu_dpcpp

      #. Navigate to ``<oneCCL install dir>/share/doc/ccl/examples``.
      #. Build the SYCL benchmark with the following command:

         .. code::

           cmake -S . -B build -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCOMPUTE_BACKEND=dpcpp -DCMAKE_INSTALL_PREFIX=$(pwd)/build/_install && cmake --build build -j $(nproc) -t install


Run oneCCL Point-to-Point Benchmark
***********************************

To run ``ccl_latency`` or ``ccl_bw``, use the following commands:

.. code::

   mpirun -n 2 -ppn <P> ccl_latency [arguments]   

   mpirun -n 2 -ppn <P> ccl_bw [arguments] 

Where:

* ``2`` is the number of processes (this benchmark runs only with two processes).  

* ``N`` is the number of processes within a node. For this benchmark, ``<P>`` can only be ``1`` or ``2``. When ``N==1``, it indicates there is a single process on each node and the benchmark runs across nodes. When ``N==2``, it indicates both processes are on the same node.

The benchmark reports:

* ``#bytes`` - the message size in the number of bytes
* ``elem_count`` - the message size in the number of elements
* ``#repetitions`` - the number of iterations
* Latency (for ccl_latency benchmark) -  time to send the data from a sender process to a receiving process, where both the send and receive operations are blocking. The time is reported in `μsec`. 
* Bandwidth (for ``ccl_bw``) - represented in Mbytes/second for the bandwidth benchmark 

Both benchmarks always transfer elements of type `int32`. 


Point-to-Point Benchmark Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ccl_latency`` and ``ccl_bw`` accept the same arguments. To see the benchmark arguments, use the ``--help`` argument. 

``ccl_latency`` and ``ccl_bw accept`` accept the following arguments:

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Option
     - Description
     - Default Value
   * - ``-b``, ``--backend``
     - Specify the backend. The possible values are cpu and gpu. For a CPU-only build, the backend is automatically set to cpu, and only the cpu option is available. 
       For a CPU-GPU build, cpu and gpu options are available, and gpu is the default value. The cpu value allocates buffers in the host (CPU) memory, while the gpu value allocates buffers in the device (GPU) memory.
     -  The default is gpu for CPU-GPU build, cpu for CPU-only build 
   * - ``-i``, ``--iters``
     - Specify the number of iterations executed by the benchmark.
     - ``16``
   * - ``-w``, ``--warmup_iters``
     - Specify the number of the warmup iterations. It means the number of iterations the benchmark runs before starting the timing of the iterations specified with the ``-i`` argument.
     - ``16``
   * - ``-p``, ``--cache``
     - Specify whether to use persistent collectives (``p=1``) or not (``p=0``).
     - ``0``
       .. note::  The benchmark currently does not support persistent collectives.  
   * - ``-e``, ``--sycl_queue_type``
     - Specify the type of SYCL queue. Possible values are 0 (out_order) or 1 (in_order).  
     - ``0 (out_order)``
   * - ``-s``, ``--wait``
     - Specifies the synchronization model, that is, whether the point to point operation is 1 (blocking) or  0 (non_blocking). Notice that currently the benchmark only supports blocking point to point operations.  
     - Default mode is 1 (blocking)
   * - ``-f``, ``--min_elem_count``
     - Specifies the minimum number of elements used for the operation.  
     - ``1``
   * - ``-t``, ``--max_elem_count``
     - Specify the maximum number of elements used for the operation.
     - ``33554432``
       .. note::  The ``-t`` and ``-f`` options specify the count in the number of elements. Therefore, the total number of bytes is obtained by multiplying the number of elements by the number of bytes of the data type. For instance, when using ``-f 128`` and data type ``fp32``, the total amount of bytes is 512 bytes (``128 element count * 4 bytes FP32``). ``ccl_latency``/``ccl_bw run`` and report performance for message sizes that correspond to the ``-t`` and ``-f`` arguments and all message sizes that are power of two in between these two numbers. 
   * - ``-y``, ``--elem_counts``
     - Specify a list with the number of elements used for the collective, such as ``[-y 4,8,32,131072]``.
     - The default value is  between ``1`` and ``33554432`` and all powers of two in between.  
   * - ``-c``, ``--check``
     - Check for correctness. The possible values are ``off`` (disable checking), ``last`` (check the last iteration), and ``all`` (check all the iterations).
     - ``last``
  
   * - ``-h``, ``--help``
     - Show all of the supported options.
     -


Examples
********

.. tab-set::

   .. tab-item:: GPU

      The following example shows how to run ``ccl_latency`` with the GPU buffers:

      .. code::

         mpirun -n 2 -ppn <P> ccl_latency -b gpu -i 20 -f 1024 -t 67108864 -e 1 
         mpirun -n 2 -ppn <P> ccl_bw -b gpu -i 20 -f 1024 -t 67108864 -e 1 


      The commands above: 

      * Run the ``ccl_latency`` or the ``ccl_bw`` benchmark  
      * Contain a total of two processes (this benchmark only supports two processes) 
      * Use P processes per node, where P can be ``1`` if running on two different nodes or ``2`` when running on a single node
      * Use GPU buffers 
      * Use 20 iterations 
      * Use element count from ``1024`` to ``67108864`` (``ccl_latency`` or ``ccl_bw`` will run with the powers of two in that range) 
      * Have in-order queue 


   .. tab-item:: CPU
      
      The following example shows how to run ``ccl_latency`` with the CPU buffers:

      .. code::

         mpirun –n 2 -ppn <P> ccl_latency -b cpu -i 20 -f 1024 -t 67108864
         mpirun –n 2 -ppn <P> ccl_bw -b cpu -i 20 -f 1024 -t 67108864   

      The commands above: 

      * Run the ``ccl_latency/ccl_bw`` benchmark  
      * Contain a total of two processes (this benchmark only supports two processes) 
      * Contain P processes per node, where ``P`` can be ``1`` if running on two different nodes or ``2`` when running on a single node
      * Use CPU buffers 
      * Use 20 iterations 
      * Use element count from ``1024`` to ``67108864`` (``ccl_latency`` will run with the power of two in that range) 

