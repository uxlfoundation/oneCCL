.. highlight:: bash

=====================================
Execution of Communication Operations
=====================================

Communication operations are executed by CCL worker threads (workers). The number of workers is controlled by the :ref:`CCL_WORKER_COUNT <CCL_WORKER_COUNT>` environment variable.

Workers affinity is controlled by :ref:`CCL_WORKER_AFFINITY <CCL_WORKER_AFFINITY>`.

By setting workers affinity you can specify which CPU cores are used by CCL workers. The general rule of thumb is to use different CPU cores for compute (e.g. by specifying ``KMP_AFFINITY``) and for CCL communication.

There are two ways to set workers affinity: automatic and explicit.

Automatic setup
###############

To set affinity automatically, set ``CCL_WORKER_AFFINITY`` to ``auto``.

.. rubric:: Example

In the example below, |product_short| creates four workers per process and pins them to the last four cores available for the process (available if ``mpirun`` launcher from |product_short| package is used, the exact IDs of CPU cores depend on the parameters passed to ``mpirun``) or to the last four cores on the node.
::

   export CCL_WORKER_COUNT=4
   export CCL_WORKER_AFFINITY=auto

Explicit setup
##############

To set affinity explicitly for all local workers, pass ID of the cores to the ``CCL_WORKER_AFFINITY`` environment variable.

.. rubric:: Example

In the example below, |product_short| creates 4 workers per process and pins them to cores with numbers 3, 4, 5, and 6, respectively:
::

   export CCL_WORKER_COUNT=4
   export CCL_WORKER_AFFINITY=3,4,5,6
