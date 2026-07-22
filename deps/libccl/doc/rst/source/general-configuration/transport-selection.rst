.. _mpi: https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html

===================
Transport Selection
===================

|product_short| supports two transports for inter-process communication: |mpi|_ and `libfabric* <https://github.com/ofiwg/libfabric>`_.

The transport selection is controlled by :ref:`CCL_ATL_TRANSPORT <CCL_ATL_TRANSPORT>`.

In case of MPI over libfabric implementation (for example, |mpi| 2021) or in case of direct libfabric transport, the selection of specific libfabric provider is controlled by the ``FI_PROVIDER`` environment variable.
