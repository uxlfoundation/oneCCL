.. _`here`: https://github.com/ofiwg/libfabric/releases/tag/v1.13.2
.. _`documentation`: https://one-api.gitlab-pages.devtools.intel.com/level_zero/core/PROG.html#affinity-mask

=================================
Enabling OFI/verbs/dmabuf Support
=================================

|product_short| provides experimental support for data transfers between Intel GPU memory and NIC using Linux dmabuf, which is exposed through OFI API for verbs provider.


Requirements
############

- Linux kernel version >= 5.12
- RDMA core version >= 34.0
- level-zero-devel package


Usage
#####

|product_short|, OFI and OFI/verbs from |base_tk| support device memory transfers. Refer to `Run instructions`_ for usage.

If you want to build software components from sources, refer to `Build instructions`_.


Build instructions
##################

OFI
***

::

    git clone --single-branch --branch v1.13.2 https://github.com/ofiwg/libfabric.git
    cd libfabric
    ./autogen.sh
    ./configure --prefix=<ofi_install_dir> --enable-verbs=<rdma_core_install_dir> --with-ze=<level_zero_install_dir> --enable-ze-dlopen=yes
    make -j install

.. note::
    You may also get OFI release package directly from `here`_.
    No need to run autogen.sh if using the release package.

|product_short|
***************

::

    cmake -DCMAKE_INSTALL_PREFIX=<ccl_install_dir> -DLIBFABRIC_DIR=<ofi_install_dir> -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCOMPUTE_BACKEND=dpcpp -DENABLE_OFI_HMEM=1 ..
    make -j install


Run instructions
################

1. Set the environment. See `Get Started Guide <https://www.intel.com/content/www/us/en/docs/oneccl/get-started-guide/2021-10/overview.html>`_.

2. Run allreduce test with ring algorithm and SYCL USM device buffers:

   ::

       export CCL_ATL_TRANSPORT=ofi
       export CCL_ATL_HMEM=1
       export CCL_ALLREDUCE=ring
       export FI_PROVIDER=verbs
       mpiexec -n 2 <ccl_install_dir>/examples/sycl/sycl_allreduce_usm_test gpu device
