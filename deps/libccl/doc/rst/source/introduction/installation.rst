.. |sys_req| replace:: |product_full| System Requirements
.. _sys_req: https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-collective-communication-library-system-requirements.html
.. |tgz_file| replace:: tar.gz file
.. _tgz_file: https://github.com/oneapi-src/oneCCL/releases
.. highlight:: bash

==========================
Install oneCCL from Source
==========================

.. note::
   In |product_short| version 2021.17 included with the 2025.3 oneAPI release, oneCCL has added support for a new **C API** that closely follows the NVIDIA Collective Communications Libary (NCCL)* API standard. Details about the new API, instructions on how to build, and run an example can be found `here <../v2/index.html>`_.

   The existing **C++ API** will remain the default API for the 2021.17 release.

Install and configure the |product_full| (|product_short|) on your system.
|product_short| supports different installation scenarios using command line interface (CLI).

System Requirements
*******************

Ensure your system meets the hardware and software requirements before starting with installing oneCCL. See |sys_req|_ to learn about hardware and software requirements for |product_short|.

Installation using CLI
**********************

To install |product_short| using command line interface (CLI), follow these steps:

#. Clone the OneCCL git repository:

   ::

      git clone https://github.com/oneapi-src/oneCCL.git

#. Navigate to the oneCCL folder:

   ::

      cd oneCCL

#. Create a new folder:

   ::

      mkdir build

#. Go to the folder created:

   ::

      cd build

#. Launch CMake:

   ::

      cmake ..

#. Install the product:

   ::

      make -j install

If you need a clean build, create a new ``build`` directory and invoke ``cmake`` within the directory.

Custom Installation
^^^^^^^^^^^^^^^^^^^

You can customize CLI-based installation (for example, you can specify the directory, compiler, and build type):

* To specify the **installation directory**, modify the ``cmake`` command:

  ::

    cmake .. -DCMAKE_INSTALL_PREFIX=</path/to/installation/directory>

  If no ``-DCMAKE_INSTALL_PREFIX`` is specified, |product_short| is installed into the ``_install`` subdirectory of the current build directory (the default installation path will be ``oneCCL/build/_install``).

* To specify the **compiler**, modify the ``cmake`` command:

  ::

     cmake .. -DCMAKE_C_COMPILER=<c_compiler> -DCMAKE_CXX_COMPILER=<cxx_compiler>

.. _enable_sycl:

* To enable ``SYCL`` devices communication support, specify ``SYCL`` compiler (only Intel\ |reg|\  oneAPI DPC++/C++ Compiler is supported):

  ::

     cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCOMPUTE_BACKEND=dpcpp

* To specify the **build type**, modify the ``cmake`` command:

  ::

     cmake .. -DCMAKE_BUILD_TYPE=[Debug|Release]

* To enable ``make`` verbose output to see all parameters used by ``make`` during compilation and linkage, modify the ``make`` command as follows:

  ::

     make -j VERBOSE=1 install

Learn More
***********
- `oneCCL Get Started Guide <https://www.intel.com/content/www/us/en/docs/oneccl/get-started-guide/current/overview.html>`_
- `oneCCL GitHub Source Code Repository <https://github.com/oneapi-src/oneCCL>`_
- `oneCCL Documentation <https://oneapi-src.github.io/oneCCL/>`_
