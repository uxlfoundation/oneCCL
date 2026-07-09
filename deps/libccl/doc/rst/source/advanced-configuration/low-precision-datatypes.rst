.. _`bfloat16`: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
.. _`float16`: https://en.wikipedia.org/wiki/Half-precision_floating-point_format

=======================
Low-precision Datatypes
=======================

|product_short| provides support for collective operations on low-precision (LP) datatypes (`bfloat16`_ and `float16`_).

Reduction of LP buffers (for example as phase in ``ccl::allreduce``) includes conversion from LP to FP32 format, reduction of FP32 values and conversion from FP32 to LP format.

|product_short| utilizes CPU vector instructions for FP32 <-> LP conversion.

For BF16 <-> FP32 conversion |product_short| provides ``AVX512F`` and ``AVX512_BF16``-based implementations.
``AVX512F``-based implementation requires GCC 4.9 or higher. ``AVX512_BF16``-based implementation requires GCC 10.0 or higher and GNU binutils 2.33 or higher.
``AVX512_BF16``-based implementation may provide less accuracy loss after multiple up-down conversions.

For FP16 <-> FP32 conversion |product_short| provides ``F16C`` and ``AVX512F``-based implementations.
Both implementations require GCC 4.9, Clang 9.0 or higher.

|product short| utilizes CPU vector instructions for LP numeric operations.

For FP16 numeric operations (arithmetic, load, store) |product short| provides ``AVX512FP16``-based implementation.
This implementation requires GCC 12.0, Clang 14.0, Intel 2021.4.0 or higher.

Refer to :ref:`Low-precision datatypes <low-precision-datatypes>` for details about relevant environment variables.
