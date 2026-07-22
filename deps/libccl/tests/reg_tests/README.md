# Directory structure

This directory contains the following subdirectories:

- cpu: tests for the CPU-only configuration, to be compiled with gcc and g++,
- sycl: gpu tests that utilze SYCL, to be compiled with icx and icpx,
- common: tests that should be run for both configurations: gpu and CPU, 
compiled and tested with both compilers
