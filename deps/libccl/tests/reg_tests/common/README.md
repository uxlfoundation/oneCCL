# Purpose

These tests are compiled and run for both possible configurations: cpu-only 
and cpu-gpu. They need to compile with both compiler stacks: gcc and g++ and 
with sycl compilers: icx and icpx.

# Directory structure

This directory contains the `package_tests` directory, which is only enabled 
when whole cpu-gpu package is built with the `build.sh` script and ommitted 
when used with cmake directly: cmake only builds directories for a single 
configuration, which are later assembled into whole cpu-gpu package by the 
`build.sh` script.
