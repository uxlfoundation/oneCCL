---
name: Report a bug or a performance issue
about: Use this template to report unexpected behavior
title: ''
labels: 'bug'
assignees: ''
---

# Summary

Provide a short summary of the issue.

# Version and environment

Include:

- Commit hash ID of oneCCL or release number if the official release is used
- MPI version (or "bundled" if the bundled mpi is used)
- Compiler type and version
- OS name and version
- GPU driver information; agama version, if agama package is used
- Hardware configuration:
    - CPU and GPU versions
    - GPU interconnect
    - Network configuration

In case of regression, report Git hash for the last known working revision.

# Reproducer

A reproducer script should be supplied.

The reproducer script should include all steps required to reproduce
the issue. The script should include the commands used for building
oneCCL from a clean git repository or for installing the oneCCL package.

Share the versions of all packages, preferably as a part of installation steps.

# Logs

Include logs obtained with `CCL_LOG_LEVEL=trace` environment variable. The logs are printed by oneCCL to standard output (stdout).

# Expected behavior

Describe what was expected to happen.

# Observed behavior

Describe what was observed. Explain how it differs from the expected behavior.

# Existing workarounds

Does any workaround exist?

# Affected projects

Please share how the bug affects your project. This will help us asses
the criticality of the bug.
