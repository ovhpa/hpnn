---
name: libhpnn BUG report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: ovhpa

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1 - initialization of libhpnn AND of separate (MPI/OpenMP/CUDA) elements when done outside of libhpnn.
2 - data passed to libhpnn
3 - each call to libhpnn API
4 - additional operation from your executable that might tamper with libhpnn.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Actual behavior**
What is actually happening (MUST be different from above).

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Desktop (please complete the following information):**
 - OS: [e.g. GNU/Linux]
 - libhpnn capability (and/or configure flags)
 - libhpnn external library versions
 + on *NIX the result of `ldd libhpnn.so` can suffice.

**Additional context**
Add any other context about the problem here.
