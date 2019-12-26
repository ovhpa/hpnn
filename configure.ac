dnl Process this file with autoconf to produce a configure script.
dnl Re-written OVHPA @2019

AC_INIT(libhpnn, 0.1)

AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([1.11])

AM_SILENT_RULES([yes])

AC_PROG_CC
AC_PROG_LIBTOOL
AC_PROG_MAKE_SET


LT_INIT
AC_MSG_NOTICE(+++---------------------------+++)
AC_MSG_NOTICE(+++ entering standard checks. +++)
AC_MSG_NOTICE(+++---------------------------+++)

# Checks for header files.
AC_CHECK_HEADERS([inttypes.h stdint.h stdlib.h string.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.
AC_CHECK_FUNCS([getcwd memset sqrt])


# PKG_CHECK_MODULES
AC_MSG_NOTICE(+++-------------------------------------+++)
AC_MSG_NOTICE(+++ entering pkg-config related checks. +++)
AC_MSG_NOTICE(+++-------------------------------------+++)
# CHECK for cblas -- this can defines cblas_CFLAGS and cblas_LIBS
PKG_CHECK_MODULES(cblas,
		  [cblas],
		  [pkg_cblas="yes"],
		  [AC_MSG_RESULT([+++ NOT FOUND: CBLAS (not fatal) +++])])
# CHECK for openblas package -- this can defines openblas_CFLAGS and openblas_LIBS
PKG_CHECK_MODULES(openblas,
		  [openblas],
		  [pkg_openblas="yes"],
		  [AC_MSG_RESULT([+++ NOT FOUND: OPENBLAS (not fatal) +++])])
# CHECK for cuda package -- this can defines cuda_CFLAGS and cuda_LIBS
PKG_CHECK_MODULES(cuda,
		  [cuda],
		  [pkg_cuda="yes"],
		  [AC_MSG_RESULT([+++ NOT FOUND: CUDA (not fatal) +++])])
# CHECK for cublas package -- this can defines cublas_CFLAGS and cublas_LIBS
PKG_CHECK_MODULES(cublas,
		  [cublas],
		  [pkg_cublas="yes"],
		  [AC_MSG_RESULT([+++ NOT FOUND: CUBLAS (not fatal) +++])])
# CHECK for mpi -- this can defines mpi_CFLAGS and mpi_LIBS
PKG_CHECK_MODULES(mpi,
		  [mpi],
		  [pkg_mpi="yes"],
		  [AC_MSG_RESULT([+++ NOT FOUND: MPI (not fatal) +++])])
# CHECK for openmpi (C compiler) -- this can defines ompic_CFLAGS and ompic_LIBS
PKG_CHECK_MODULES(ompic,
		  [ompi-c],
		  [pkg_ompic="yes"],
		  [AC_MSG_RESULT([+++ NOT FOUND: OMPI-C (not fatal) +++])])
# CHECK for mkl
# TODO

# now check for non pkg-config libraries/installations
AC_MSG_NOTICE(+++---------------------------------------------+++)
AC_MSG_NOTICE(+++ entering libraries/features related checks. +++)
AC_MSG_NOTICE(+++---------------------------------------------+++)




AC_OUTPUT([
Makefile
src/libhpnn-0.1.pc
src/Makefile
])
