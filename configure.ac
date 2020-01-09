dnl Process this file with autoconf to produce a configure script.
dnl Re-written OVHPA @2019

AC_INIT(libhpnn, 0.2)

AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([1.11])

AM_SILENT_RULES([yes])

AC_PROG_CC
AC_PROG_LIBTOOL
AC_PROG_MAKE_SET

LT_INIT

# +++ deal with switches +++
# CUDA
AC_MSG_CHECKING([if CUDA was required])
AC_ARG_WITH([libcuda],
            [AS_HELP_STRING([--without-cuda],
              [disable support for cuda])],
            [],
            [with_cuda=yes])
AC_MSG_RESULT([$with_cuda])
# CUBLAS
# it is _possible_ to use naive cuda interface without CUBLAS
AC_MSG_CHECKING([if CUBLAS was required])
AC_ARG_WITH([libcublas],
            [AS_HELP_STRING([--without-cublas],
              [disable support for cublas])],
            [],
            [with_cublas=yes])
AC_MSG_RESULT([$with_cublas])
# MPI
AC_MSG_CHECKING([if MPI was required])
AC_ARG_WITH([libmpi],
            [AS_HELP_STRING([--with-mpi],
              [compile with MPI support @<:@default=auto@:>@])],
            [],
            [with_mpi=auto])
AC_MSG_RESULT([$with_mpi])
# OMP
AC_MSG_CHECKING([if OMP was required])
AC_ARG_WITH([libomp],
            [AS_HELP_STRING([--with-omp],
              [compile with OpenMP support @<:@default=auto@:>@])],
            [],
            [with_omp=auto])
AC_MSG_RESULT([$with_omp])


AC_MSG_NOTICE(+++---------------------------+++)
AC_MSG_NOTICE(+++ entering standard checks. +++)
AC_MSG_NOTICE(+++---------------------------+++)

# Checks for header files.
AC_CHECK_HEADERS([inttypes.h stdint.h stdlib.h string.h unistd.h])
AC_CHECK_HEADERS([omp.h],[use_omp='yes'], [use_omp='no'])

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
if test "x$with_mpi" != xno; then
	PKG_CHECK_MODULES(mpi,
		  [mpi],
		  [pkg_mpi="yes"],
		  [AC_MSG_RESULT([+++ NOT FOUND: MPI (not fatal) +++])])
fi
# CHECK for openmpi (C compiler) -- this can defines ompic_CFLAGS and ompic_LIBS
PKG_CHECK_MODULES(ompic,
		  [ompi-c],
		  [pkg_ompic="yes"],
		  [AC_MSG_RESULT([+++ NOT FOUND: OMPI-C (not fatal) +++])])
# CHECK for mkl
# TODO

# now check for libraries/installations
AC_MSG_NOTICE(+++---------------------------------------------+++)
AC_MSG_NOTICE(+++ entering libraries/features related checks. +++)
AC_MSG_NOTICE(+++---------------------------------------------+++)
# MPI
if test "x$with_mpi" != xno; then
	AC_CHECK_PROGS(use_mpicc, mpicc hcc mpxlc_r mpxlc mpcc cmpicc, none)
	if test "x$pkg_mpi" != xyes; then
		CFLAGS+=" -D_MPI"
		AC_MSG_NOTICE(^^^ no mpi package -> configuring $use_mpicc)
		AC_MSG_CHECKING([if mpicc is current compiler])
		AC_LANG_PUSH(C)
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include <stdio.h>
#include <mpi.h>
]],[[
MPI_Init(NULL, NULL);
MPI_Finalize();
return 0;
]])],[mpicc_gcc='yes'], [mpicc_gcc='no'])
		AC_LANG_POP()
		AC_MSG_RESULT([$mpicc_gcc])
	fi
else
	CFLAGS+=" -D_MPI"
fi
#FAIL if we need to (ie user asked for MPI but none could be found)
if test "x$with_mpi" = xyes; then
	if test "x$pkg_mpi" != xyes -a "x$use_mpicc" = xnone; then
		AC_MSG_ERROR(MPI was requested with --use-mpi=yes but no MPI found!)
	fi
fi
# OMP
if test "x$use_omp" = xyes; then
	# TODO determine if libomp is needed
	CFLAGS+=" -fopenmp -D_OMP"
fi
#FAIL if we need to (ie user asked for OMP but omp.h could not be found)
if test "x$with_omp" = xyes; then
	if test "x$use_omp" != xyes; then
		AC_MSG_ERROR(OMP was requested with --use-omp=yes but no omp.h header was not found!)
	fi
fi


AC_OUTPUT([
Makefile
src/libhpnn.pc
src/Makefile
tests/Makefile
])
