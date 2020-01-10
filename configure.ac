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
# MPI
AC_MSG_CHECKING([if MPI was required])
AC_ARG_WITH([mpi],
            [AS_HELP_STRING([--with-mpi],
              [compile with MPI support @<:@default=auto@:>@])],
            [],
            [with_mpi=auto])
AC_MSG_RESULT([$with_mpi])
# OMP
AC_MSG_CHECKING([if OMP was required])
AC_ARG_WITH([omp],
            [AS_HELP_STRING([--with-omp],
              [compile with OpenMP support @<:@default=auto@:>@])],
            [],
            [with_omp=auto])
AC_MSG_RESULT([$with_omp])
# CUDA
AC_MSG_CHECKING([if CUDA was required])
AC_ARG_WITH([cuda],
            [AS_HELP_STRING([--with-cuda],
              [compile with [CUDA] support @<:@default=auto@:>@])],
            [],
            [with_cuda=auto])
AC_MSG_RESULT([$with_cuda])
# CUBLAS
# it is _possible_ to use naive cuda interface without CUBLAS
AC_MSG_CHECKING([if CUBLAS was required])
AC_ARG_WITH([cublas],
            [AS_HELP_STRING([--with-cublas],
              [compile with [CUBLAS] support @<:@default=auto@:>@])],
            [],
            [with_cublas=auto])
AC_MSG_RESULT([$with_cublas])
# switch "magic"
# --with-cuda=no set --with-cublas=no from --with-cublas=auto
if test "x$with_cuda" = xno; then
	if test "x$with_cublas" = xauto; then
		AC_MSG_NOTICE([^^^ disable CUBLAS (no CUDA) ^^^])
		with_cublas='no'
	fi
fi
# cublas req cuda
if test "x$with_cublas" = xyes; then
	if test "x$with_cuda" = xno; then
		AC_MSG_ERROR(CUBLAS was required without CUDA?!)
	fi
fi
# --with-cublas=yes set --with-cuda=yes from --with-cuda=auto
if test "x$with_cublas" = xyes; then
	if test "x$with_cuda" = xauto; then
		AC_MSG_NOTICE([^^^ forcing CUDA on (CUBLAS required) ^^^])
		with_cuda='yes'
	fi
fi
# BLAS
AC_MSG_CHECKING([if specific BLAS was required])
AC_ARG_WITH([blas],
            [AS_HELP_STRING([--with-blas],
              [determine which BLAS flavor to use @<:@default=auto@:>@])],
            [],
            [with_blas=auto])
AC_MSG_RESULT([$with_blas])


AC_MSG_NOTICE(+++---------------------------+++)
AC_MSG_NOTICE(+++ entering standard checks. +++)
AC_MSG_NOTICE(+++---------------------------+++)

# first of all determine if we need to switch compiler
if test "x$with_mpi" != xno; then
	AC_CHECK_PROGS(use_mpicc, mpicc hcc mpxlc_r mpxlc mpcc cmpicc, none)
	AC_MSG_CHECKING([if current compiler ($CC) is MPI-able])
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
	if test "x$mpicc_gcc" = xno; then
		# $use_mpicc is NOT current CC compiler: try to reset CC
		ORIG_CC="$CC"
		CC="$use_mpicc"
		AC_MSG_CHECKING([if we can switch to $use_mpicc])
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
		if test "x$mpicc_gcc" = xno; then
			AC_MSG_NOTICE(^^^ $use_mpicc detected but not usable, swiching back to $PUSH_CC)
			CC="$ORIG_CC"
		else
			AC_MSG_NOTICE(^^^ WARNING: CC compiler is now $CC ^^^)
			CFLAGS+=" -D_MPI"
		fi
	fi
fi
#determine if compiler is INTEL
AC_MSG_CHECKING([if current compiler ($CC) is intel compiler])
AC_LANG_PUSH(C)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include <stdio.h>
]],[[
#ifndef __ICC
#error "NOT ICC"
#endif
return 0;
]])],[is_icc='yes'], [is_icc='no'])
AC_LANG_POP()
AC_MSG_RESULT([$is_icc])
# Checks for header files.
AC_CHECK_HEADERS([inttypes.h stdint.h stdlib.h string.h unistd.h])
#if not --with-omp=no test for OpenMP header
if test "x$with_omp" != xno; then
	AC_CHECK_HEADERS([omp.h],[use_omp='yes'], [use_omp='no'])
else
	use_omp=no
fi
#auto BLAS flavous (TODO)
	# I - with auto, try in order:
	#1 - mkl if intel compiler
	#2 - openblas
	#3 - generic cblas
#check fo specific BLAS requirement
#unless --with-blas=no or --with-blas=none switch is used
if test "x$with_blas" != xno -o "x$with_blas" != xnone -o "x$with_blas" != auto; then
	# II - specific cases
	if test "x$with_blas" = xmkl; then
		AC_CHECK_HEADERS([mkl.h mkl_cblas.h],[use_mkl='yes'],[use_mkl='no'])
		if test "x$is_icc" = xyes; then
			#if CC is intel compiler only -mkl is necessary
			#+ the default is to use parallel blas with mkl
			CFLAGS+=" -mkl=parallel -D_MKL -DPBLAS"
		else
			#we don't have intel compiler...
			if test "x$use_mkl" = xyes; then
				#but found mkl.h
				#maybe user gave correct FLAGS?
				AC_CHECK_FUNCS([cblas_dgemv cblas_dger mkl_set_dynamic mkl_set_num_threads_local],[can_mkl='yes'],[can_mkl='no'])
				if test "x$can_mkl" = xyes; then
					CFLAGS+=" -D_MKL -DPBLAS"
				else
					#try to setup a default gcc target...
					#slowly... one lib at a time
					AC_SEARCH_LIBS([dlopen],[dl])# -ldl
					AC_SEARCH_LIBS([sqrt],[m])# -lm
					AC_SEARCH_LIBS([pthread_create], [pthread])# -lpthread
					AC_SEARCH_LIBS([omp_set_num_threads],[gomp])# -lgomp
					AC_SEARCH_LIBS([mkl_blas_xdgemv],[mkl_core],[],[],[-lmkl_sequential])# -lmkl_core
					AC_SEARCH_LIBS([mkl_blas_dgemv_omp],[mkl_gnu_thread],[],[],[-lmkl_sequential])# -lmkl_gnu_threads
					AC_SEARCH_LIBS([cblas_dgemv], [mkl_gf_lp64],[can_mkl='yes'],[can_mkl='no'])
					#check again (kill the cache before!)
					AS_UNSET([ac_cv_func_cblas_dgemv])
					AS_UNSET([ac_cv_func_cblas_dger])
					AS_UNSET([ac_cv_func_mkl_set_dynamic])
					AS_UNSET([ac_cv_func_mkl_set_num_threads_local])
					AC_CHECK_FUNCS([cblas_dgemv cblas_dger mkl_set_dynamic mkl_set_num_threads_local],[can_mkl='yes'],[can_mkl='no'])
					if test "x$can_mkl" = xyes; then
						CFLAGS+=" -D_MKL -DPBLAS"
					else
						AC_MSG_ERROR(MKL was requested with --use-blas=mkl but LDFLAGS can't bet set!)
					fi
				fi
			else
				AC_MSG_ERROR(MKL was requested with --use-blas=mkl but mkl.h can't be found!)
			fi
		fi
	fi
	if test "x$with_blas" = xopenblas; then
		AC_CHECK_HEADERS([cblas.h],[use_cblas='yes'],[use_cblas='no'])
		if test "x$use_cblas" = xyes; then
			AC_SEARCH_LIBS([cblas_dgemv],[openblaso],[can_pblas='yes'],[can_pblas='no'])
			if test "x$can_pblas" != xyes; then
				#sometimes openblaso parallel library is not available
				AC_SEARCH_LIBS([cblas_dgemv],[openblas],[can_sblas='yes'],[can_sblas='no'])
				if test "x$can_sblas" != xyes; then
					AC_MSG_ERROR(OPENBLAS was requested with --use-blas=openblas but LDFLAGS can't be set!)
				else
					AC_MSG_NOTICE([^^^ WARNING! PARALLEL OPENBLAS COULD NOT BE FOUND! ^^^])
					AC_MSG_NOTICE([^^^ !!!!!!!! WILL USE SEQUENTIAL OPENBLAS INSTEAD! ^^^])
					AC_MSG_NOTICE([^^^ !!!!!!!! NOTE: PERFORMANCE WILL SUFFER (much)! ^^^])
					AC_CHECK_FUNCS([cblas_ddot cblas_daxpy cblas_dscal cblas_dger])
					CFLAGS+=" -DSBLAS"
				fi
			else
				AC_CHECK_FUNCS([cblas_dgemv cblas_dger])
				CFLAGS+=" -DPBLAS"
			fi
		else
			AC_MSG_ERROR(OPENBLAS was requested with --use-blas=openblas but cblas.h can't be found!)
		fi
	fi
	if test "x$with_blas" = xatlas; then
		#libraries such as atlas use [-lcblas -ltatlas] [-ltatlas] or [-lsatlas]
		AC_CHECK_HEADERS([cblas.h],[use_cblas='yes'],[use_cblas='no'])
		if test "x$use_cblas" = xyes; then
			AC_SEARCH_LIBS([ATL_dgemv],[tatlas])# -ltatlas
			#this one might not be needed
			AC_SEARCH_LIBS([cblas_dgemv],[cblas],[can_pblas='yes'],[can_pblas='no'])# -lcblas
			if test "x$can_pblas" != xyes; then
				#try sequential version...
				AC_SEARCH_LIBS([ATL_dgemv],[satlas])# -lsatlas
				#this one might not be needed
				AC_SEARCH_LIBS([cblas_ddot],[cblas],[can_sblas='yes'],[can_sblas='no'])# -lcblas
				if test "x$can_sblas" != xyes; then
					AC_MSG_ERROR(ATLAS BLAS was requested with --use-blas=atlas but LDFLAGS can't be set!)
				else
					AC_MSG_NOTICE([^^^ WARNING! PARALLEL OPENBLAS COULD NOT BE FOUND! ^^^])
					AC_MSG_NOTICE([^^^ !!!!!!!! WILL USE SEQUENTIAL OPENBLAS INSTEAD! ^^^])
					AC_MSG_NOTICE([^^^ !!!!!!!! NOTE: PERFORMANCE WILL SUFFER (much)! ^^^])
					AC_CHECK_FUNCS([cblas_ddot cblas_daxpy cblas_dscal cblas_dger])
					CFLAGS+=" -DSBLAS"
				fi
			else
				AC_CHECK_FUNCS([cblas_dgemv cblas_dger])
				CFLAGS+=" -DPBLAS"
			fi
		else
			AC_MSG_ERROR(CBLAS was requested with --use-blas=atlas but cblas.h can't be found!)
		fi
	fi
fi
# Checks for typedefs, structures, and compiler characteristics.
#TODO - optimization level -O3 for intel compiler can't be use with MPI

# Checks for library functions.
AC_SEARCH_LIBS([sqrt], [m])
AC_CHECK_FUNCS([getcwd memset exp sqrt])

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
# do not check if --with-cuda=no
if test "x$with_cuda" != xno; then
PKG_CHECK_MODULES(cuda,
		  [cuda],
		  [pkg_cuda="yes"],
		  [AC_MSG_RESULT([+++ NOT FOUND: CUDA (not fatal) +++])])
fi
# CHECK for cublas package -- this can defines cublas_CFLAGS and cublas_LIBS
# do not check if --with-cublas=no
if test "x$with_cublas" != xno; then
PKG_CHECK_MODULES(cublas,
		  [cublas],
		  [pkg_cublas="yes"],
		  [AC_MSG_RESULT([+++ NOT FOUND: CUBLAS (not fatal) +++])])
fi
# MPI checks
# do not check if --with-mpi=no
if test "x$with_mpi" != xno; then
	#only check if not MPI-able
	if test "x$mpi_gcc" = xno; then
		#first check for ompi-c  -- this can defines ompic_CFLAGS and ompic_LIBS
		PKG_CHECK_MODULES(ompic,
			  [ompi-c],
			  [pkg_ompic="yes"],
			  [AC_MSG_RESULT([+++ NOT FOUND: OMPI-C (not fatal) +++])])
		if test "x$pkg_ompic" != xyes; then
			#check MPI -- this can defines mpi_CFLAGS and mpi_LIBS
			PKG_CHECK_MODULES(mpi,
				  [mpi],
				  [pkg_mpi="yes"],
				  [AC_MSG_RESULT([+++ NOT FOUND: MPI (not fatal) +++])])
		fi
	fi
fi
# CHECK for mkl
# TODO

# now check for libraries/installations
AC_MSG_NOTICE(+++---------------------------------------------+++)
AC_MSG_NOTICE(+++ entering libraries/features related checks. +++)
AC_MSG_NOTICE(+++---------------------------------------------+++)
#FAIL MPI if we need to (ie user asked for MPI but none could be found)
if test "x$with_mpi" = xyes; then
	if test "x$pkg_mpi" != xyes -a "x$use_mpicc" = xnone; then
		AC_MSG_ERROR(MPI was requested with --use-mpi=yes but no MPI found!)
	fi
fi
# OMP
if test "x$use_omp" = xyes; then
	#try to determine the FLAGS for openmp
	AC_LANG_PUSH(C)
	AC_OPENMP
	AC_SUBST(OPENMP_CFLAGS)
	AC_MSG_NOTICE(^^^ OpenMP detected and configured ^^^)
	CFLAGS="${CFLAGS} ${OPENMP_CFLAGS}"
	AC_LANG_POP()
	# _OMP switch is also necessary!
	CFLAGS+=" -D_OMP"
else
	#in case we don't use OpenMP, don't spam with unknown-pragma
	CFLAGS+=" -Wno-unknown-pragmas"
fi
#FAIL OMP if we need to (ie user asked for OMP but omp.h could not be found)
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
