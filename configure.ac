dnl Process this file with autoconf to produce a configure script.
dnl Re-written OVHPA @2019

AC_INIT(libhpnn, 0.2)

AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([1.11])

AM_SILENT_RULES([yes])

AC_PROG_CC
AC_PROG_MAKE_SET

# +++ deal with switches +++
# MPI
AC_MSG_CHECKING([if MPI was required])
AC_ARG_WITH([mpi],
            [AS_HELP_STRING([--with-mpi],
              [compile with [MPI] support @<:@default=auto@:>@])],
            [],
            [with_mpi=auto])
AC_MSG_RESULT([$with_mpi])
# OMP
AC_MSG_CHECKING([if OMP was required])
AC_ARG_WITH([omp],
            [AS_HELP_STRING([--with-omp],
              [compile with [OpenMP] support @<:@default=auto@:>@])],
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
              [determine which BLAS flavor to use @<:@default=auto@:>@\
		valid values: [auto,yes,no,none,openblas,mkl,atlas]])],
            [],
            [with_blas=auto])
AC_MSG_RESULT([$with_blas])
# -------
# OpenMPI
# -------
# first of all determine if we need to switch compiler
AC_MSG_NOTICE(+++------------+++)
AC_MSG_NOTICE(+++ check MPI. +++)
AC_MSG_NOTICE(+++------------+++)
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
else
	AC_MSG_NOTICE(^^^ REQUESTED: NO MPI ^^^)
fi
#FAIL MPI if we need to (ie user asked for MPI but none could be found)
if test "x$with_mpi" = xyes; then
	if test "x$pkg_mpi" != xyes -a "x$use_mpicc" = xnone; then
		AC_MSG_ERROR(MPI was requested with --use-mpi=yes but no MPI found!)
	fi
fi
# --------
# COMPILER
# --------
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
# ------------
# INIT LIBTOOL
# ------------
# to avoid strange libtool errors this has to be done after
# switching to mpicc (in case we actually switch). -- OVHPA
LT_INIT
if test "x$is_icc" != xyes; then
	#intel compiler include math library
	AC_SEARCH_LIBS([sqrt], [m])
	AC_CHECK_FUNCS([getcwd memset exp sqrt])
fi
# --------------
# +++ OpenMP +++
# --------------
AC_MSG_NOTICE(+++---------------+++)
AC_MSG_NOTICE(+++ check OpenMP. +++)
AC_MSG_NOTICE(+++---------------+++)
#if not --with-omp=no test for OpenMP header
use_omp="unused"
if test "x$with_omp" != xno -a "x$with_omp" != xnone; then
	AC_CHECK_HEADERS([omp.h],[use_omp='yes'], [use_omp='no'])
else
	use_omp=no
	AC_MSG_NOTICE(^^^ REQUESTED: NO OpenMP ^^^)
fi
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
# ----
# BLAS
# ----
AC_MSG_NOTICE(+++-------------+++)
AC_MSG_NOTICE(+++ check BLAS. +++)
AC_MSG_NOTICE(+++-------------+++)
#use mkl if available
auto_blas='none'
if test "x$with_blas" = xauto; then
	AC_CHECK_HEADERS([mkl.h mkl_cblas.h],[auto_blas='mkl'])
	AS_UNSET([ac_cv_header_mkl_h])
	AS_UNSET([ac_cv_header_mkl_cblas_h])
	if test "x$auto_blas" = xmkl; then
		AC_MSG_NOTICE([^^^ auto_blas: found MKL ^^^])
	fi
fi
#use openblas if available
if test "x$with_blas" = xauto -a "x$auto_blas" = xnone; then
	AC_SEARCH_LIBS([cblas_dgemv],[openblaso openblas],[auto_blas='openblas'])
	AS_UNSET([ac_cv_func_cblas_dgemv])
	if test "x$auto_blas" = xopenblas; then
		AC_MSG_NOTICE([^^^ auto_blas: found OPENBLAS ^^^])
	fi
fi
if test "x$with_blas" = xauto -a "x$auto_blas" = xnone; then
	AC_SEARCH_LIBS([ATL_dgemv],[tatlas satlas],[auto_blas='atlas'])
	AS_UNSET([ac_cv_func_ATL_dgemv])
	if test "x$auto_blas" = xatlas; then
		AC_MSG_NOTICE([^^^ auto_blas: found ATLAS ^^^])
	fi
fi
#check fo specific BLAS requirement
#unless --with-blas=no or --with-blas=none switch is used
if test "x$with_blas" != xno -a "x$with_blas" != xnone; then
	# II - specific cases
	if test "x$with_blas" = xmkl -o "x$auto_blas" = xmkl; then
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
	if test "x$with_blas" = xopenblas -o "x$auto_blas" = xopenblas; then
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
	if test "x$with_blas" = xatlas -o "x$auto_blas" = xatlas; then
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
else
	AC_MSG_NOTICE(^^^ REQUESTED: NO BLAS ^^^)
fi
# Checks for typedefs, structures, and compiler characteristics.
#TODO - optimization level -O3 for intel compiler can't be use with MPI
# ----
# CUDA
# ----
AC_MSG_NOTICE(+++-------------+++)
AC_MSG_NOTICE(+++ check CUDA. +++)
AC_MSG_NOTICE(+++-------------+++)
if test "x$with_cuda" != xno -a "x$with_cuda" != xnone; then
	auto_cuda=no
	AC_CHECK_HEADERS([cuda_runtime.h],[auto_cuda='yes'])
	# fail if we have to
	if test "x$with_cuda" = xyes -a "x$auto_cuda" != xyes; then
		AC_MSG_ERROR(CUDA was requested with --use-cuda=yes but no cuda_runtime.h header was not found!)
	fi
	if test "x$auto_cuda" = xyes; then
		AC_CHECK_PROGS(use_nvcc, nvcc, none)
		if test "x$use_nvcc" != xnone; then
			#do we have cuda_runtime?
			nvcc_cmp='ok'
			AC_MSG_CHECKING([if CUDA compiler ($use_nvcc) is OK])
			AC_MSG_RESULT([$nvcc_cmp])
			NVCC="$use_nvcc"
			AC_SUBST(NVCC)
			CFLAGS+=" -D_CUDA"
			NVCCFLAGS+="-w --device-w --default-stream per-thread --use_fast_math -lineinfo -D_CUDA"
			AC_SUBST(NVCCFLAGS)
			LDFLAGS+=" -lcudart -lstdc++"
		fi
	fi
else
	AC_MSG_NOTICE(^^^ REQUESTED: NO CUDA ^^^)
fi
AM_CONDITIONAL([HAVE_CUDA], [test "x$nvcc_cmp" = "xok"])
AC_SUBST(HAVE_CUDA)

# ---------
# Makefiles
# ---------
AC_OUTPUT([
Makefile
src/libhpnn.pc
src/Makefile
tests/Makefile
])
