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
AC_ARG_ENABLE([mpi],
	[AS_HELP_STRING([--enable-mpi],
	[enable [MPI] @<:@default=auto if not given@:>@])],
	[],
	[enable_mpi=auto])
AC_MSG_RESULT([$enable_mpi])
AC_ARG_WITH([mpicc],
	    [AS_HELP_STRING([--with-mpicc],
		[specify [MPI] compiler])],
	    [],
	    [with_mpicc=auto])
# OMP
AC_MSG_CHECKING([if OMP was required])
AC_ARG_ENABLE([omp],
	[AS_HELP_STRING([--enable-omp],
	[enable [OpenMP] @<:@default=auto if not given@:>@])],
	[],
	[enable_omp=auto])
AC_MSG_RESULT([$enable_omp])
# CUDA
AC_MSG_CHECKING([if CUDA was required])
AC_ARG_ENABLE([cuda],
	[AS_HELP_STRING([--enable-cuda],
	[enable [CUDA] @<:@default=auto if not given@:>@])],
	[],
	[enable_cuda=auto])
AC_MSG_RESULT([$enable_cuda])
AC_ARG_WITH([cuda_include],
	    [AS_HELP_STRING([--with-cuda-include],
		[specify [CUDA] include path])],
	    [],
	    [with_cuda_include=auto])
AC_ARG_WITH([cuda_lib],
	    [AS_HELP_STRING([--with-cuda-lib],
		[specify [CUDA] library path])],
	    [],
	    [with_cuda_lib=auto])
# CUBLAS
# it is _possible_ to use naive cuda interface without CUBLAS
AC_MSG_CHECKING([if CUBLAS was required])
AC_ARG_ENABLE([cublas],
	[AS_HELP_STRING([--enable-cublas],
	[enable [CUBLAS] @<:@default=auto if not given with CUDA@:>@])],
	[],
	[enable_cublas=auto])
AC_MSG_RESULT([$enable_cublas])
# switch "magic"
# --disable-cuda set --disable-cublas from auto
if test "x$enable_cuda" = xno; then
	if test "x$enable_cublas" = xauto; then
		AC_MSG_NOTICE([^^^ disable CUBLAS (no CUDA) ^^^])
		enable_cublas='no'
	fi
fi
# cublas req cuda
if test "x$enable_cublas" = xyes; then
	if test "x$enable_cuda" = xno; then
		AC_MSG_ERROR(CUBLAS was required without CUDA?!)
	fi
fi
# --with-cublas=yes set --enable-cuda from auto
if test "x$enable_cublas" = xyes; then
	if test "x$enable_cuda" = xauto; then
		AC_MSG_NOTICE([^^^ forcing CUDA on (CUBLAS required) ^^^])
		enable_cuda='yes'
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
AC_ARG_WITH([blas_include],
	    [AS_HELP_STRING([--with-blas-include],
		[specify [BLAS] include path])],
	    [],
	    [with_blas_include=auto])
AC_ARG_WITH([blas_lib],
	    [AS_HELP_STRING([--with-blas-lib],
		[specify [BLAS] library path])],
	    [],
	    [with_blas_lib=auto])
# ENDOF: switches
# -------
# OpenMPI
# -------
# first of all determine if we need to switch compiler
AC_MSG_NOTICE(+++------------+++)
AC_MSG_NOTICE(+++ check MPI. +++)
AC_MSG_NOTICE(+++------------+++)
if test "x$enable_mpi" != xno; then
	if test "x$with_mpicc" = xauto; then
		AC_CHECK_PROGS(use_mpicc, mpicc hcc mpxlc_r mpxlc mpcc cmpicc, none)
	else
		use_mpicc="$with_mpicc"
	fi
	AC_MSG_CHECKING([if current compiler ($CC) is MPI-able])
	AC_LANG_PUSH(C)
	AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include <stdio.h>
#include <mpi.h>
]],[[
MPI_Init(NULL, NULL);
MPI_Finalize();
return 0;]])],[mpicc_gcc='yes'], [mpicc_gcc='no'])
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
return 0;]])],[mpicc_gcc='yes'], [mpicc_gcc='no'])
		AC_LANG_POP()
		AC_MSG_RESULT([$mpicc_gcc])
		if test "x$mpicc_gcc" = xno; then
			if test "x$enable_mpi" = xauto; then
				AC_MSG_NOTICE(^^^ $use_mpicc detected but not usable, swiching back to $PUSH_CC)
				CC="$ORIG_CC"
				enable_mpi="no"
			else
				AC_MSG_ERROR(MPI was requested with --enable-mpi but no mpicc found!)
			fi
		else
			AC_MSG_NOTICE(^^^ WARNING: CC compiler is now $CC ^^^)
			CFLAGS+=" -D_MPI"
		fi
	fi
else
	AC_MSG_NOTICE(^^^ REQUESTED: NO MPI ^^^)
fi
#FAIL MPI if we need to (ie user asked for MPI but none could be found)
if test "x$enable_mpi" = xyes; then
	if test "x$use_mpicc" = xnone; then
		AC_MSG_ERROR(MPI was requested with --enable-mpi but no MPI found!)
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
#determine if compiler is clang
AC_MSG_CHECKING([if current compiler ($CC) is clang compiler])
AC_LANG_PUSH(C)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include <stdio.h>
]],[[
#ifndef __clang__
#error "NOT CLANG"
#endif
return 0;
]])],[is_clang='yes'], [is_clang='no'])
AC_LANG_POP()
AC_MSG_RESULT([$is_clang])
if test "x$is_clang" = xyes; then
	CFLAGS+=" -Wno-absolute-value"
fi
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
#if not --disable-omp test for OpenMP header
use_omp="unused"
if test "x$enable_omp" != xno -a "x$enable_omp" != xnone; then
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
	CFLAGS+=" ${OPENMP_CFLAGS}"
	AC_LANG_POP()
	# _OMP switch is also necessary!
	CFLAGS+=" -D_OMP"
else
	#in case we don't use OpenMP, don't spam with unknown-pragma
	CFLAGS+=" -Wno-unknown-pragmas"
fi
#FAIL OMP if we need to (ie user asked for OMP but omp.h could not be found)
if test "x$enable_omp" = xyes; then
	if test "x$use_omp" != xyes; then
		AC_MSG_ERROR(OMP was requested with --enable-omp but no omp.h header was not found!)
	fi
fi
# ----
# BLAS
# ----
AC_MSG_NOTICE(+++-------------+++)
AC_MSG_NOTICE(+++ check BLAS. +++)
AC_MSG_NOTICE(+++-------------+++)
# PUSH LIBS, CFLAGS, and LDFLAGS in case no BLAS can be configured
ORIG_LIBS="$LIBS"
ORIG_CFLAGS="$CFLAGS"
ORIG_LDFLAGS="$LDFLAGS"
auto_blas='none'
if test "x$with_blas" != xno -o "x$with_blas" != xnone; then
	AC_MSG_CHECKING([if BLAS include PATH was given])
	AS_IF([test "x$with_blas_include" != "xauto"], [
	  AC_MSG_RESULT(yes)
	  AC_MSG_NOTICE(adding ${with_blas_include} to CFLAGS)
	  CFLAGS+=" -I${with_blas_include}"
	],AC_MSG_RESULT(no))
	AC_MSG_CHECKING([if BLAS library PATH was given])
	AS_IF([test "x$with_blas_lib" != "xauto"], [
	  AC_MSG_RESULT(yes)
	  AC_MSG_NOTICE(adding ${with_blas_lib} to LDFLAGS)
	  LDFLAGS+=" -L${with_blas_lib}"
	],AC_MSG_RESULT(no))
fi
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
						if test "x$with_blas" = xmkl; then
							AC_MSG_ERROR(MKL was requested with --with-blas=mkl but LDFLAGS can't bet set!)
						else
							AC_MSG_NOTICE(MKL was found be could not be configured)
							#PULL LIBS
							LIBS="$ORIG_LIBS"
							AC_SEARCH_LIBS([cblas_dgemv],[openblaso openblas],[auto_blas='openblas'])
							AS_UNSET([ac_cv_func_cblas_dgemv])
							if test "x$auto_blas" = xopenblas; then
								AC_MSG_NOTICE([^^^ auto_blas: found OPENBLAS (will try it instead of MKL) ^^^])
							else
								AC_SEARCH_LIBS([ATL_dgemv],[tatlas satlas],[auto_blas='atlas'])
								AS_UNSET([ac_cv_func_ATL_dgemv])
								if test "x$auto_blas" = xatlas; then
									AC_MSG_NOTICE([^^^ auto_blas: found ATLAS (will try it instead of MKL) ^^^])
								else
									AC_MSG_NOTICE([^^^ auto_blas: could not find other BLAS (disable BLAS)^^^])
									#PULL LIBS, CFLAGS, LDFLAGS
									LIBS="$ORIG_LIBS"
									CFLAGS="$ORIG_CFLAGS"
									LDFLAGS="$ORIG_LDFLAGS"
								fi
							fi
						fi
					fi
				fi
			else
				if test "x$with_blas" = xmkl; then
					AC_MSG_ERROR(MKL was requested with --with-blas=mkl but mkl.h can't be found!)
				else
					AC_MSG_NOTICE(MKL could not be configured)
					#PULL LIBS
					LIBS="$ORIG_LIBS"
					AC_SEARCH_LIBS([cblas_dgemv],[openblaso openblas],[auto_blas='openblas'])
					AS_UNSET([ac_cv_func_cblas_dgemv])
					if test "x$auto_blas" = xopenblas; then
						AC_MSG_NOTICE([^^^ auto_blas: found OPENBLAS (will try it instead of MKL) ^^^])
					else
						#PULL LIBS
						LIBS="$ORIG_LIBS"
						AC_SEARCH_LIBS([ATL_dgemv],[tatlas satlas],[auto_blas='atlas'])
						AS_UNSET([ac_cv_func_ATL_dgemv])
						if test "x$auto_blas" = xatlas; then
							AC_MSG_NOTICE([^^^ auto_blas: found ATLAS (will try it instead of MKL) ^^^])
						else
							AC_MSG_NOTICE([^^^ auto_blas: could not find other BLAS (disable BLAS)^^^])
							#PULL LIBS, CFLAGS, LDFLAGS
							LIBS="$ORIG_LIBS"
							CFLAGS="$ORIG_CFLAGS"
							LDFLAGS="$ORIG_LDFLAGS"
						fi
					fi
				fi
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
			if test "x$with_blas" = xopenblas; then
				AC_MSG_ERROR(OPENBLAS was requested with --with-blas=openblas but cblas.h can't be found!)
			else
				AC_MSG_NOTICE(OPENBLAS be could not be configured)
				#PULL LIBS
				LIBS="$ORIG_LIBS"
				AC_SEARCH_LIBS([ATL_dgemv],[tatlas satlas],[auto_blas='atlas'])
				AS_UNSET([ac_cv_func_ATL_dgemv])
				if test "x$auto_blas" = xatlas; then
					AC_MSG_NOTICE([^^^ auto_blas: found ATLAS (will try it instead of MKL) ^^^])
				else
					AC_MSG_NOTICE([^^^ auto_blas: could not find other BLAS (disable BLAS)^^^])
					#PULL LIBS, CFLAGS, LDFLAGS
					LIBS="$ORIG_LIBS"
					CFLAGS="$ORIG_CFLAGS"
					LDFLAGS="$ORIG_LDFLAGS"
				fi
			fi
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
			if test "x$with_blas" = xatlas; then
				AC_MSG_ERROR(ATLAS was requested with --with-blas=atlas but cblas.h can't be found!)
			else
				AC_MSG_NOTICE(ATLAS be could not be configured)
				AC_MSG_NOTICE([^^^ auto_blas: could not find other BLAS (disable BLAS)^^^])
				#PULL LIBS, CFLAGS, LDFLAGS
				LIBS="$ORIG_LIBS"
				CFLAGS="$ORIG_CFLAGS"
				LDFLAGS="$ORIG_LDFLAGS"
			fi
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
AC_CHECK_PROGS([use_nvcc], [nvcc], [none])
if test "x$enable_cuda" != xno -a "x$enable_cuda" != xnone; then
	auto_cuda=no
	AC_MSG_CHECKING([if CUDA include PATH was given])
	AS_IF([test "x$with_cuda_include" != "xauto"], [
	  AC_MSG_RESULT(yes)
	  AC_MSG_NOTICE(adding ${with_cuda_include} to CFLAGS)
	  CFLAGS+=" -I${with_cuda_include}"
	],AC_MSG_RESULT(no))
	AC_MSG_CHECKING([if CUDA library PATH was given])
	AS_IF([test "x$with_cuda_lib" != "xauto"], [
	  AC_MSG_RESULT(yes)
	  AC_MSG_NOTICE(adding ${with_cuda_lib} to LDFLAGS)
	  LDFLAGS+=" -L${with_cuda_lib}"
	],AC_MSG_RESULT(no))
	# PUSH LIBS (to manage lib added but can't configure)
	ORIG_LIBS="$LIBS"
	# TODO: replace that _guess_ with a proper test
	# because it might not be the case in future...
	if test "x$is_icc" != xyes; then
		# gcc require libstdc++ (icc doesn't)
		AC_CHECK_LIB(stdc++, main, , AC_MSG_ERROR([$CC require missing libstdc++ to compile CUDA code]))
	fi
	AC_SEARCH_LIBS([cudaGetDeviceCount],[cudart],[auto_cuda='yes'],[auto_cuda='no'])
	# fail if we have to
	if test "x$enable_cuda" = xyes -a "x$auto_cuda" != xyes; then
		AC_MSG_ERROR(CUDA was requested with --enable-cuda but libcudart was not found!)
	fi
	AC_CHECK_HEADERS([cuda_runtime.h],[auto_cuda='yes'],[auto_cuda='no'])
	# fail if we have to
	if test "x$enable_cuda" = xyes -a "x$auto_cuda" != xyes; then
		AC_MSG_ERROR(CUDA was requested with --enable-cuda but cuda_runtime.h header was not found!)
	fi
	if test "x$auto_cuda" = xyes; then
		if test "x$use_nvcc" = xnone; then
			AC_MSG_ERROR(CUDA was requested with --enable-cuda but nvcc compiler was not found!)
		fi
		NVCC="$use_nvcc"
		AC_SUBST(NVCC)
		CFLAGS+=" -D_CUDA"
		NVCCFLAGS+="-w --device-w --default-stream per-thread --use_fast_math -lineinfo -D_CUDA"
		#now (and only now) check for cublas
		if test "x$enable_cublas" != xno; then
			AC_CHECK_HEADERS([cublas_v2.h],[auto_cublas='yes'],[auto_cublas='no'])
			if test "x$enable_cublas" = xyes -a "x$auto_cublas" != xyes; then
				AC_MSG_ERROR(CUBLAS was requested with --enable-cublas but cublas_v2.h header was not found!)
			fi
			#push LIBS in case CUBLAS can't be found
			ORIG_LIBS="$LIBS"
			AC_SEARCH_LIBS([cublasDgemv],[cublas],[auto_cublas='yes'],[auto_cublas='no'])
			if test "x$auto_cublas" = xyes; then
				CFLAGS+=" -D_CUBLAS"
				NVCCFLAGS+=" -D_CUBLAS"
			else
				AC_MSG_NOTICE(^^^ WARNING: could not configure CUBLAS ^^^)
				LIBS="$ORIG_LIBS"
			fi
		else
			AC_MSG_NOTICE(^^^^ REQUESTED: NO CUBLAS ^^^)
		fi
		AC_SUBST(NVCCFLAGS)
	else
		AC_MSG_NOTICE(^^^ WARNING: could not configure CUDA ^^^)
		use_nvcc='none'
		LIBS="$ORIG_LIBS"
	fi
else
	use_nvcc='none'
	AC_MSG_NOTICE(^^^ REQUESTED: NO CUDA ^^^)
fi
AM_CONDITIONAL([HAVE_CUDA], [test "x$use_nvcc" != "xnone"])
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
# -------
# SUMMARY
# -------
AC_MSG_NOTICE(+++---------+++)
AC_MSG_NOTICE(+++ SUMMARY +++)
AC_MSG_NOTICE(+++---------+++)
AC_MSG_NOTICE(>>> CC = $CC )
AC_MSG_NOTICE(>>> CFLAGS = $CFLAGS )
AC_MSG_NOTICE(>>> LDFLAGS = $LDFLAGS )
AC_MSG_NOTICE(>>> LIBS = $LIBS )
if test "x$use_nvcc" != xnone; then
	AC_MSG_NOTICE(>>> NVCC = $NVCC )
	AC_MSG_NOTICE(>>> NVCCFLAGS = $NVCCFLAGS )
fi
#try to give a full configure command
configure_cmd="./configure "
#MPI
if test "x$enable_mpi" != xno; then
	configure_cmd+="--enable-mpi "
	if test "x$with_mpicc" != xauto; then
		configure_cmd+="--with-mpicc=${with_mpicc} "
	fi
else
	configure_cmd+="--disable-mpi "
fi
#OMP
if test "x$use_omp" = xyes; then
	configure_cmd+="--enable-omp "
fi
#BLAS
if test "x$with_blas" != xno -a "x$with_blas" != xnone; then
	#determine which BLAS
	if test "x$with_blas" = xmkl -o "x$auto_blas" = xmkl; then
		configure_cmd+="--with-blas=mkl "
	fi
	if test "x$with_blas" = xopenblas -o "x$auto_blas" = xopenblas; then
		configure_cmd+="--with-blas=openblas "
	fi
	if test "x$with_blas" = xatlas -o "x$auto_blas" = xatlas; then
		configure_cmd+="--with-blas=atlas "
	fi
	if test "x$with_blas_include" != "xauto"; then
		configure_cmd+=" --with-blas-include=${with_blas_include}"
	fi
	if test "x$with_blas_lib" != "xauto"; then
		configure_cmd+=" --with-blas-lib=${with_blas_lib}"
	fi
fi
#CUDA
if test "x$use_nvcc" != xnone; then
	configure_cmd+="--enable-cuda "

	if test "x$with_cuda_include" != "xauto"; then
		configure_cmd+=" --with-cuda-include=${with_cuda_include}"
	fi
	if test "x$with_cuda_lib" != "xauto"; then
		configure_cmd+=" --with-cuda-lib=${with_cuda_lib}"
	fi
fi
AC_MSG_NOTICE(>>> configure COMMAND:)
AC_MSG_RESULT($configure_cmd)
AC_MSG_NOTICE(+++------------------+++)
AC_MSG_NOTICE(+++ ENDOF: configure +++)
AC_MSG_NOTICE(+++------------------+++)
