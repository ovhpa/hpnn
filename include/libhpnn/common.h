/*
+++ libhpnn - High Performance Neural Network library - file: common.h +++
    Copyright (C) 2019  Okadome Valencia Hubert

    This file is part of libhpnn.

    libhpnn is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    libhpnn is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef COMMON_H
#define COMMON_H
#ifdef USE_GLIB
#include <glib.h>
#include <glib/gstdio.h>
#else
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <dirent.h>
#endif //USE_GLIB
#include <math.h>
#ifdef _CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif
#ifdef _MPI
#include <mpi.h>
#endif

/* A header for the common useful C functions, and
 * most used defines.
 *
 * Linux users can import common.h as usual, there
 * should be no problem doing:
 * #include "common.h"
 * For portability, a version can also be found of
 * common.h defines using glib.h
 * In such cases, prior to importing common.h, the
 * users are required to #define USE_GLIB for pre-
 * compiler:
 * #define USE_GLIB
 * #include "common.h"
 *
 * ------------------- (c) OVHPA: Okadome Valencia
 * mail: hubert.valencia _at_ imass.nagoya-u.ac.jp */

#include "unroll.def"
/*defines*/
#if defined(__GNUC__) || (defined(__ICC) && (__ICC >= 600))
#define FUNCTION __PRETTY_FUNCTION__
#elif (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)) \
	|| (defined(__cplusplus) && (__cplusplus >= 201103))
#define FUNCTION __func__
#elif defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)
#define FUNCTION __FUNCTION__
#elif defined(__FUNCSIG__)
#define FUNCTION __FUNCSIG__
#else
#define FUNCTION "???"
#endif
#define PREP_READLINE() size_t _readline_len=0
#define READLINE(fp,buffer) do{\
	ssize_t _read_count=0;\
	_read_count=getline(&buffer,&_readline_len,fp);\
}while(0)
#define QUOTE(a) #a
#define QUOTE2(a,b) TH_QUOTE(a ## b)
#define TINY 1E-14
/*outputs*/
#ifdef _MPI
#define _OUT(_file,...) do{\
        int _rank;\
        MPI_Comm_rank(MPI_COMM_WORLD,&_rank);\
        if(_rank==0) fprintf((_file), __VA_ARGS__);\
}while(0)
#else /*_MPI*/
#define _OUT(_file,...) do{\
         fprintf((_file), __VA_ARGS__);\
}while(0)
#endif /*_MPI*/
/*USING GLIB?*/
#ifdef USE_GLIB
#define DIR_S GDir
#define CHAR gchar
#define UCHAR guchar
#define SHORT gshort
#define UINT guint
#define UINT64 guint64
#define DOUBLE gdouble
#define BOOL gboolean
#define STRFIND(a,b) g_strstr(b,a)
#define ISDIGIT g_ascii_isdigit
#define ISGRAPH g_ascii_isgraph
#define ISSPACE g_ascii_isspace
#define STR2ULL g_ascii_strtoull
#define STR2D g_ascii_strtod
#define ALLOC(pointer,size,type) do{\
	pointer=g_malloc0((size)*sizeof(type));\
	if(pointer==NULL) {\
		_OUT(stderr,"Alloc error (function %s, line %i)\n",FUNCTION,__LINE__);\
		exit(-1);\
	}\
}while(0)
#define FREE(pointer) do{\
	g_free(pointer);\
	pointer=NULL;\
}while(0)
#define STRDUP(src,dest) do{\
	dest=g_strdup(str);\
}while(0)
#define STRDUP_REPORT(src,dest,mem) do{\
	UINT __len=0;\
	STRLEN(src,__len);\
	STRDUP(src,dest);\
	mem+=__len*sizeof(CHAR);\
}while(0)
#define STRCAT(dest,src1,src2) do{\
	dest=g_strconcat(src1,src2,NULL);\
}while(0)
/*files*/
#define GET_CWD(cwd) do{\
	cwd=g_get_current_dir(void);\
}while(0)
#define OPEN_DIR(dir,path) do{\
	dir=g_dir_open(path,0,NULL);\
}while(0)
#define FILE_FROM_DIR(dir,file) do{\
	file=g_dir_read_name(dir);\
}while(0)
#define CLOSE_DIR(dir,ok) do{\
	g_dir_close(dir);\
	ok=0;\
}while(0)

#else /*USE_GLIB*/
#define DIR_S DIR
#define CHAR char
#define UCHAR unsigned char
#define SHORT short
#define UINT unsigned int
#define UINT64 uint64_t
#define DOUBLE double
#define BOOL int
#define STRFIND(a,b) strstr(b,a)
#define ISDIGIT(a) isdigit(a)
#define ISGRAPH(a) isgraph(a)
#define ISSPACE(a) isspace(a)
#define STR2ULL strtoull
#define STR2D strtod
#define ALLOC(pointer,size,type) do{\
	pointer=(type *)calloc((size),sizeof(type));\
	if(pointer==NULL) {\
		_OUT(stderr,"Alloc error (function %s, line %i)\n",FUNCTION,__LINE__);\
		exit(-1);\
	}\
}while(0)
#define FREE(pointer) do{\
	if(pointer!=NULL) free(pointer);\
	pointer=NULL;\
}while(0)
#define TRUE (1==1)
#define FALSE (1==0)
#define STRLEN(src,len) while((len<UINT_LEAST32_MAX)&&(src[len]!='\0')) len++
/*the len+1 in line 4 is a place for '\0' similarily to strcpy*/
#define STRDUP(src,dest) do{\
	UINT __len=0;\
	STRLEN(src,__len);\
	if(__len!=0){\
		ALLOC(dest,__len+1,CHAR);\
		__len--;\
		while(__len>0) {\
			dest[__len]=src[__len];\
			__len--;\
		}\
		dest[0]=src[0];\
	} else {\
		dest=NULL;\
	}\
}while(0)
#define STRDUP_REPORT(src,dest,mem) do{\
	UINT __len=0;\
	STRLEN(src,__len);\
	if(__len!=0){\
		ALLOC(dest,__len+1,CHAR);\
		mem+=__len*sizeof(CHAR);\
		__len--;\
		while(__len>0) {\
			dest[__len]=src[__len];\
			__len--;\
		}\
		dest[0]=src[0];\
	} else {\
		dest=NULL;\
	}\
}while(0)
#define STRCAT(dest,src1,src2) do{\
	UINT __acc=0,__len=0;\
	STRLEN(src1,__acc);\
	STRLEN(src2,__len);\
	__acc+=__len;\
	if((__acc>0)&&(__len>0)){\
		ALLOC(dest,__acc+1,CHAR);\
		dest[0]='\0';\
		strcat(dest,src1);\
		strcat(dest,src2);\
	} else {\
		dest=NULL;\
	}\
}while(0)


#define GET_CWD(cwd) do{\
	cwd=getcwd(NULL,0);\
}while(0)
#define OPEN_DIR(dir,path) do{\
        dir=opendir(path);\
}while(0)
#define FILE_FROM_DIR(dir,file) do{\
	struct dirent *_entry;\
	_entry=readdir(dir);\
	if(_entry==NULL) file=NULL;\
	else {\
		STRDUP(_entry->d_name,file);\
	}\
}while(0)
#define CLOSE_DIR(dir,ok) do{\
	ok=closedir(dir);\
}while(0)
#endif //USE_GLIB
/**/


/*report memory usage*/
#define ALLOC_REPORT(pointer,size,type,mem) do{\
	ALLOC(pointer,size,type);\
	mem+=size*sizeof(type);\
}while(0)
/*useful*/
#define SKIP_BLANK(pointer) \
	while((!ISGRAPH(*pointer))&&(*pointer!='\n')&&(*pointer!='\0')) pointer++
#define SKIP_NUM(pointer) \
	while((ISDIGIT(*pointer))&&(*pointer!='\n')&&(*pointer!='\0')) pointer++
#define STR_CLEAN(pointer) do{\
	CHAR *_ptr=pointer;\
	while(*_ptr!='\0'){\
		if(*_ptr=='\t') *_ptr='\0';\
		if(*_ptr==' ') *_ptr='\0';\
		if((*_ptr=='\n')||(*_ptr=='#')) *_ptr='\0';\
		else _ptr++;\
	}\
}while(0)
#define GET_LAST_LINE(fp,buffer) do{\
	fseek(fp,-2,SEEK_END);\
	while(fgetc(fp)!='\n') fseek(fp,-2,SEEK_CUR);\
	fseek(fp,+1,SEEK_CUR);\
	READLINE(fp,buffer);\
}while(0)
#define GET_UINT(i,in,out) do{\
	i=(UINT)STR2ULL(in,&(out),10);\
}while(0)
#define GET_DOUBLE(d,in,out) do{\
	d=(DOUBLE)STR2D(in,&(out));\
}while(0)
#define ARRAY_CP(src,dest,n) do{\
	if(src!=NULL){\
		UINT _i;\
		for(_i=0;_i<(n);_i++) dest[_i]=src[_i];\
	}\
}while(0)

#define ASSERTPTR(pointer,retval) do{\
	if(pointer==NULL){\
		_OUT(stderr,"Error: NULL pointer (function %s, line %i):\n%s=NULL\n",\
			FUNCTION,__LINE__,QUOTE(pointer));\
		return retval;\
	}\
}while(0)

#define ASSERT_GOTO(pointer,label) do{\
	if(pointer==NULL){\
		_OUT(stderr,"Error: NULL pointer (function %s, line %i):\n%s=NULL\n",\
			FUNCTION,__LINE__,QUOTE(pointer));\
		goto label;\
	}\
}while(0)

/*CUDA*/
#ifdef _CUDA
/*ERROR*/
#define CUBLAS_ERR_CASE(err) case err: _OUT(stderr,\
	"CUBLAS ERROR: %s\t(function %s, line %i)\n",QUOTE(err),FUNCTION,__LINE__);\
	break
#define CUBLAS_ERR(err) do {\
	if(err != CUBLAS_STATUS_SUCCESS) {\
		switch(err) {\
		CUBLAS_ERR_CASE(CUBLAS_STATUS_NOT_INITIALIZED);\
		CUBLAS_ERR_CASE(CUBLAS_STATUS_ALLOC_FAILED);\
		CUBLAS_ERR_CASE(CUBLAS_STATUS_INVALID_VALUE);\
		CUBLAS_ERR_CASE(CUBLAS_STATUS_ARCH_MISMATCH);\
		CUBLAS_ERR_CASE(CUBLAS_STATUS_MAPPING_ERROR);\
		CUBLAS_ERR_CASE(CUBLAS_STATUS_EXECUTION_FAILED);\
		CUBLAS_ERR_CASE(CUBLAS_STATUS_INTERNAL_ERROR);\
		CUBLAS_ERR_CASE(CUBLAS_STATUS_NOT_SUPPORTED);\
		CUBLAS_ERR_CASE(CUBLAS_STATUS_LICENSE_ERROR);\
		default:\
			_OUT(stderr,"CUBLAS ??? ERROR!\t(val= %i, function %s, line %i)\n",\
				err,FUNCTION,__LINE__);\
		}\
		exit(-1);\
	}\
}while(0)
#define _Q(a) #a
#ifdef   DEBUG
#define CHK_ERR(func) do{\
	cudaError_t _itmp=cudaGetLastError();\
	if(_itmp!=cudaSuccess){\
		_OUT(stderr,"CUDA ERROR %i: %s in function %s!\n",_itmp,\
			cudaGetErrorString(_itmp),_Q(func));\
		exit(1);\
	}\
}while(0)
#else  /*DEBUG*/
#define CHK_ERR(func)
#endif /*DEBUG*/
/*allocations*/
#define CUDA_ALLOC(pointer,size,type) do{\
	cudaError_t _err;\
	_err=cudaMalloc((void **)(&pointer),size*sizeof(type));\
	if(_err!=cudaSuccess) {\
		_OUT(stderr,"CUDA alloc error (function %s, line %i)\n",\
			FUNCTION,__LINE__);\
		_OUT(stderr,"CUDA report: %s\n",cudaGetErrorString(_err));\
		exit(-1);\
	}\
	_err=cudaMemset((void *)pointer,0,size*sizeof(type));\
	if(_err!=cudaSuccess) {\
		_OUT(stderr,"CUDA memset error (function %s, line %i)\n",\
			FUNCTION,__LINE__);\
		_OUT(stderr,"CUDA report: %s\n",cudaGetErrorString(_err));\
		exit(-1);\
	}\
}while(0)
#define CUDA_ALLOC_REPORT(pointer,size,type,mem) do{\
	CUDA_ALLOC(pointer,size,type);\
	mem+=size*sizeof(type);\
}while(0)
#define CUDA_FREE(pointer) do{\
	if(pointer!=NULL) cudaFree(pointer);\
	pointer=NULL;\
}while(0)
#define CUDA_RAZ(pointer,size,type) do{\
	cudaError_t _err;\
	_err=cudaMemset((void *)pointer,0,size*sizeof(type));\
	if(_err!=cudaSuccess) {\
		_OUT(stderr,"CUDA memset error (function %s, line %i)\n",\
			 FUNCTION,__LINE__);\
		exit(-1);\
	}\
}while(0)
/*managed memory*/
#define CUDA_ALLOC_MM(pointer,size,type) do{\
	cudaError_t _err;\
	_err=cudaMallocManaged((void **)(&pointer),\
		size*sizeof(type),cudaMemAttachGlobal);\
	if(_err!=cudaSuccess) {\
		_OUT(stderr,"CUDA alloc_MM error (function %s, line %i)\n",\
			FUNCTION,__LINE__);\
		exit(-1);\
	}\
	_err=cudaMemset((void *)pointer,0,size*sizeof(type));\
	if(_err!=cudaSuccess) {\
		_OUT(stderr,"CUDA memset error (function %s, line %i)\n",\
			FUNCTION,__LINE__);\
		exit(-1);\
	}\
}while(0)
#define CUDA_ALLOC_MM_REPORT(pointer,size,type,mem) do{\
	CUDA_ALLOC_MM(pointer,size,type);\
	mem+=size*sizeof(type);\
}while(0)

#define CUDA_LAST_ERROR(num) do{\
	cudaError_t _err=cudaGetLastError();\
	_OUT(stderr,"CUDA step %i report: %s\n",num,cudaGetErrorString(_err));\
}while(0)

/*sync*/
#define CUDA_C2G_CP(cpu,gpu,size,type) do{\
	cudaMemcpy(gpu,cpu,size*sizeof(type),cudaMemcpyHostToDevice);\
}while(0)
#define CUDA_G2C_CP(cpu,gpu,size,type) do{\
	cudaMemcpy(cpu,gpu,size*sizeof(type),cudaMemcpyDeviceToHost);\
}while(0)
#define CUBLAS_SET_VECTOR(cpu_v,ldc,gpu_v,ldg,size,type) do{\
	cublasStatus_t _err;\
	_err=cublasSetVector(size,sizeof(type),cpu_v,ldc,gpu_v,ldg);\
	if(_err != CUBLAS_STATUS_SUCCESS){\
		_OUT(stderr,"CPU to GPU transfer error (function %s, line %i)\n",\
			FUNCTION,__LINE__);\
		exit(-1);\
	}\
}while(0)
#define CUBLAS_GET_VECTOR(cpu_v,ldc,gpu_v,ldg,size,type) do{\
	cublasStatus_t _err;\
	_err=cublasGetVector(size,sizeof(type),gpu_v,ldg,cpu_v,ldc);\
	if(_err != CUBLAS_STATUS_SUCCESS){\
		_OUT(stderr,"GPU to CPU transfer error (function %s, line %i)\n",\
			FUNCTION,__LINE__);\
		exit(-1);\
	}\
}while(0)
/*a lot harder due to painful column-major of CUBLAS*/
/*FIXME: these function are deprecated in libhpnn... remove?*/
#define CUBLAS_SET_MATRIX(cpu_m,gpu_m,cpu_row,cpu_col,type) do{\
	cublasStatus_t _err;\
	_err=cublasSetMatrix(cpu_col,cpu_row,sizeof(type),\
		cpu_m,cpu_col,gpu_m,cpu_col);\
	CUBLAS_ERR(_err);\
}while(0)
#define CUBLAS_GET_MATRIX(cpu_m,gpu_m,cpu_row,cpu_col,type) do{\
	cublasStatus_t _err;\
	_err=cublasGetMatrix(cpu_col,cpu_row,sizeof(type),\
		gpu_m,cpu_col,cpu_m,cpu_col);\
	if(_err != CUBLAS_STATUS_SUCCESS){\
		_OUT(stderr,"GPU to CPU transfer error (function %s, line %i)\n",\
			FUNCTION,__LINE__);\
		exit(-1);\
	}\
}while(0)
/*COPY*/
#define CUDA_G2G_CP(src,dest,size,type) do{\
	cudaError_t _err;\
	_err=cudaMemcpy(dest,src,size*sizeof(type),cudaMemcpyDeviceToDevice);\
	if(_err!=cudaSuccess) {\
		_OUT(stderr,"GPU to GPU transfer error (function %s, line %i)\n",\
			FUNCTION,__LINE__);\
		exit(-1);\
	}\
}while(0)
#endif /*_CUDA*/

typedef enum {
	CUDA_MEM_NONE, 	/*no need for model only 1 GPU*/
	CUDA_MEM_EXP, 	/*explicit copy on each device*/
	CUDA_MEM_P2P, 	/*all devices can peer to peer*/
	CUDA_MEM_CMM, 	/*device can manage memory concurrently*/
} cudas_mem;

typedef struct {
	UINT n_gpu;
#ifdef _CUBLAS
	cublasHandle_t *cuda_handle;
#else /*_CUBLAS*/
	int *cuda_handle;
#endif /*_CUBLAS*/
	UINT        cuda_n_streams;
#ifdef _CUDA
	cudaStream_t *cuda_streams;
#else /*_CUDA*/
	void *cuda_streams;
#endif /*_CUDA*/
	cudas_mem mem_model;
} cudastreams;



#endif//COMMON_H
