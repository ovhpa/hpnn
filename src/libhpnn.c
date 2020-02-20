/*
+++ libhpnn - High Performance Neural Network library - file: libhpnn.c +++
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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
/* Artificial Neuron Network abstract layer, interfaces with the HPNN library */
/* -------------------------------------------- Hubert Okadome Valencia, 2019 */
/*^^^ MPI specific*/
#ifdef _MPI
#include <mpi.h>
#endif
/*^^^ BLAS / MKL specific*/
#if defined (PBLAS) || defined (SBLAS)
#ifndef _MKL
#include <cblas.h>
#else /*_MKL*/
#include <mkl.h>
#include <mkl_cblas.h>
#endif /*_MKL*/
#endif /*PBLAS*/
/*^^^ CUDA specific*/
#ifdef _CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif /*_CUDA*/
/*^^^ OMP specific*/
#ifdef _OMP
#include <omp.h>
#endif
/*^^^main header*/
#include <libhpnn.h>
#include <libhpnn/ann.h>
#include <libhpnn/snn.h>
/*GLOBAL VARIABLE: there it a unique runtime per run
 *  for which each use of library routine refers to.*/
nn_runtime lib_runtime;
#define TOTAL_S (lib_runtime.cudas.cuda_n_streams*lib_runtime.cudas.n_gpu)
/*------------------*/
/*+++ NN methods +++*/
/*------------------*/
/*^^^ depending on the host program, MPI/OMP/CUDA/BLAS initialization can be un-
 * necessary. However, proper initialization _has to be done_!        -- OVHPA*/
/*--------------------------*/
/*+++ initialize library +++*/
/*--------------------------*/
void _NN(inc,verbose)(){
	if(lib_runtime.nn_verbose>2) return;
        lib_runtime.nn_verbose++;
	NN_DBG(stdout,"verbosity set to %i.\n",lib_runtime.nn_verbose);
}
void _NN(dec,verbose)(){
	if(lib_runtime.nn_verbose<1) return;
	lib_runtime.nn_verbose--;
}
void _NN(set,verbose)(SHORT verbosity){
	lib_runtime.nn_verbose=verbosity;
}
void _NN(get,verbose)(SHORT *verbosity){
	*verbosity=lib_runtime.nn_verbose;
}
int _NN(return,verbose)(){
	return lib_runtime.nn_verbose;
}
void _NN(toggle,dry)(){
        lib_runtime.nn_dry^=lib_runtime.nn_dry;
}
nn_cap _NN(get,capabilities)(){
	UINT res=0;
#ifdef _OMP
	res+=(1<<0);
#endif
#ifdef _MPI
	res+=(1<<1);
#endif
#ifdef _CUDA
	res+=(1<<2);
#endif
#ifdef _CUBLAS
	res+=(1<<3);
#endif
	/*(1<<4) is reserved for OCL*/
#ifdef PBLAS
	res+=(1<<5);
#elif defined(SBLAS)
	res+=(1<<6);
#endif
	return (nn_cap)res;
}
void _NN(unset,capability)(nn_cap capability){
	switch (capability){
	case NN_CAP_OMP:
		lib_runtime.capability &= ~(1<<0);
		break;
	case NN_CAP_MPI:
		lib_runtime.capability &= ~(1<<1);
		break;
	case NN_CAP_CUDA:
		lib_runtime.capability &= ~(1<<2);
		break;
	case NN_CAP_CUBLAS:
		lib_runtime.capability &= ~(1<<3);
		break;
	case NN_CAP_PBLAS:
		lib_runtime.capability &= ~(1<<5);
		break;
	case NN_CAP_SBLAS:
		lib_runtime.capability &= ~(1<<6);
		break;
	case NN_CAP_NONE:
	default:
		return;
	}
}
void _NN(init,runtime)(){
	lib_runtime.capability=_NN(get,capabilities)();
	lib_runtime.nn_verbose=0;
	lib_runtime.nn_dry=FALSE;
	lib_runtime.nn_num_threads=1;
	lib_runtime.nn_num_blas =  1;
	lib_runtime.nn_num_tasks = 1;
	lib_runtime.cudas.n_gpu =  1;
	lib_runtime.cudas.cuda_handle =NULL;
	lib_runtime.cudas.cuda_n_streams =1;
	lib_runtime.cudas.cuda_streams=NULL;
	lib_runtime.cudas.mem_model=CUDA_MEM_NONE;
}
BOOL _NN(init,OMP)(){
#ifndef _OMP
	NN_WARN(stdout,"failed to init OMP (no capability).\n");
	return FALSE;
#else
	NN_OUT(stdout,"NN: OMP init done.\n");
	return TRUE;
#endif
}
BOOL _NN(init,MPI)(){
#ifndef _MPI
	NN_WARN(stdout,"failed to init MPI (no capability).\n");
	return FALSE;
#else /*_MPI*/
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD,&(lib_runtime.nn_num_tasks));
	if(lib_runtime.nn_num_tasks<2) {
		NN_WARN(stdout,"#WARNING: libhpnn was compiled with MPI,\n");
		NN_WARN(stdout,"but only one task is used, which may not\n");
		NN_WARN(stdout,"be what you intended and is inefficient.\n");
		NN_WARN(stdout,"Please switch to serial version of hpnn,\n");
		NN_WARN(stdout,"or use several parallel tasks with -np X\n");
		NN_WARN(stdout,"option of mpirun.               -- OVHPA\n");
	}
	NN_OUT(stdout,"MPI started %i tasks.\n",lib_runtime.nn_num_tasks);
	return TRUE;
#endif /*_MPI*/
}
BOOL _NN(init,CUDA)(){
#ifndef _CUDA
	NN_WARN(stdout,"failed to init CUDA (no capability).\n");
	return FALSE;
#else /*_CUDA*/
	int is_ok,gpu;
	cudaGetDeviceCount(&(lib_runtime.cudas.n_gpu));
	CHK_ERR(init_device_count);
	if(lib_runtime.cudas.n_gpu<1) {
		NN_WARN(stderr,"CUDA error: no CUDA-capable device reported.\n");
		return FALSE;
	}
	NN_OUT(stdout,"CUDA started, found %i GPU(s).\n",lib_runtime.cudas.n_gpu);
#ifdef _CUBLAS
	cublasStatus_t err;
	ALLOC(lib_runtime.cudas.cuda_handle,lib_runtime.cudas.n_gpu,cublasHandle_t);
	for(gpu=0;gpu<lib_runtime.cudas.n_gpu;gpu++){
		cudaSetDevice(gpu);
		err=cublasCreate(&(lib_runtime.cudas.cuda_handle[gpu]));
		if(err!=CUBLAS_STATUS_SUCCESS){
			NN_ERROR(stderr,
					 "CUDA: can't create CUBLAS context for GPU[%i].\n",gpu);
			/*this is bad, and we should fail*/
			return FALSE;
		}
		err=cublasSetPointerMode(lib_runtime.cudas.cuda_handle[gpu],
								 CUBLAS_POINTER_MODE_HOST);
		if(err!=CUBLAS_STATUS_SUCCESS){
			NN_WARN(stderr,
					"CUBLAS: fail to set pointer mode for GPU[%i].\n",gpu);
			/*this is probably bad*/
		}
	}
#else /*_CUBLAS*/
	ALLOC(lib_runtime.cudas.cuda_handle,lib_runtime.cudas.n_gpu,int);
	for(gpu=0;gpu<lib_runtime.cudas.n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaGetDevice(&(lib_runtime.cudas.cuda_handle[gpu]));
		CHK_ERR(init_device_handle);
	}
#endif /*_CUBLAS*/
//	lib_runtime.cudas.mem_model=CUDA_MEM_CMM;// TEST CMM
	/*deal with multi-GPU*/
	if(lib_runtime.cudas.n_gpu>1){
		BOOL test_mem;
		/*determine the memory model*/
		/*try p2p*/
		test_mem=TRUE;
		for(gpu=0;gpu<lib_runtime.cudas.n_gpu;gpu++){
			cudaSetDevice(gpu);
			cudaDeviceCanAccessPeer(&is_ok,gpu,0);
			CHK_ERR(chk_peer_access);
			test_mem&=(is_ok==1);
		}
		if(test_mem==TRUE) lib_runtime.cudas.mem_model=CUDA_MEM_P2P;
		else{
			/*try cmm*/
			test_mem=TRUE;
			for(gpu=0;gpu<lib_runtime.cudas.n_gpu;gpu++){
				/*check for managed memory support*/
				cudaDeviceGetAttribute(&is_ok,cudaDevAttrManagedMemory,gpu);
				CHK_ERR(chk_cmm_access);
				test_mem&=(is_ok==1);
				/*check for concurrency in mm*/
				cudaDeviceGetAttribute(&is_ok,
									cudaDevAttrConcurrentManagedAccess,gpu);
				CHK_ERR(chk_cmm_access);
				test_mem&=(is_ok==1);
			}
			if(test_mem=TRUE) lib_runtime.cudas.mem_model=CUDA_MEM_CMM;
			else{
				lib_runtime.cudas.mem_model=CUDA_MEM_EXP;
			}
		}
	}
	switch(lib_runtime.cudas.mem_model){
		case CUDA_MEM_P2P:
			for(gpu=1;gpu<lib_runtime.cudas.n_gpu;gpu++){
				cudaSetDevice(gpu);
				cudaDeviceEnablePeerAccess(0,0);
				CHK_ERR(enable_peer_access);
			}
			NN_DBG(stdout,"multi-GPU will use peer access from GPU[0]\n");
			break;
		case CUDA_MEM_CMM:
			NN_DBG(stdout,"multi-GPU will use managed memory\n");
			break;
		case CUDA_MEM_EXP:
			NN_DBG(stdout,"multi-GPU using explicit BCAST.\n");
			break;
		case CUDA_MEM_NONE:
			break;
		default:
			/*should not happen*/
			NN_ERROR(stderr,"CUDA: unknown memory model!\n");
			return FALSE;
	}
#endif /*_CUDA*/
}
BOOL _NN(init,BLAS)(){
#if !defined (PBLAS) && !defined (SBLAS)
	NN_WARN(stdout,"NOT USING BLAS.\n");
	return FALSE;
#else /*PBLAS or SBLAS*/
#ifdef _MKL
        mkl_set_dynamic(0);
        lib_runtime.nn_num_blas = mkl_domain_get_max_threads(MKL_DOMAIN_BLAS);
#endif /*_MKL*/
#ifdef _OPENBLAS
        lib_runtime.nn_num_blas = openblas_get_num_threads();
#endif /*_OPENBLAS*/
#ifdef PBLAS
	NN_DBG(stdout,"USING PBLAS.\n");
#else /*PBLAS*/
	NN_DBG(stdout,"USING SBLAS.\n");
#endif /*PBLAS*/
	return TRUE;
#endif /*PBLAS or SBLAS*/
}
int _NN(init,all)(UINT init_verbose){
	BOOL is_ok=FALSE;
	nn_cap capability;
	_NN(init,runtime)();
lib_runtime.nn_verbose=init_verbose;
	capability = lib_runtime.capability;
	if(capability & NN_CAP_MPI) {
		is_ok|=_NN(init,MPI)();
	}
	if(capability & NN_CAP_OMP) {
		is_ok|=_NN(init,OMP)();
	}
	if(capability & NN_CAP_CUDA) {
		is_ok|=_NN(init,CUDA)();
	}
	if((capability & NN_CAP_PBLAS)||(capability & NN_CAP_SBLAS)){
		is_ok|=_NN(init,BLAS)();
	}
lib_runtime.nn_verbose=0;
	if(is_ok) return 0;
	else return -1;
}
BOOL _NN(deinit,OMP)(){
#ifndef _OMP
	return FALSE;
#else
	/*nothing to do (for now)*/
	return TRUE;
#endif
}
BOOL _NN(deinit,MPI)(){
#ifndef _MPI
	return FALSE;
#else
	/*this should be done last*/
	MPI_Finalize();
	return TRUE;
#endif	
}
BOOL _NN(deinit,CUDA)(){
#ifndef _CUDA
	return FALSE;
#else
	UINT idx,gpu;
	for(idx=0;idx<TOTAL_S;idx++){
		gpu=idx/lib_runtime.cudas.cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cudaStreamDestroy(lib_runtime.cudas.cuda_streams[idx]);
	}
	FREE(lib_runtime.cudas.cuda_streams);
#ifdef _CUBLAS
	for(gpu=0;gpu<lib_runtime.cudas.n_gpu;gpu++) {
		cudaSetDevice(gpu);
		cublasDestroy(lib_runtime.cudas.cuda_handle[gpu]);
	}
#endif
	FREE(lib_runtime.cudas.cuda_handle);
	cudaDeviceReset();/*this also kills peer memory ability!*/
	return TRUE;
#endif
}
BOOL _NN(deinit,BLAS)(){
#if !defined (PBLAS) && !defined (SBLAS)
	return FALSE;
#else
	/*nothing to do (for now)*/
	return TRUE;
#endif
}
int _NN(deinit,all)(){
	BOOL is_ok=FALSE;
	nn_cap capability = _NN(get,capabilities)();
	if(capability & NN_CAP_OMP) is_ok|=_NN(deinit,OMP)();
	if(capability & NN_CAP_MPI) is_ok|=_NN(deinit,MPI)();
	if(capability & NN_CAP_CUDA) is_ok|=_NN(deinit,CUDA)();
	if((capability & NN_CAP_PBLAS)||(capability & NN_CAP_SBLAS)) 
		is_ok|=_NN(deinit,BLAS)();
	if(is_ok) return 0;
	else return -1;
}
/*------------------------------*/
/*+++ set/get lib parameters +++*/
/*------------------------------*/
BOOL _NN(set,omp_threads)(UINT n_threads){
#ifndef _OMP
	NN_WARN(stdout,"failed to set OMP num_threads (no capability).\n");
	return FALSE;
#else
	lib_runtime.nn_num_threads = n_threads;
	omp_set_num_threads(lib_runtime.nn_num_threads);
	return TRUE;
#endif /*_OMP*/
}
BOOL _NN(get,omp_threads)(UINT *n_threads){
#ifndef _OMP
	*n_threads = 1;
	return FALSE;
#else
	*n_threads = lib_runtime.nn_num_threads;
	return TRUE;
#endif
}
int _NN(return,omp_threads)(){
	return lib_runtime.nn_num_threads;
}
BOOL _NN(set,mpi_tasks)(UINT n_tasks){
#ifndef _MPI
	NN_WARN(stdout,"failed to set MPI num_tasks (no capability).\n");
	return FALSE;
#else
	NN_WARN(stdout,"Changing MPI num_tasks is not implemented yet.\n");
	/*however the possibility is left opened for future implementations*/
	return TRUE;
#endif
}
BOOL _NN(get,mpi_tasks)(UINT *n_tasks){
#ifndef _MPI
	*n_tasks=1;
	return FALSE;
#else
	*n_tasks=lib_runtime.nn_num_tasks;
	return TRUE;
#endif
}
BOOL _NN(get,curr_mpi_task)(UINT *task){
#ifndef _MPI
	return FALSE;
#else
	MPI_Comm_rank(MPI_COMM_WORLD,task);
	return TRUE;
#endif
}
BOOL _NN(set,n_gpu)(UINT n_gpu){
	NN_WARN(stdout,"Changing the number of GPU is not implemented yet.\n");
	return FALSE;
}
BOOL _NN(get,n_gpu)(UINT *n_gpu){
#ifndef  _CUDA
	n_gpu=0;
	return FALSE;
#else  /*_CUDA*/
	*n_gpu=lib_runtime.cudas.n_gpu;
#endif /*_CUDA*/
}
BOOL _NN(set,cuda_streams)(UINT n_streams){
#ifndef _CUDA
	return FALSE;
#else
	UINT idx,gpu;
	/*setting new cuda_streams should reset the cuda_streams*/
	/*only if cuda_streams was initialized properly before..*/
	if(lib_runtime.cudas.cuda_streams!=NULL){
		/*first we need to wipe previous streams*/
		for(idx=0;idx<TOTAL_S;idx++){
			gpu=idx/lib_runtime.cudas.cuda_n_streams;/*gpu number*/
			cudaSetDevice(gpu);
			cudaStreamDestroy(lib_runtime.cudas.cuda_streams[idx]);
		}
		FREE(lib_runtime.cudas.cuda_streams);
	}
	if(n_streams<2) lib_runtime.cudas.cuda_n_streams=1;
	else lib_runtime.cudas.cuda_n_streams=n_streams;
	ALLOC(lib_runtime.cudas.cuda_streams,
		  TOTAL_S*sizeof(cudaStream_t),cudaStream_t);
	for(idx=0;idx<TOTAL_S;idx++){
		gpu=idx/lib_runtime.cudas.cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cudaStreamCreateWithFlags(&(lib_runtime.cudas.cuda_streams[idx]),
								  cudaStreamNonBlocking);
	}
#ifdef _CUBLAS
	/*this step is optional, but it seems that CUBLAS prefers
	 *to start on its own first stream (on first GPU)...*/
        cublasSetStream(lib_runtime.cudas.cuda_handle[0],
						lib_runtime.cudas.cuda_streams[0]);
#endif /*_CUBLAS*/
	return TRUE;
#endif
}
BOOL _NN(get,cuda_streams)(UINT *n_streams){
#ifndef _CUDA
	return FALSE;
#else
	*n_streams = lib_runtime.cudas.cuda_n_streams;
	return TRUE;
#endif
}
BOOL _NN(set,omp_blas)(UINT n_blas){
#if !defined (PBLAS) && !defined (SBLAS)
	return FALSE;
#else
	lib_runtime.nn_num_blas = n_blas;
/*note that atlas BLAS can't change the number of threads*/
#ifdef _MKL
	mkl_domain_set_num_threads(lib_runtime.nn_num_blas, MKL_DOMAIN_BLAS);
#endif /*_MKL*/
#ifdef _OPENBLAS
	openblas_set_num_threads(lib_runtime.nn_num_blas);
#endif
	return TRUE;
#endif
}
BOOL _NN(get,omp_blas)(UINT *n_blas){
#if !defined (PBLAS) && !defined (SBLAS)
	return FALSE;
#else
	*n_blas=lib_runtime.nn_num_blas;
	return TRUE;
#endif
}
cudastreams *_NN(get,cudas)(){
	return &(lib_runtime.cudas);
}
/*---------------------*/
/*+++ configuration +++*/
/*---------------------*/
#define _CONF (*conf)
void _NN(init,conf)(nn_def *conf){
	/*init default conf*/
	_CONF.rr=&lib_runtime;
	_CONF.name=NULL;
	_CONF.type=NN_TYPE_UKN;
	_CONF.need_init=FALSE;
	_CONF.seed=0;
	_CONF.kernel=NULL;
	_CONF.f_kernel=NULL;
	_CONF.train=NN_TRAIN_UKN;
	_CONF.samples=NULL;
	_CONF.tests=NULL;
}
void _NN(deinit,conf)(nn_def *conf){
	if(_CONF.kernel!=NULL) _NN(free,kernel)(conf);
	FREE(_CONF.kernel);
	_CONF.rr=NULL;/*detach runtime*/
	FREE(_CONF.name);
	_CONF.type=NN_TYPE_UKN;
	_CONF.need_init=FALSE;
	_CONF.seed=0;
	FREE(_CONF.f_kernel);
	_CONF.train=NN_TRAIN_UKN;
	FREE(_CONF.samples);
	FREE(_CONF.tests);
}
void _NN(set,name)(nn_def *conf,const CHAR *name){
	FREE(_CONF.name);
	STRDUP(name,_CONF.name);
}
void _NN(get,name)(nn_def *conf,CHAR **name){
	/*will initialize and return name*/
	/*USER need to free name! --OVHPA*/
//	FREE(*name);
	STRDUP(_CONF.name,*name);
}
char *_NN(return,name)(nn_def *conf){
	return _CONF.name;
}
void _NN(set,type)(nn_def *conf,nn_type type){
	_CONF.type=type;
}
void _NN(get,type)(nn_def *conf,nn_type *type){
	*type=_CONF.type;
}
nn_type _NN(return,type)(nn_def *conf){
	return _CONF.type;
}
void _NN(set,need_init)(nn_def *conf,BOOL need_init){
	_CONF.need_init=need_init;
}
void _NN(get,need_init)(nn_def *conf,BOOL *need_init){
	*need_init=_CONF.need_init;
}
BOOL _NN(return,need_init)(nn_def *conf){
	return _CONF.need_init;
}
void _NN(set,seed)(nn_def *conf,UINT seed){
	_CONF.seed=seed;
}
void _NN(get,seed)(nn_def *conf,UINT *seed){
	*seed=_CONF.seed;
}
UINT _NN(return,seed)(nn_def *conf){
	return _CONF.seed;
}
void _NN(set,kernel_filename)(nn_def *conf,CHAR *f_kernel){
        FREE(_CONF.f_kernel);
        STRDUP(f_kernel,_CONF.f_kernel);
}
void _NN(get,kernel_filename)(nn_def *conf,CHAR **f_kernel){
	/*will initialize and return f_kernel*/
	/*USER need to free f_kernel! --OVHPA*/
//	FREE(*f_kernel);
	STRDUP(_CONF.f_kernel,*f_kernel);
}
char *_NN(return,kernel_filename)(nn_def *conf){
	return _CONF.f_kernel;
}
void _NN(set,train)(nn_def *conf,nn_train train){
	_CONF.train=train;
}
void _NN(get,train)(nn_def *conf,nn_train *train){
	*train=_CONF.train;
}
nn_train _NN(return,train)(nn_def *conf){
	return _CONF.train;
}
void _NN(set,samples_directory)(nn_def *conf,CHAR *samples){
        FREE(_CONF.samples);
        STRDUP(samples,_CONF.samples);
}
void _NN(get,samples_directory)(nn_def *conf,CHAR **samples){
	/*will initialize and return samples*/
	/*USER need to free samples! --OVHPA*/
//	FREE(*samples);
	STRDUP(_CONF.samples,*samples);
}
char *_NN(return,samples_directory)(nn_def *conf){
	return _CONF.samples;
}
void _NN(set,tests_directory)(nn_def *conf,CHAR *tests){
        FREE(_CONF.tests);
        STRDUP(tests,_CONF.tests);
}
void _NN(get,tests_directory)(nn_def *conf,CHAR **tests){
	/*will initialize and return tests*/
	/*USER need to free tests! --OVHPA*/
//	FREE(*tests);
	STRDUP(_CONF.tests,*tests);
}
char *_NN(return,tests_directory)(nn_def *conf){
	return _CONF.tests;
}
nn_def *_NN(load,conf)(const CHAR *filename){
#define FAIL read_conf_fail
	PREP_READLINE();
	CHAR *line=NULL;
	CHAR *ptr,*ptr2;
	UINT *parameter;
	UINT *n_hiddens;
	UINT64 allocate;
	nn_def  *conf;
	BOOL is_ok;
	UINT   idx;
	FILE   *fp;
	/*init*/
	allocate=0;
	n_hiddens=NULL;
	ALLOC_REPORT(conf,1,nn_def,allocate);
	_NN(init,conf)(conf);
	ALLOC(parameter,3,UINT);
	/**/
	fp=fopen(filename,"r");
	if(!fp){
		NN_ERROR(stderr,"Error opening configuration file: %s\n",filename);
		goto FAIL;
	}
	READLINE(fp,line);/*first line is usually a comment*/
	do{
		ptr=STRFIND("[name",line);
		if(ptr!=NULL){
			/*get name {any string}*/
			ptr+=6;SKIP_BLANK(ptr);
			/*process line (remove comment and \n)*/
			STR_CLEAN(ptr);
			STRDUP_REPORT(ptr,_CONF.name,allocate);
		}
		ptr=STRFIND("[type",line);
		if(ptr!=NULL){
			/*get type {"ANN", "SNN", ...}*/
			ptr+=6;SKIP_BLANK(ptr);
			switch (*ptr){
			case 'L':
				_CONF.type=NN_TYPE_LNN;
				break;
			case 'S':
				_CONF.type=NN_TYPE_SNN;
				break;
			case 'A':
				/*fallthrough*/
			default:
				_CONF.type=NN_TYPE_ANN;
				break;
			}
		}
		ptr=STRFIND("[init",line);
		if(ptr!=NULL){
			/*get init {"generate", "file"}*/
			ptr+=6;SKIP_BLANK(ptr);
			if((STRFIND("generate",line)!=NULL)
			 ||(STRFIND("GENERATE",line)!=NULL)){
NN_OUT(stdout,"generating kernel!\n");
				_CONF.need_init=TRUE;
			}else{
NN_OUT(stdout,"loading kernel!\n");
				_CONF.need_init=FALSE;
				STR_CLEAN(ptr);
				STRDUP_REPORT(ptr,_CONF.f_kernel,allocate);
				if(_CONF.f_kernel==NULL){
					NN_ERROR(stderr,"Malformed NN configuration file!\n");
					NN_ERROR(stderr,"[init] can't read filename: %s\n",ptr);
					goto FAIL;
				}
			}
		}
		ptr=STRFIND("[seed",line);
		if(ptr!=NULL){
			ptr+=6;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				NN_ERROR(stderr,"Malformed NN configuration file!\n");
				NN_ERROR(stderr,"[seed] value: %s\n",ptr);
				goto FAIL;
			}
			GET_UINT(_CONF.seed,ptr,ptr2);
		}
		ptr=STRFIND("[input",line);
		if(ptr!=NULL){
			/*get number of inputs {integer}*/
			ptr+=7;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				NN_ERROR(stderr,"Malformed NN configuration file!\n");
				NN_ERROR(stderr,"[input] value: %s\n",ptr);
				goto FAIL;
			}
			GET_UINT(parameter[0],ptr,ptr2);
			/*can be 0 or missing (it then will be set while loading kernel)*/
		}
		ptr=STRFIND("[hidden",line);
		if(ptr!=NULL){
			/*get number of hidden neurons {integer x n_layers}*/
			ptr+=8;SKIP_BLANK(ptr);
			/*count the number of integers -> n_hiddens*/
			if(!ISDIGIT(*ptr)) {
				NN_ERROR(stderr,"Malformed NN configuration file!\n");
				NN_ERROR(stderr,"[hidden] value: %s\n",ptr);
				goto FAIL;
			}
			parameter[1]=1;ptr2=ptr;
			while(ISGRAPH(*ptr)){
				while(ISDIGIT(*ptr)) ptr++;
				SKIP_BLANK(ptr);
				if(ISDIGIT(*ptr)) parameter[1]++;
				else break;
			}
			/*now get each hidden layer n_neurons*/
			ALLOC_REPORT(n_hiddens,parameter[1],UINT,allocate);
			ptr=ptr2;
			for(idx=0;idx<parameter[1];idx++){
				GET_DOUBLE(n_hiddens[idx],ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
				ptr=ptr2+1;SKIP_BLANK(ptr);
			}
		}
		ptr=STRFIND("[output",line);
		if(ptr!=NULL){
			/*get the number of output {integer}*/
			ptr+=8;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				NN_ERROR(stderr,"Malformed NN configuration file!\n");
				NN_ERROR(stderr,"[output] value: %s\n",ptr);
				goto FAIL;
			}
			GET_UINT(parameter[2],ptr,ptr2);
		}
		ptr=STRFIND("[train",line);
		if(ptr!=NULL){
			/*get the training type {"BP","BPM","CG" ...}*/
			ptr+=7;SKIP_BLANK(ptr);
			switch (*ptr){
				case 'B':
					if(*(ptr+2)=='M') _CONF.train=NN_TRAIN_BPM;
					else _CONF.train=NN_TRAIN_BP;
					break;
				case 'C':
					_CONF.train=NN_TRAIN_CG;
					break;
				default:
					_CONF.train=NN_TRAIN_UKN;
			}
		}
		ptr=STRFIND("[sample_dir",line);
		if(ptr!=NULL){
			/*get the sample directory {"dir"}*/
			ptr+=12;SKIP_BLANK(ptr);
			STR_CLEAN(ptr);
			STRDUP_REPORT(ptr,_CONF.samples,allocate);
		}
		ptr=STRFIND("[test_dir",line);
		if(ptr!=NULL){
			/*get the test directory {"dir"}*/
			ptr+=10;SKIP_BLANK(ptr);
			STR_CLEAN(ptr);
			STRDUP_REPORT(ptr,_CONF.tests,allocate);
		}
		READLINE(fp,line);
	}while(!feof(fp));
	fclose(fp);
	/*checks*/
	if(_CONF.type==NN_TYPE_UKN){
		NN_ERROR(stderr,"Malformed NN configuration file!\n");
		NN_ERROR(stderr,"[type] unknown or missing...\n");
		goto FAIL;
	}
	if(_CONF.need_init==TRUE){
		if(parameter[0]==0){
			NN_ERROR(stderr,"Malformed NN configuration file!\n");
			NN_ERROR(stderr,"[input] wrong or missing...\n");
			goto FAIL;
		}
		if(parameter[1]==0){
			NN_ERROR(stderr,"Malformed NN configuration file!\n");
			NN_ERROR(stderr,"[hidden] wrong or missing...\n");
			goto FAIL;
		}
		if(parameter[2]==0){
			NN_ERROR(stderr,"Malformed NN configuration file!\n");
			NN_ERROR(stderr,"[output] wrong or missing...\n");
			goto FAIL;
		}
		is_ok=TRUE;
		for(idx=0;idx<parameter[1];idx++) is_ok=(is_ok && (n_hiddens[idx]!=0));
		if(!is_ok) {
			NN_ERROR(stderr,"Malformed NN configuration file!\n");
			NN_ERROR(stderr,"[hidden] some have a 0 neuron content!\n");
		}
		is_ok=_NN(generate,kernel)(conf,parameter[0],parameter[1],
				parameter[2],n_hiddens);
		if(!is_ok){
			NN_ERROR(stderr,"FAILED to generate NN kernel!\n");
			NN_ERROR(stderr,"[type] unsupported...\n");
			goto FAIL;
		}
	}else{
		is_ok=_NN(load,kernel)(conf);
		if(!is_ok){
			NN_ERROR(stderr,"FAILED to load the NN kernel!\n");
			goto FAIL;
		}
	}
	if(_CONF.kernel==NULL){
		NN_ERROR(stderr,"Initialization or load of NN kernel FAILED!\n");
		goto FAIL;
	}
	FREE(parameter);
	FREE(n_hiddens);
NN_OUT(stdout,"NN definition allocation: %lu (bytes)\n",allocate);
	return conf;
read_conf_fail:
	FREE(_CONF.name);
	FREE(_CONF.f_kernel);
	FREE(_CONF.samples);
	FREE(_CONF.tests);
	FREE(conf);
	FREE(parameter);
	FREE(n_hiddens);
	return NULL;
#undef FAIL
}
void _NN(dump,conf)(nn_def *conf,FILE *fp){
	UINT n_hiddens;
	UINT idx;
	if(fp==NULL) return;
	NN_WRITE(fp,"# NN configuration\n");
	NN_WRITE(fp,"[name] %s\n",_CONF.name);
	switch(_CONF.type){
		case NN_TYPE_LNN:
			NN_WRITE(fp,"[type] LNN\n");
			break;
		case NN_TYPE_SNN:
			NN_WRITE(fp,"[type] SNN\n");
			break;
		case NN_TYPE_ANN:
		default:
			NN_WRITE(fp,"[type] ANN\n");
	}
	if(_CONF.need_init) NN_WRITE(fp,"[init] generate\n");
	else {
		if(_CONF.f_kernel!=NULL) NN_WRITE(fp,"[init] %s\n",_CONF.f_kernel);
		else NN_WRITE(fp,"[init] INVALID <- this should trigger an error\n");
	}
	NN_WRITE(fp,"[seed] %i\n",_CONF.seed);
	NN_WRITE(fp,"[inputs] %i\n",_NN(get,n_inputs)(conf));
	n_hiddens=_NN(get,n_hiddens)(conf);
	NN_WRITE(fp,"[hiddens] ");
	for(idx=0;idx<n_hiddens;idx++){
		NN_WRITE(fp,"%i ",_NN(get,h_neurons)(conf,idx));
	}
	NN_WRITE(fp,"\n");
	NN_WRITE(fp,"[outputs] %i\n",_NN(get,n_outputs)(conf));
	switch(_CONF.train){
		case NN_TRAIN_BP:
			NN_WRITE(fp,"[train] BP\n");
			break;
		case NN_TRAIN_BPM:
			NN_WRITE(fp,"[train] BPM\n");
			break;
		case NN_TRAIN_CG:
			NN_WRITE(fp,"[train] CG\n");
			break;
		default:
			NN_WRITE(fp,"[train] none\n");
	}

	if(_CONF.samples!=NULL) NN_WRITE(fp,"[sample_dir] %s\n",_CONF.samples);
	else NN_WRITE(fp,"[sample_dir] INVALID <- this should trigger an error\n");
	if(_CONF.tests!=NULL) NN_WRITE(fp,"[test_dir] %s\n",_CONF.tests);
	else NN_WRITE(fp,"[test_dir] INVALID <- this should trigger an error\n");
}
/*----------------------------*/
/*+++ manipulate NN kernel +++*/
/*----------------------------*/
void _NN(free,kernel)(nn_def *conf){
        switch (_CONF.type){
	case NN_TYPE_SNN:
		/*fallthrough*/
        case NN_TYPE_ANN:
		ann_kernel_free((kernel_ann *)_CONF.kernel);
		break;
        case NN_TYPE_LNN:
        case NN_TYPE_UKN:
        default:
                return;
        }
}
BOOL _NN(generate,kernel)(nn_def *conf,...){
	va_list ap;
	switch (_CONF.type){
	case NN_TYPE_SNN:
		/*fallthrough*/
	case NN_TYPE_ANN:
		{
		UINT n_inputs;
		UINT n_hiddens;
		UINT n_outputs;
		UINT *hiddens;
		va_start(ap,conf);
		n_inputs=va_arg(ap,UINT);
		n_hiddens=va_arg(ap,UINT);
		n_outputs=va_arg(ap,UINT);
		hiddens=va_arg(ap,UINT*);
		_CONF.kernel=(void *)ann_generate(&(_CONF.seed),n_inputs,n_hiddens,
			n_outputs,hiddens);
		if(_CONF.kernel==NULL) return FALSE;
		}
		return TRUE;
	case NN_TYPE_LNN:
	case NN_TYPE_UKN:
	default:
		return FALSE;
	}
}
BOOL _NN(load,kernel)(nn_def *conf){
	if(_CONF.f_kernel==NULL) return FALSE;
	switch (_CONF.type){
	case NN_TYPE_SNN:
		/*fallthrough*/
	case NN_TYPE_ANN:
		_CONF.kernel=(void *)ann_load(_CONF.f_kernel);
		if(_CONF.kernel==NULL) return FALSE;
		return TRUE;
	case NN_TYPE_LNN:
	case NN_TYPE_UKN:
	default:
		return FALSE;
	}
}
void _NN(dump,kernel)(nn_def *conf, FILE *output){
	if(_CONF.kernel==NULL) return;
	switch (_CONF.type){
	case NN_TYPE_SNN:
		/*fallthrough*/
	case NN_TYPE_ANN:
		ann_dump((kernel_ann *)_CONF.kernel,output);
		break;
	case NN_TYPE_LNN:
	case NN_TYPE_UKN:
	default:
		return;
	}
}
/*----------------------------*/
/*+++ Access NN parameters +++*/
/*----------------------------*/
UINT _NN(get,n_inputs)(nn_def *conf){
	if(_CONF.f_kernel==NULL) return FALSE;
	switch (_CONF.type){
	case NN_TYPE_SNN:
		/*fallthrough*/
	case NN_TYPE_ANN:
		return ((kernel_ann *)_CONF.kernel)->n_inputs;
	case NN_TYPE_LNN:
	case NN_TYPE_UKN:
	default:
		return 0;
	}
}
UINT _NN(get,n_hiddens)(nn_def *conf){
	if(_CONF.f_kernel==NULL) return FALSE;
	switch (_CONF.type){
	case NN_TYPE_SNN:
		/*fallthrough*/
	case NN_TYPE_ANN:
		return ((kernel_ann *)_CONF.kernel)->n_hiddens;
	case NN_TYPE_LNN:
	case NN_TYPE_UKN:
	default:
		return 0;
	}
}
UINT _NN(get,n_outputs)(nn_def *conf){
	if(_CONF.f_kernel==NULL) return FALSE;
	switch (_CONF.type){
	case NN_TYPE_SNN:
		/*fallthrough*/
	case NN_TYPE_ANN:
		return ((kernel_ann *)_CONF.kernel)->n_outputs;
	case NN_TYPE_LNN:
	case NN_TYPE_UKN:
	default:
		return 0;
	}
}
UINT _NN(get,h_neurons)(nn_def *conf,UINT layer){
	if(_CONF.f_kernel==NULL) return FALSE;
	switch (_CONF.type){
	case NN_TYPE_SNN:
		/*fallthrough*/
	case NN_TYPE_ANN:
		if(layer > ((kernel_ann *)_CONF.kernel)->n_hiddens) return FALSE;
		if(((kernel_ann *)_CONF.kernel)->hiddens==NULL) return FALSE;
		return ((kernel_ann *)_CONF.kernel)->hiddens[layer].n_neurons;
	case NN_TYPE_LNN:
	case NN_TYPE_UKN:
	default:
		return 0;
	}
}
/*------------------*/
/*+++ sample I/O +++*/
/*------------------*/
BOOL _NN(read,sample)(CHAR *filename,DOUBLE **in,DOUBLE **out){
#define FAIL nn_sample_read_fail
	PREP_READLINE();
	CHAR *line=NULL;
	CHAR *ptr,*ptr2;
	UINT n_in,n_out;
	UINT idx;
	FILE *fp;
	/**/
	if(filename==NULL) return FALSE;
	fp=fopen(filename,"r");
	if(fp==NULL) return FALSE;
	READLINE(fp,line);
	if(line==NULL){
		NN_ERROR(stderr,"sample %s read failed!\n",filename);
		fclose(fp);
		return FALSE;
	}
	do{
		ptr=STRFIND("[input",line);
		if(ptr!=NULL){
			/*read inputs*/
			ptr+=7;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				NN_ERROR(stderr,"sample %s input read failed!\n",filename);
				goto FAIL;
			}
			GET_UINT(n_in,ptr,ptr2);
			if(n_in==0){
				NN_ERROR(stderr,"sample %s input read failed!\n",filename);
				goto FAIL;
			}
			READLINE(fp,line);/*line immediately after should contain input*/
			ALLOC(*in,n_in,DOUBLE);
			ptr=&(line[0]);SKIP_BLANK(ptr);
			for(idx=0;idx<(n_in-1);idx++){
				GET_DOUBLE((*in)[idx],ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
				ptr=ptr2+1;SKIP_BLANK(ptr);
			}
			/*get the last one*/
			GET_DOUBLE((*in)[n_in-1],ptr,ptr2);/*no assert here*/
		}
		ptr=STRFIND("[output",line);
		if(ptr!=NULL){
			/*read outputs*/
			ptr+=8;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				NN_ERROR(stderr,"sample %s output read failed!\n",filename);
				goto FAIL;
			}
			GET_UINT(n_out,ptr,ptr2);
			if(n_out==0){
				NN_ERROR(stderr,"sample %s input read failed!\n",filename);
				goto FAIL;
			}
			READLINE(fp,line);/*line immediately after should contain input*/
			ALLOC(*out,n_out,DOUBLE);
			ptr=&(line[0]);SKIP_BLANK(ptr);
			for(idx=0;idx<(n_out-1);idx++){
				GET_DOUBLE((*out)[idx],ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
				ptr=ptr2+1;SKIP_BLANK(ptr);
			}
			/*get the last one*/
			GET_DOUBLE((*out)[n_out-1],ptr,ptr2);/*no assert here*/
		}
		READLINE(fp,line);
	}while(!feof(fp));
	fclose(fp);
	return TRUE;
nn_sample_read_fail:
	fclose(fp);
	FREE(*in);
	FREE(*out);
	return FALSE;
#undef FAIL
}
/*---------------------*/
/*+++ execute NN OP +++*/
/*---------------------*/
BOOL _NN(train,kernel)(nn_def *conf){
	DIR_S *directory;
	CHAR  *curr_file;
	CHAR   *curr_dir;
	DOUBLE    *tr_in;
	DOUBLE   *tr_out;
	UINT file_number;
	CHAR     **flist;
	CHAR  *tmp,**ptr;
	UINT is_ok;
	UINT   idx;
	UINT   jdx;
	DOUBLE res;
	/**/
	curr_file=NULL;
	curr_dir =NULL;
	flist = NULL;
	/**/
	if(_CONF.kernel==NULL) return FALSE;
	if(_CONF.samples==NULL) return FALSE;
	if(_CONF.type==NN_TYPE_UKN) return FALSE;
	/*initialize momentum*/
	switch (_CONF.type){
	case NN_TYPE_SNN:
		/*fallthrough*/
	case NN_TYPE_ANN:
		if(_CONF.train==NN_TRAIN_BPM)
			ann_momentum_init((kernel_ann *)_CONF.kernel);
		break;
	case NN_TYPE_LNN:
	case NN_TYPE_UKN:
	default:
		NN_ERROR(stdout,"unimplemented NN type!\n");
	}
	/*process sample files*/
	OPEN_DIR(directory,_CONF.samples);
	if(directory==NULL){
		NN_ERROR(stderr,"can't open sample directory: %s\n",
			_CONF.samples);
		return FALSE;
	}
	STRCAT(curr_dir,_CONF.samples,"/");
	/*count the number of file in directory*/
	FILE_FROM_DIR(directory,curr_file);
	file_number=0;
	while(curr_file!=NULL){
		if(curr_file[0]=='.') {
			FREE(curr_file);
			FILE_FROM_DIR(directory,curr_file);/*NEXT*/
			continue;
		}
		/*POSIX says char d_name[] has no fixed size*/
		STRDUP(curr_file,tmp);
		file_number++;
		ALLOC(ptr,file_number,CHAR *);
		for(idx=0;idx<(file_number-1);idx++){
			ptr[idx]=flist[idx];
		}
		ptr[file_number-1]=tmp;
		FREE(flist);
		flist=ptr;
		tmp=NULL;ptr=NULL;
		FREE(curr_file);
		FILE_FROM_DIR(directory,curr_file);/*NEXT*/
	}
	CLOSE_DIR(directory,is_ok);
	if(is_ok){
		NN_ERROR(stderr,"trying to close %s directory. IGNORED\n",curr_dir);
	}
	if(_CONF.seed==0) _CONF.seed=time(NULL);
	srandom(_CONF.seed);
	jdx=0;
	while(jdx<file_number){
		/*get a random number between 0 and file_number-1*/
		idx=(UINT) ((DOUBLE) random()*file_number / RAND_MAX);
		while(flist[idx]==NULL){
			/*no good, get another random number*/
			idx=(UINT) ((DOUBLE) random()*file_number / RAND_MAX);
		}
		STRDUP(flist[idx],curr_file);
		FREE(flist[idx]);flist[idx]=NULL;jdx++;
		NN_OUT(stdout,"TRAINING FILE: %16.16s\t",curr_file);
		/*this should never happen (but static analysis choked)*/
		if(curr_file==NULL) continue;
		STRCAT(tmp,curr_dir,curr_file);
		_NN(read,sample)(tmp,&tr_in,&tr_out);
		FREE(tmp);
		if((tr_in==NULL)||(tr_out==NULL)){
			/*something went wrong, skipping*/
			FREE(curr_file);
			FREE(tr_in);
			FREE(tr_out);
			continue;
		}
		switch (_CONF.type){
		case NN_TYPE_ANN:
			/*check training*/
			switch (_CONF.train){
			case NN_TRAIN_BPM:
				res=ann_train_BPM((kernel_ann *)_CONF.kernel,tr_in,tr_out,
					0.2,0.00001);/*TODO: set as parameters*/
				break;
			case NN_TRAIN_BP:
				res=ann_train_BP((kernel_ann *)_CONF.kernel,tr_in,tr_out,
					0.000001);/*TODO: set as parameter*/
				break;
			case NN_TRAIN_CG:
			default:
				res=0.;
				break;
			}
			break;
		case NN_TYPE_LNN:
		case NN_TYPE_SNN:
			/*check training*/
			switch (_CONF.train){
			case NN_TRAIN_BPM:
				res=snn_train_BPM((kernel_ann *)_CONF.kernel,tr_in,tr_out,
					0.2,0.00001);/*TODO: set as parameters*/
				break;
			case NN_TRAIN_BP:
				res=snn_train_BP((kernel_ann *)_CONF.kernel,tr_in,tr_out,
					0.000001);/*TODO: set as parameter*/
				break;
			case NN_TRAIN_CG:
			default:
				res=0.;
				break;
			}
			break;
		case NN_TYPE_UKN:
			res=0.;/*not ready yet*/
			break;
		default:
			/*can't happen*/
			res=0.;
		}
		if(res>0.1) NN_DBG(stdout,"bad optimization!\n");
		FREE(curr_file);
		FREE(tr_in);
		FREE(tr_out);
	}
	FREE(curr_dir);
	FREE(flist);
	/*free momentum - if any*/
	switch (_CONF.type){
	case NN_TYPE_SNN:
		/*fallthrough*/
	case NN_TYPE_ANN:
		if(_CONF.train==NN_TRAIN_BPM)
			ann_momentum_free((kernel_ann *)_CONF.kernel);
		break;
	case NN_TYPE_LNN:
	case NN_TYPE_UKN:
	default:
		NN_ERROR(stdout,"unimplemented NN type!\n");
	}
	return TRUE;
}
void _NN(run,kernel)(nn_def *conf){
	DIR_S *directory;
	CHAR  *curr_file;
	CHAR   *curr_dir;
	DOUBLE    *tr_in;
	DOUBLE   *tr_out;
	DOUBLE     probe;
	UINT file_number;
	CHAR     **flist;
	CHAR  *tmp,**ptr;
	DOUBLE res, *out;
	UINT is_ok;
	UINT guess;
	UINT   idx;
	UINT   jdx;
#ifdef   _CUDA
	cudastreams *cudas=_NN(get,cudas)();
	cudaSetDevice(0);/*useful?*/
#endif /*_CUDA*/
	/**/
	curr_file=NULL;
	curr_dir =NULL;
	flist = NULL;
	/**/
	if(_CONF.kernel==NULL) return;
	if(_CONF.tests==NULL) return;
	if(_CONF.type==NN_TYPE_UKN) return;
	/*process sample files*/
	OPEN_DIR(directory,_CONF.tests);
	if(directory==NULL){
		NN_ERROR(stderr,"can't open sample directory: %s\n",
			_CONF.samples);
		return;
	}
	STRCAT(curr_dir,_CONF.tests,"/");
	/*count the number of file in directory*/
	FILE_FROM_DIR(directory,curr_file);
	file_number=0;
	while(curr_file!=NULL){
		if(curr_file[0]=='.') {
			FREE(curr_file);
			FILE_FROM_DIR(directory,curr_file);/*NEXT*/
			continue;
		}
		/*POSIX says char d_name[] has no fixed size*/
		STRDUP(curr_file,tmp);
		file_number++;
		ALLOC(ptr,file_number,CHAR *);
		for(idx=0;idx<(file_number-1);idx++){
			ptr[idx]=flist[idx];
		}
		ptr[file_number-1]=tmp;
		FREE(flist);
		flist=ptr;
		tmp=NULL;ptr=NULL;
		FREE(curr_file);
		FILE_FROM_DIR(directory,curr_file);/*NEXT*/
	}
        CLOSE_DIR(directory,is_ok);
	if(is_ok){
		NN_ERROR(stderr,"trying to close %s directory. IGNORED\n",curr_dir);
	}
	if(_CONF.seed==0) _CONF.seed=time(NULL);
	srandom(_CONF.seed);
	jdx=0;
	while(jdx<file_number){
#define _K ((kernel_ann *)(_CONF.kernel))
#ifdef   _CUDA
		if(cudas->mem_model==CUDA_MEM_CMM){
			/*Prefetch input array to CPU*/
			cudaMemPrefetchAsync(_K->in,
				_K->n_inputs*sizeof(DOUBLE),cudaCpuDeviceId,NULL);
		}
#endif /*_CUDA*/
		/*get a random number between 0 and file_number-1*/
		idx=(UINT) ((DOUBLE) random()*file_number / RAND_MAX);
		while(flist[idx]==NULL){
			/*no good, get another random number*/
			idx=(UINT) ((DOUBLE) random()*file_number / RAND_MAX);
		}
		STRDUP(flist[idx],curr_file);
		FREE(flist[idx]);flist[idx]=NULL;jdx++;
		NN_OUT(stdout,"TESTING FILE: %16.16s\t",curr_file);
		/*this should never happen (but static analysis choked)*/
		if(curr_file==NULL) continue;
		STRCAT(tmp,curr_dir,curr_file);
		_NN(read,sample)(tmp,&tr_in,&tr_out);
		FREE(tmp);
		if((tr_in==NULL)||(tr_out==NULL)){
			FREE(curr_file);
			FREE(tr_in);
			FREE(tr_out);
			continue;
		}
		switch (_CONF.type){
		case NN_TYPE_ANN:
#ifndef  _CUDA
			ARRAY_CP(tr_in,_K->in,_K->n_inputs);
#else  /*_CUDA*/
			/*copy to GPU*/
			if(cudas->mem_model!=CUDA_MEM_CMM){
				CUDA_C2G_CP(tr_in,_K->in,_K->n_inputs,DOUBLE);
			}else{
				cudaDeviceSynchronize();/*we are still on GPU[0]*/
				ARRAY_CP(tr_in,_K->in,_K->n_inputs);
				/*Prefetch input array to GPU[0]*/
				cudaMemPrefetchAsync(_K->in,
					_K->n_inputs*sizeof(DOUBLE),0,NULL);
				cudaDeviceSynchronize();/*necessary?*/
			}
#endif /*_CUDA*/
			ann_kernel_run(_K);
#ifndef  _CUDA
			out=_K->output.vec;
#else  /*_CUDA*/
			/*copy to GPU*/
			if(cudas->mem_model!=CUDA_MEM_CMM){
				ALLOC(out,_K->n_outputs,DOUBLE);
				CUDA_G2C_CP(out,_K->output.vec,_K->n_outputs,DOUBLE);
			}else{
				/*Prefetch the output array to CPU*/
				cudaSetDevice(0);/*useful?*/
				cudaMemPrefetchAsync(_K->output.vec,
					_K->n_outputs*sizeof(DOUBLE),cudaCpuDeviceId,NULL);
				out=_K->output.vec;
				cudaDeviceSynchronize();/*necessary?*/
			}
#endif /*_CUDA*/
			res=-1.;is_ok=TRUE;
			for(idx=0;idx<_K->n_outputs;idx++){
				if(res<out[idx]) {
					guess=idx;
					res=out[idx];
				}
				if(tr_out[idx]>0.5) is_ok=idx;
			}
//			NN_COUT(stdout," init=%15.10f",res);
			if(guess==is_ok) NN_COUT(stdout," [PASS]\n");
			else NN_COUT(stdout," [FAIL idx=%i]\n",is_ok+1);
			fflush(stdout);
			break;
		case NN_TYPE_LNN:
		case NN_TYPE_SNN:
#ifndef  _CUDA
			ARRAY_CP(tr_in,_K->in,_K->n_inputs);
#else  /*_CUDA*/
			/*copy to GPU*/
			if(cudas->mem_model!=CUDA_MEM_CMM){
				CUDA_C2G_CP(tr_in,_K->in,_K->n_inputs,DOUBLE);
			}else{
				cudaDeviceSynchronize();/*we are still on GPU[0]*/
				ARRAY_CP(tr_in,_K->in,_K->n_inputs);
				/*Prefetch input array to GPU[0]*/
				cudaMemPrefetchAsync(_K->in,
					_K->n_inputs*sizeof(DOUBLE),0,NULL);
				cudaDeviceSynchronize();/*necessary?*/
			}
#endif /*_CUDA*/
			snn_kernel_run(_K);
#ifndef  _CUDA
			out=_K->output.vec;
#else  /*_CUDA*/
			/*copy to GPU*/
			if(cudas->mem_model!=CUDA_MEM_CMM){
				ALLOC(out,_K->n_outputs,DOUBLE);
				CUDA_G2C_CP(out,_K->output.vec,_K->n_outputs,DOUBLE);
			}else{
				/*Prefetch the output array to CPU*/
				cudaSetDevice(0);/*useful?*/
				cudaMemPrefetchAsync(_K->output.vec,
					_K->n_outputs*sizeof(DOUBLE),cudaCpuDeviceId,NULL);
				out=_K->output.vec;
				cudaDeviceSynchronize();/*necessary?*/
			}
#endif /*_CUDA*/
			res=0.;guess=0;is_ok=0.;
			NN_DBG(stdout," CLASS | PROBABILITY (%%)\n");
			NN_DBG(stdout,"-------|----------------\n");
			for(idx=0;idx<_K->n_outputs;idx++){
				NN_DBG(stdout,
					   " %5i | %15.10f\n",idx+1,out[idx]*100.);
				if(out[idx]>res) {
					res=out[idx];
					guess=idx;
				}
				if(tr_out[idx]>0.1) is_ok=idx;
			}
			NN_DBG(stdout,"-------|----------------\n");
			NN_COUT(stdout," BEST CLASS idx=%i P=%15.10f",guess+1,res*100);
			if(guess==is_ok) NN_COUT(stdout," [PASS]\n");
			else NN_COUT(stdout," [FAIL idx=%i]\n",is_ok+1);
			fflush(stdout);
			break;
		case NN_TYPE_UKN:
		default:
			break;
		}
		FREE(curr_file);
		FREE(tr_in);
		FREE(tr_out);
#ifdef   _CUDA
		if(cudas->mem_model!=CUDA_MEM_CMM) FREE(out);
		else{
			/*Prefetch output array to GPU[0]*/
			cudaMemPrefetchAsync(_K->output.vec,
				_K->n_outputs*sizeof(DOUBLE),0,NULL);
		}
#endif /*_CUDA*/
	}
	FREE(curr_dir);
	FREE(flist);
#undef _K
}

#undef _CONF
