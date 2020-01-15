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
/*GLOBAL VARIABLE: there it a unique runtime per run
 *  for which each use of library routine refers to.*/
nn_runtime lib_runtime;
/*new defs to take into account verbosity*/
#define NN_DBG(_file,...) do{\
	if(lib_runtime.nn_verbose>2){\
		_OUT((_file),"NN(DBG): ");\
		_OUT((_file), __VA_ARGS__);\
	}\
}while(0)
#define NN_OUT(_file,...) do{\
	if(lib_runtime.nn_verbose>1){\
		_OUT((_file),"NN: ");\
		_OUT((_file), __VA_ARGS__);\
	}\
}while(0)
#define NN_WARN(_file,...) do{\
	if(lib_runtime.nn_verbose>0){\
		_OUT((_file),"NN(WARN): ");\
		_OUT((_file), __VA_ARGS__);\
	}\
}while(0)
#define NN_ERROR(_file,...) do{\
	_OUT((_file),"NN(ERR): ");\
	_OUT((_file), __VA_ARGS__);\
}while(0)
#define NN_WRITE _OUT
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
#ifdef _PBLAS
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
#else
	cudaGetDeviceCount(&(lib_runtime.cudas.n_gpu));
	CHK_ERR(init_device_count);
	if(lib_runtime.cudas.n_gpu<1) {
		NN_WARN(stderr,"CUDA error: no CUDA-capable device reported.\n");
		return FALSE;
	}
	NN_OUT(stdout,"CUDA started, found %i GPU(s).\n",lib_runtime.cudas.n_gpu);
#ifdef _CUBLAS
	cublasStatus_t err;
	err=cublasCreate(&(lib_runtime.cudas.cuda_handle));
	if(err!=CUBLAS_STATUS_SUCCESS){
		NN_ERROR(stderr,"CUDA error: can't create a CUBLAS context.\n");
		_NN(unset,capability)(NN_CAP_CUBLAS);
		return TRUE;
	}
	err=cublasSetPointerMode(lib_runtime.cudas.cuda_handle,CUBLAS_POINTER_MODE_HOST);
	if(err!=CUBLAS_STATUS_SUCCESS){
		NN_WARN(stderr,"CUBLAS error: fail to set pointer mode.\n");
		return TRUE;
	}
#else /*_CUBLAS*/
	cudaGetDevice(&(lib_runtime.cudas.cuda_handle));
	CHK_ERR(init_device_handle);
#endif /*_CUBLAS*/
#endif
}
BOOL _NN(init,BLAS)(){
#ifdef PBLAS
#ifdef _MKL
        mkl_set_dynamic(0);
#endif /*_MKL*/
	NN_OUT(stdout,"USING PBLAS.\n");
	return TRUE;
#elif defined(SBLAS)
#ifdef _MKL
        mkl_set_dynamic(0);
#endif /*_MKL*/
	NN_OUT(stdout,"USING SBLAS.\n");
	return TRUE;
#else /*no PBLAS no SBLAS*/
	NN_WARN(stdout,"NOT USING BLAS.\n");
	return FALSE;
#endif
}
int _NN(init,all)(){
	BOOL is_ok=FALSE;
	nn_cap capability = _NN(get,capabilities)();
	if(capability & NN_CAP_OMP) {
		is_ok|=_NN(init,OMP)();
	}
	if(capability & NN_CAP_MPI) {
		is_ok|=_NN(init,MPI)();
	}
	if(capability & NN_CAP_CUDA) {
		is_ok|=_NN(init,CUDA)();
	}
	if((capability & NN_CAP_PBLAS)||(capability & NN_CAP_SBLAS)){
		is_ok|=_NN(init,BLAS)();

	}
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
	UINT idx;
	if(lib_runtime.cudas.cuda_n_streams>1){
		for(idx=0;idx<lib_runtime.cudas.cuda_n_streams;idx++)
			cudaStreamDestroy(lib_runtime.cudas.cuda_streams[idx]);
	}
	free(lib_runtime.cudas.cuda_streams);
	lib_runtime.cudas.cuda_streams=NULL;
#ifdef _CUBLAS
	cublasDestroy(lib_runtime.cudas.cuda_handle);
#endif
	cudaDeviceReset();
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
	if((capability & NN_CAP_PBLAS)||(capability & NN_CAP_SBLAS)) is_ok|=_NN(deinit,BLAS)();
	if(is_ok) return 0;
	else return -1;
}
/*--------------------------*/
/*+++ set/get parameters +++*/
/*--------------------------*/
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
	NN_WARN(stdout,"Changing MPI num_tasks is not implemented yet (and is generally not a good idea).\n");
	NN_WARN(stdout,"However, the possibility is left open for future implementations... -- OVHPA\n");
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
BOOL _NN(set,cuda_streams)(UINT n_streams){
#ifndef _CUDA
	return FALSE;
#else
	UINT idx;
	/*setting new cuda_streams should reset the cuda_streams*/
	/*only if cuda_streams was initialized properly before..*/
	if(lib_runtime.cudas.cuda_streams!=NULL){
		/*first we need to wipe previous streams*/
		if(lib_runtime.cudas.cuda_n_streams>1){
			for(idx=0;idx<lib_runtime.cudas.cuda_n_streams;idx++)
				cudaStreamDestroy(lib_runtime.cudas.cuda_streams[idx]);
		}
		free(lib_runtime.cudas.cuda_streams);
		lib_runtime.cudas.cuda_streams=NULL;
	}
	if(n_streams<2){
		/*assign a unique "NULL" stream*/
		lib_runtime.cudas.cuda_n_streams=1;
		ALLOC(lib_runtime.cudas.cuda_streams,sizeof(cudaStream_t),cudaStream_t);
		lib_runtime.cudas.cuda_streams[0]=NULL;
	}else{
		lib_runtime.cudas.cuda_n_streams=n_streams;
		ALLOC(lib_runtime.cudas.cuda_streams,n_streams*sizeof(cudaStream_t),cudaStream_t);
		for(idx=0;idx<lib_runtime.cudas.cuda_n_streams;idx++){
			cudaStreamCreateWithFlags(&(lib_runtime.cudas.cuda_streams[idx]),
				cudaStreamNonBlocking);
		}
	}
#ifdef _CUBLAS
	/*this step is optional, but it seems that CUBLAS prefers
	 *to start on its own first stream...*/
        cublasSetStream(lib_runtime.cudas.cuda_handle,lib_runtime.cudas.cuda_streams[0]);
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
	lib_runtime.nn_num_blas=n_blas;
#ifdef _OMP
	omp_set_num_threads(lib_runtime.nn_num_threads);
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
/*load neural network definition file*/
nn_def *_NN(conf,load)(CHAR *filename){
#define FAIL read_conf_fail
	PREP_READLINE();
	CHAR *line=NULL;
	CHAR *ptr,*ptr2;
	UINT *parameter;
	UINT *n_hiddens;
	UINT64 allocate;
	nn_def  *neural;
	BOOL is_ok;
	UINT   idx;
	FILE   *fp;
	/*init*/
	allocate=0;
	n_hiddens=NULL;
	ALLOC_REPORT(neural,1,nn_def,allocate);
	neural->need_init=FALSE;
	neural->train=NN_TRAIN_UKN;
	neural->type=NN_TYPE_UKN;
	ALLOC(parameter,3,UINT);
	/**/
	fp=fopen(filename,"r");
	if(!fp){
		_OUT(stderr,"Error opening configuration file: %s\n",filename);
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
			STRDUP_REPORT(ptr,neural->name,allocate);
		}
		ptr=STRFIND("[type",line);
		if(ptr!=NULL){
			/*get type {"ANN", "PNN", ...}*/
			ptr+=6;SKIP_BLANK(ptr);
			switch (*ptr){
			case 'L':
				neural->type=NN_TYPE_LNN;
				break;
			case 'P':
				neural->type=NN_TYPE_PNN;
				break;
			case 'A':
				/*fallthrough*/
			default:
				neural->type=NN_TYPE_ANN;
				break;
			}
		}
		ptr=STRFIND("[init",line);
		if(ptr!=NULL){
			/*get init {"generate", "file"}*/
			ptr+=6;SKIP_BLANK(ptr);
			if((STRFIND("generate",line)!=NULL)
			 ||(STRFIND("GENERATE",line)!=NULL)){
_OUT(stdout,"NN generating kernel!\n");
				neural->need_init=TRUE;
			}else{
_OUT(stdout,"NN loading kernel!\n");
				neural->need_init=FALSE;
				STR_CLEAN(ptr);
				STRDUP_REPORT(ptr,neural->f_kernel,allocate);
				if(neural->f_kernel==NULL){
					_OUT(stderr,"Malformed NN configuration file!\n");
					_OUT(stderr,"keyword: init, can't read filename: %s\n",ptr);
					goto FAIL;
				}
			}
		}
		ptr=STRFIND("[seed",line);
		if(ptr!=NULL){
			ptr+=6;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				_OUT(stderr,"Malformed NN configuration file!\n");
				_OUT(stderr,"keyword: seed, value: %s\n",ptr);
				goto FAIL;
			}
			GET_UINT(neural->seed,ptr,ptr2);
		}
		ptr=STRFIND("[input",line);
		if(ptr!=NULL){
			/*get number of inputs {integer}*/
			ptr+=7;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				_OUT(stderr,"Malformed NN configuration file!\n");
				_OUT(stderr,"keyword: input, value: %s\n",ptr);
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
				_OUT(stderr,"Malformed NN configuration file!\n");
				_OUT(stderr,"keyword: hidden, value: %s\n",ptr);
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
				_OUT(stderr,"Malformed NN configuration file!\n");
				_OUT(stderr,"keyword: output, value: %s\n",ptr);
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
					if(*(ptr+2)=='M') neural->train=NN_TRAIN_BPM;
					else neural->train=NN_TRAIN_BP;
					break;
				case 'C':
					neural->train=NN_TRAIN_CG;
					break;
				default:
					neural->train=NN_TRAIN_UKN;
			}
		}
		ptr=STRFIND("[sample_dir",line);
		if(ptr!=NULL){
			/*get the sample directory {"dir"}*/
			ptr+=12;SKIP_BLANK(ptr);
			STR_CLEAN(ptr);
			STRDUP_REPORT(ptr,neural->samples,allocate);
		}
		ptr=STRFIND("[test_dir",line);
		if(ptr!=NULL){
			/*get the test directory {"dir"}*/
			ptr+=10;SKIP_BLANK(ptr);
			STR_CLEAN(ptr);
			STRDUP_REPORT(ptr,neural->tests,allocate);
		}
		READLINE(fp,line);
	}while(!feof(fp));
	/*checks*/
	if(neural->type==NN_TYPE_UKN){
		_OUT(stderr,"Malformed NN configuration file!\n");
		_OUT(stderr,"keyword: type; unknown or missing...\n");
		goto FAIL;
	}
	if(neural->need_init==TRUE){
		if(parameter[0]==0){
			_OUT(stderr,"Malformed NN configuration file!\n");
			_OUT(stderr,"keyword: input; wrong or missing...\n");
			goto FAIL;
		}
		if(parameter[1]==0){
			_OUT(stderr,"Malformed NN configuration file!\n");
			_OUT(stderr,"keyword: hidden; wrong or missing...\n");
			goto FAIL;
		}
		if(parameter[2]==0){
			_OUT(stderr,"Malformed NN configuration file!\n");
			_OUT(stderr,"keyword: output; wrong or missing...\n");
			goto FAIL;
		}
		is_ok=_NN(kernel,generate)(neural,parameter[0],parameter[1],
				parameter[2],n_hiddens);
		if(!is_ok){
			_OUT(stderr,"FAILED to generate NN kernel!\n");
			_OUT(stderr,"keyword: type; unsupported...\n");
			goto FAIL;
		}
	}else{
		is_ok=_NN(kernel,load)(neural);
		if(!is_ok){
			_OUT(stderr,"FAILED to load the NN kernel!\n");
			goto FAIL;
		}
	}
	if(neural->kernel==NULL){
		_OUT(stderr,"Initialization or load of NN kernel FAILED!\n");
		goto FAIL;
	}
	FREE(parameter);
	FREE(n_hiddens);
	fclose(fp);
_OUT(stdout,"NN definition allocation: %lu (bytes)\n",allocate);
	return neural;
read_conf_fail:
	FREE(neural->name);neural->name=NULL;
	FREE(neural->samples);neural->samples=NULL;
	FREE(neural->tests);neural->tests=NULL;
	FREE(neural);
	FREE(parameter);
	FREE(n_hiddens);
	return NULL;
#undef FAIL
}

void _NN(conf,dump)(FILE *fp,nn_def *neural){
	UINT n_hiddens;
	UINT idx;
	if(fp==NULL) return;
	_OUT(fp,"# NN configuration\n");
	_OUT(fp,"[name] %s\n",neural->name);
	switch(neural->type){
		case NN_TYPE_LNN:
			_OUT(fp,"[type] LNN\n");
			break;
		case NN_TYPE_PNN:
			_OUT(fp,"[type] PNN\n");
			break;
		case NN_TYPE_ANN:
		default:
			_OUT(fp,"[type] ANN\n");
	}
	if(neural->need_init) _OUT(fp,"[init] generate\n");
	else _OUT(fp,"[init] %s\n",neural->f_kernel);
	_OUT(fp,"[seed] %i\n",neural->seed);
	_OUT(fp,"[inputs] %i\n",_NN(get,n_inputs)(neural));
	n_hiddens=_NN(get,n_hiddens)(neural);
	_OUT(fp,"[hiddens] ");
	for(idx=0;idx<n_hiddens;idx++){
		_OUT(fp,"%i ",_NN(get,h_neurons)(neural,idx));
	}
	_OUT(fp,"\n");
	_OUT(fp,"[outputs] %i\n",_NN(get,n_outputs)(neural));
	switch(neural->train){
		case NN_TRAIN_BP:
			_OUT(fp,"[train] BP\n");
			break;
		case NN_TRAIN_BPM:
			_OUT(fp,"[train] BPM\n");
			break;
		case NN_TRAIN_CG:
			_OUT(fp,"[train] CG\n");
			break;
		default:
			_OUT(fp,"[train] none\n");
	}

	if(neural->samples!=NULL) _OUT(fp,"[sample_dir] %s\n",neural->samples);
	if(neural->tests!=NULL) _OUT(fp,"[test_dir] %s\n",neural->tests);
}

/*----------------------------*/
/*+++ Access NN parameters +++*/
/*----------------------------*/
UINT _NN(get,n_inputs)(nn_def *neural){
	switch (neural->type){
	case NN_TYPE_ANN:
		return ((_kernel *)neural->kernel)->n_inputs;
	case NN_TYPE_LNN:
	case NN_TYPE_PNN:
	case NN_TYPE_UKN:
	default:
		return 0;
	}
}
UINT _NN(get,n_hiddens)(nn_def *neural){
	switch (neural->type){
	case NN_TYPE_ANN:
		return ((_kernel *)neural->kernel)->n_hiddens;
	case NN_TYPE_LNN:
	case NN_TYPE_PNN:
	case NN_TYPE_UKN:
	default:
		return 0;
	}
}
UINT _NN(get,n_outputs)(nn_def *neural){
	switch (neural->type){
	case NN_TYPE_ANN:
		return ((_kernel *)neural->kernel)->n_outputs;
	case NN_TYPE_LNN:
	case NN_TYPE_PNN:
	case NN_TYPE_UKN:
	default:
		return 0;
	}
}
UINT _NN(get,h_neurons)(nn_def *neural,UINT layer){
	switch (neural->type){
	case NN_TYPE_ANN:
		return ((_kernel *)neural->kernel)->hiddens[layer].n_neurons;
	case NN_TYPE_LNN:
	case NN_TYPE_PNN:
	case NN_TYPE_UKN:
	default:
		return 0;
	}
}
/*GENERAL*/
BOOL _NN(kernel,generate)(nn_def *neural,UINT n_inputs,UINT n_hiddens,
							UINT n_outputs,UINT *hiddens){
	switch (neural->type){
	case NN_TYPE_ANN:
		neural->kernel=(void *)ann_generate(&(neural->seed),n_inputs,n_hiddens,
			n_outputs,hiddens);
		if(neural->kernel==NULL) return FALSE;
		return TRUE;
	case NN_TYPE_LNN:
	case NN_TYPE_PNN:
	case NN_TYPE_UKN:
	default:
		return FALSE;
	}
}
BOOL _NN(kernel,load)(nn_def *neural){
	switch (neural->type){
	case NN_TYPE_ANN:
		neural->kernel=(void *)ann_load(neural->f_kernel);
		if(neural->kernel==NULL) return FALSE;
		return TRUE;
	case NN_TYPE_LNN:
	case NN_TYPE_PNN:
	case NN_TYPE_UKN:
	default:
		return FALSE;
	}
}

void _NN(kernel,dump)(nn_def *neural,FILE *output){
	switch (neural->type){
	case NN_TYPE_ANN:
		ann_dump((_kernel *)neural->kernel,output);
		break;
	case NN_TYPE_LNN:
	case NN_TYPE_PNN:
	case NN_TYPE_UKN:
	default:
		return;
	}
}

BOOL _NN(sample,read)(CHAR *filename,DOUBLE **in,DOUBLE **out){
#define FAIL nn_sample_read_fail
	PREP_READLINE();
	CHAR *line=NULL;
	CHAR *ptr,*ptr2;
	UINT n_in,n_out;
	UINT idx;
	FILE *fp;
	/**/
	n_in=0;
	n_out=0;
	/**/
	fp=fopen(filename,"r");
	if(fp==NULL) return FALSE;
	READLINE(fp,line);
	if(line==NULL){
		_OUT(stderr,"NN ERROR: sample %s read failed!\n",filename);
		return FALSE;
	}
	do{
		ptr=STRFIND("[input",line);
		if(ptr!=NULL){
			/*read inputs*/
			ptr+=7;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				_OUT(stderr,"NN ERR: sample %s input read failed!\n",filename);
				goto FAIL;
			}
			GET_UINT(n_in,ptr,ptr2);
			if(n_in==0){
				_OUT(stderr,"NN ERR: sample %s input read failed!\n",filename);
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
				_OUT(stderr,"NN ERR: sample %s output read failed!\n",filename);
				goto FAIL;
			}
			GET_UINT(n_out,ptr,ptr2);
			if(n_out==0){
				_OUT(stderr,"NN ERR: sample %s input read failed!\n",filename);
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
	FREE(in);
	FREE(out);
	return FALSE;
#undef FAIL
}


BOOL _NN(kernel,train)(nn_def *neural){
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
	if(neural->type==NN_TYPE_UKN) return FALSE;
	/*initialize momentum*/
	switch (neural->type){
	case NN_TYPE_ANN:
		if(neural->train==NN_TRAIN_BPM)
			ann_momentum_init((_kernel *)neural->kernel);
		break;
	case NN_TYPE_LNN:
	case NN_TYPE_PNN:
	case NN_TYPE_UKN:
	default:
		_OUT(stdout,"NN type not ready!\n");
	}
	/*process sample files*/
	OPEN_DIR(directory,neural->samples);
	if(directory==NULL){
		_OUT(stderr,"NN ERR: can't open sample directory: %s\n",
			neural->samples);
		return FALSE;
	}
	STRCAT(curr_dir,neural->samples,"/");
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
		_OUT(stderr,"ERROR: trying to close %s directory. IGNORED\n",curr_dir);
	}
	if(neural->seed==0) neural->seed=time(NULL);
	srandom(neural->seed);
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
		_OUT(stdout,"TRAINING FILE: %s\t",curr_file);
		STRCAT(tmp,curr_dir,curr_file);
		_NN(sample,read)(tmp,&tr_in,&tr_out);
		switch (neural->type){
		case NN_TYPE_ANN:
			/*do all case of type*/
			switch (neural->train){
			case NN_TRAIN_BPM:
				res=ann_train_BPM((_kernel *)neural->kernel,tr_in,tr_out,
					0.2,0.00001);/*TODO: set as parameters*/
				break;
			case NN_TRAIN_BP:
				res=ann_train_BP((_kernel *)neural->kernel,tr_in,tr_out,
					0.000001);/*TODO: set as parameter*/
			case NN_TRAIN_CG:
			default:
				res=0.;
				break;
			}
			break;
		case NN_TYPE_LNN:
		case NN_TYPE_PNN:
		case NN_TYPE_UKN:
			res=0.;/*not ready yet*/
			break;
		default:
			/*can't happen*/
			res=0.;
		}
		if(res>0.1) _OUT(stdout,"#");
		FREE(curr_file);
		FREE(tr_in);
		FREE(tr_out);
	}
	FREE(flist);
	return TRUE;
}

void _NN(kernel,run)(nn_def *neural){
	DIR_S *directory;
	CHAR  *curr_file;
	CHAR   *curr_dir;
	DOUBLE    *tr_in;
	DOUBLE   *tr_out;
	DOUBLE     probe;
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
	if(neural->type==NN_TYPE_UKN) return;
	/*process sample files*/
	OPEN_DIR(directory,neural->tests);
	if(directory==NULL){
		_OUT(stderr,"NN ERR: can't open sample directory: %s\n",
			neural->samples);
		return;
	}
	STRCAT(curr_dir,neural->tests,"/");
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
		_OUT(stderr,"ERROR: trying to close %s directory. IGNORED\n",curr_dir);
	}
	if(neural->seed==0) neural->seed=time(NULL);
	srandom(neural->seed);
	jdx=0;
	while(jdx<file_number){
#define _K ((_kernel *)(neural->kernel))
		/*get a random number between 0 and file_number-1*/
		idx=(UINT) ((DOUBLE) random()*file_number / RAND_MAX);
		while(flist[idx]==NULL){
			/*no good, get another random number*/
			idx=(UINT) ((DOUBLE) random()*file_number / RAND_MAX);
		}
		STRDUP(flist[idx],curr_file);
		FREE(flist[idx]);flist[idx]=NULL;jdx++;
		_OUT(stdout,"TESTING FILE: %s\t",curr_file);
		STRCAT(tmp,curr_dir,curr_file);
		_NN(sample,read)(tmp,&tr_in,&tr_out);
		switch (neural->type){
		case NN_TYPE_ANN:
			ARRAY_CP(tr_in,_K->in,_K->n_inputs);
			ann_kernel_run(_K);
			res=0.;is_ok=TRUE;
			for(idx=0;idx<_K->n_outputs;idx++){
				res+=(tr_out[idx]-_K->output.vec[idx])*
					(tr_out[idx]-_K->output.vec[idx]);
				if(_K->output.vec[idx]>0.1) probe=1.0;
				else probe=-1.0;
				if(tr_out[idx]!=probe) is_ok=FALSE;

			}
			res*=0.5;
			_OUT(stdout," init=%15.10f",res);
			if(is_ok==TRUE) _OUT(stdout," SUCCESS!\n");
			else _OUT(stdout," FAIL!\n");
			fflush(stdout);
			break;
#undef _K
		case NN_TYPE_LNN:
		case NN_TYPE_PNN:
		case NN_TYPE_UKN:
			res=0.;/*not ready yet*/
			break;
		default:
			/*can't happen*/
			res=0.;
		}
		FREE(curr_file);
		FREE(tr_in);
		FREE(tr_out);
	}
	FREE(flist);
}


