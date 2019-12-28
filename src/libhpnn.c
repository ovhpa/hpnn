/*
 * libhpnn.c
 *
 * Copyright (C) 2019 - Hubert Valencia
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
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

/*GLOBAL VARIABLES TODO: include in the main handler*/
SHORT	nn_verbose =0;
BOOL	nn_dry =FALSE;
/*^^^ OMP specific*/
#ifdef _OMP
UINT nn_num_threads=1;
UINT nn_num_blas  = 1;
#endif
/*^^^ CUDA specific*/
#ifdef _CUDA
cudastreams cudas;
#endif
/*------------------*/
/*+++ NN methods +++*/
/*------------------*/
/*^^^ depending on the host program, MPI/OMP/CUDA/BLAS initialization can be un-
 * necessary. However, proper initialization _has to be done_!        -- OVHPA*/
/*--------------------------*/
/*+++ initialize library +++*/
/*--------------------------*/
void _NN(inc,verbose)(){
        nn_verbose++;
        if(nn_verbose>0) _OUT(stdout,"# NN: increasing verbosity\n");
}
void _NN(toggle,dry)(){
        nn_dry^=nn_dry;
}
int _NN(init,all)(){
#ifdef _CUDA
	/*general GPU device init*/
	//cudaError_t err;
	cudaGetDeviceCount(&(cudas.n_gpu));
	CHK_ERR(init_device_count);
	if(cudas.n_gpu<1) {
		_OUT(stderr,"CUDA error: no CUDA-capable device reported.\n");
		exit(-1);
	}
	_OUT(stdout,"CUDA started, found %i GPU(s).\n",cudas.n_gpu);
	/*TODO: create 1 context / GPU*/
#ifdef _CUBLAS
	cublasStatus_t err;
	err=cublasCreate(&cudas.cuda_handle);
	if(err!=CUBLAS_STATUS_SUCCESS){
		_OUT(stderr,"CUDA error: can't create a CUBLAS context.\n");
		exit(-1);
	}
	err=cublasSetPointerMode(cudas.cuda_handle,CUBLAS_POINTER_MODE_HOST);
	if(err!=CUBLAS_STATUS_SUCCESS){
		_OUT(stderr,"CUBLAS error: fail to set pointer mode.\n");
		exit(-1);
	}
#else /*_CUBLAS*/
	cudaGetDevice(&(cudas.cuda_handle));
	CHK_ERR(init_device_handle);
#endif /*_CUBLAS*/
#endif /*_CUDA*/
#ifdef _MPI
	int n_streams;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD,&n_streams);
	if(n_streams<2) {
		_OUT(stdout,"#WARNING: libhpnn was compiled with MPI,\n");
		_OUT(stdout,"but only one task is used, which may not\n");
		_OUT(stdout,"be what you intended and is inefficient.\n");
		_OUT(stdout,"Please switch to serial version of hpnn,\n");
		_OUT(stdout,"or use several parallel tasks with -np X\n");
		_OUT(stdout,"option of mpirun.               -- OVHPA\n");
	}
	_OUT(stdout,"MPI started %i tasks.\n",n_streams);
#endif /*_MPI*/
#ifdef _MKL
#if defined (PBLAS) || defined (SBLAS)
	mkl_set_dynamic(0);
#endif
//        omp_set_nested(1);
//^^^^^^	mkl_set_num_threads(nn_num_threads);
	omp_set_num_threads(nn_num_threads);

//^^^^^^	mkl_domain_set_num_threads(nn_num_blas, MKL_DOMAIN_BLAS);
//	omp_set_max_active_levels(2);
        /*hyper-threading*/
        _OUT(stdout,"MKL started.\n");
#endif /*_MKL*/
#ifdef _OMP
        _OUT(stdout,"ANN started with %i OMP threads.\n",nn_num_threads);
#ifdef _MKL
	_OUT(stdout,"and with %i BLAS MKL threads.\n",nn_num_blas);
#endif /*_MKL*/
	fflush(stdout);
#endif /*_OMP*/
        return 0;
}
int _NN(deinit,all)(){
#ifdef _CUDA
	UINT idx;
	if(cudas.cuda_n_streams>1)
		for(idx=0;idx<cudas.cuda_n_streams;idx++)
			cudaStreamDestroy(cudas.cuda_streams[idx]);
	else {
		free(cudas.cuda_streams);
		cudas.cuda_streams=NULL;
	}
#ifdef _CUBLAS
	cublasDestroy(cudas.cuda_handle);
#else /*_CUBLAS*/
	cudaDeviceReset();
#endif /*_CUBLAS*/
#endif /*_CUDA*/
#ifdef _MPI
	MPI_Finalize();
#endif
	return 0;
}
#ifdef _CUDA
#ifdef _CUBLAS
cublasHandle_t _NN(get,cuda_handle)(){
#else /*_CUBLAS*/
int _NN(get,cuda_handle)(){
#endif /*_CUBLAS*/
	return cudas.cuda_handle;
}
cudastreams *_NN(get,cudas)(){
	return &cudas;
}
void _NN(set,cuda_streams)(UINT n_streams){
UINT idx;
	if(n_streams<2) {
		cudas.cuda_n_streams=1;
		ALLOC(cudas.cuda_streams,sizeof(cudaStream_t),cudaStream_t);
		cudas.cuda_streams[0]=NULL;
	}else{
		cudas.cuda_n_streams=n_streams;
		ALLOC(cudas.cuda_streams,n_streams*sizeof(cudaStream_t),cudaStream_t);
		for(idx=0;idx<cudas.cuda_n_streams;idx++){
			cudaStreamCreateWithFlags(&(cudas.cuda_streams[idx]),
				cudaStreamNonBlocking);
		}
	}
	_OUT(stdout,"ANN started with %i CUDA streams.\n",n_streams);
#ifdef _CUBLAS
	cublasSetStream(cudas.cuda_handle,cudas.cuda_streams[0]);
#endif /*_CUBLAS*/
}
#endif /*_CUDA*/
#ifdef _OMP
void _NN(set,omp_threads)(UINT n){
	nn_num_threads=n;	
}
UINT _NN(get,omp_threads)(){
	return nn_num_threads;
}
void _NN(set,omp_blas)(UINT n){
        nn_num_blas=n;
}
UINT _NN(get,omp_blas)(){
	return nn_num_blas;
}
#endif /*_OMP*/
/*---------------------*/
/*+++ configuration +++*/
/*---------------------*/
/*TODO: move STR_CLEAN in common.h*/
#define STR_CLEAN(pointer) do{\
	CHAR *_ptr=pointer;\
	while(*_ptr!='\0'){\
		if(*_ptr=='\t') *_ptr='\0';\
		if(*_ptr==' ') *_ptr='\0';\
		if((*_ptr=='\n')||(*_ptr=='#')) *_ptr='\0';\
		else _ptr++;\
	}\
}while(0)
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


