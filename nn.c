#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>

/* Artificial Neuron Network, abstract layer. */
/* ------------ Hubert Okadome Valencia, 2019 */

#include "common.h"
#include "ann.h"

#if defined (PBLAS) || defined (SBLAS)
#ifndef _MKL
#include <cblas.h>
#else /*_MKL*/
#include <mkl.h>
#include <mkl_cblas.h>
#endif /*_MKL*/
#endif /*PBLAS*/

#ifdef _CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif /*_CUDA*/

#ifdef _OMP
#include <omp.h>
#endif

#include "nn.h"

/*GLOBAL VARIABLES*/
SHORT	nn_verbose =0;
BOOL	nn_dry =FALSE;
#ifdef _OMP
UINT nn_num_threads=1;
UINT nn_num_blas  = 1;
#endif
#ifdef _CUDA
cublasHandle_t cuda_handle;
cudastreams cudas;
#endif


void _NN(inc,verbose)(){
        nn_verbose++;
        if(nn_verbose>0) fprintf(stdout,"# NN: increasing verbosity\n");
}

void _NN(toggle,dry)(){
        nn_dry^=nn_dry;
}

int _NN(init,all)(){
#ifdef _CUDA
	cublasStatus_t err;
	err=cublasCreate(&cuda_handle);
	if(err!=CUBLAS_STATUS_SUCCESS){
		fprintf(stderr,"CUDA error: can't create a CUDA context.\n");
		exit(-1);
	}
	cudas.cuda_handle=cuda_handle;
	err=cublasSetPointerMode(cuda_handle,CUBLAS_POINTER_MODE_HOST);
	if(err!=CUBLAS_STATUS_SUCCESS){
		fprintf(stderr,"CUDA error: fail to set pointer mode.\n");
		exit(-1);
	}
#endif
#ifdef _MKL
#if defined (PBLAS) || defined (SBLAS)
        mkl_set_dynamic(0);
#endif
        omp_set_nested(1);
//	mkl_set_num_threads(nn_num_threads);
	omp_set_num_threads(nn_num_threads);
//	mkl_domain_set_num_threads(nn_num_blas, MKL_DOMAIN_BLAS);
        omp_set_max_active_levels(2);
        /*hyper-threading*/
        fprintf(stdout,"MKL started.\n");
#endif /*_MKL*/
#ifdef _OMP
        fprintf(stdout,"\nANN started with %i threads.\n",nn_num_threads);
#ifdef _MKL
	fprintf(stdout,"and with %i BLAS threads.\n",nn_num_blas);
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
	cublasDestroy(cudas.cuda_handle);
#endif /*_CUDA*/
	return 0;
}
#ifdef _CUDA
cublasHandle_t _NN(get,cuda_handle)(){
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
			cudaStreamCreateWithFlags(&(cudas.cuda_streams[idx]),cudaStreamNonBlocking);
		}
	}
	fprintf(stdout,"ANN started with %i CUDA streams.\n",n_streams);
	cublasSetStream(cudas.cuda_handle,cudas.cuda_streams[0]);
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
BOOL _NN(kernel,generate)(nn_def *neural,UINT n_inputs,UINT n_hiddens,UINT n_outputs,UINT *hiddens){
	switch (neural->type){
	case NN_TYPE_ANN:
		neural->kernel=(void *)ann_generate(&(neural->seed),n_inputs,n_hiddens,n_outputs,hiddens);
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
		fprintf(stderr,"NN ERROR: sample %s read failed!\n",filename);
		return FALSE;
	}
	do{
		ptr=STRFIND("[input",line);
		if(ptr!=NULL){
			/*read inputs*/
			ptr+=7;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				fprintf(stderr,"NN ERROR: sample %s input read failed!\n",filename);
				goto FAIL;
			}
			GET_UINT(n_in,ptr,ptr2);
			if(n_in==0){
				fprintf(stderr,"NN ERROR: sample %s input read failed!\n",filename);
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
				fprintf(stderr,"NN ERROR: sample %s output read failed!\n",filename);
				goto FAIL;
			}
			GET_UINT(n_out,ptr,ptr2);
			if(n_out==0){
				fprintf(stderr,"NN ERROR: sample %s input read failed!\n",filename);
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
		if(neural->train==NN_TRAIN_BPM) ann_momentum_init((_kernel *)neural->kernel);
		break;
	case NN_TYPE_LNN:
	case NN_TYPE_PNN:
	case NN_TYPE_UKN:
	default:
		fprintf(stdout,"NN type not ready!\n");
	}
	/*process sample files*/
	OPEN_DIR(directory,neural->samples);
	if(directory==NULL){
		fprintf(stderr,"NN ERROR: can't open sample directory: %s\n",neural->samples);
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
		fprintf(stderr,"ERROR: trying to close %s directory. IGNORED\n",curr_dir);
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
		fprintf(stdout,"TRAINING FILE: %s\t",curr_file);
		STRCAT(tmp,curr_dir,curr_file);
		_NN(sample,read)(tmp,&tr_in,&tr_out);
		switch (neural->type){
		case NN_TYPE_ANN:
			/*do all case of type*/
			switch (neural->train){
			case NN_TRAIN_BPM:
				res=ann_train_BPM((_kernel *)neural->kernel,tr_in,tr_out,0.2,0.00001);
				break;
			case NN_TRAIN_BP:
				res=ann_train_BP((_kernel *)neural->kernel,tr_in,tr_out,0.000001);
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
//		if(res==0.) fprintf(stdout,"#");
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
		fprintf(stderr,"NN ERROR: can't open sample directory: %s\n",neural->samples);
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
		fprintf(stderr,"ERROR: trying to close %s directory. IGNORED\n",curr_dir);
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
		fprintf(stdout,"TESTING FILE: %s\t",curr_file);
		STRCAT(tmp,curr_dir,curr_file);
		_NN(sample,read)(tmp,&tr_in,&tr_out);
		switch (neural->type){
		case NN_TYPE_ANN:
			ARRAY_CP(tr_in,_K->in,_K->n_inputs);
			ann_kernel_run(_K);
			res=0.;is_ok=TRUE;
			for(idx=0;idx<_K->n_outputs;idx++){
				res+=(tr_out[idx]-_K->output.vec[idx])*(tr_out[idx]-_K->output.vec[idx]);
				if(_K->output.vec[idx]>0.1) probe=1.0;
				else probe=-1.0;
				if(tr_out[idx]!=probe) is_ok=FALSE;

			}
			res*=0.5;
			fprintf(stdout," init=%15.10f",res);
			if(is_ok==TRUE) fprintf(stdout," SUCCESS!\n");
			else fprintf(stdout," FAIL!\n");
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


