/*
+++ libhpnn - High Performance Neural Network library - file: ann.c +++
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
/*^^^  MPI specific*/
#ifdef _MPI
#include <mpi.h>
#endif
/*^^^ BLAS/MKL specific*/
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
#endif
/*^^^ OMP specific*/
#ifdef _OMP
#include <omp.h>
#endif
/*link to the main library*/
#include <libhpnn.h>
#include <libhpnn/ann.h>
#ifdef _CUDA
#include <libhpnn/cuda_ann.h>
#endif /*_CUDA*/
/*----------------------*/
/*+++ useful defines +++*/
/*----------------------*/
/*^^^ MKL specific*/
#ifdef _MKL
#define _HT mkl_set_num_threads_local(_NN(get,omp_blas)())
#else
#define _HT
#endif
/*^^^ OMP specific*/
#ifdef _OMP
#define _NT num_threads(_NN(return,omp_threads)())
#else
#define _NT
#endif
/*make life easier*/
#define KERN (*kernel)
/*-----------------------*/
/*+++ free ANN kernel +++*/
/*-----------------------*/
BOOL ann_kernel_free(kernel_ann *kernel){
	UINT idx;
	if(kernel==NULL) return FALSE;
#ifdef   _CUDA
	scuda_ann_deallocate(kernel,_NN(get,cudas)());
	FREE(KERN.hiddens);
	FREE(KERN.kerns);
	scuda_ann_free_momentum(kernel,_NN(get,cudas)());
	FREE(KERN.dw);
#else  /*_CUDA*/
	FREE(KERN.name);
	FREE(KERN.in);
	FREE(KERN.output.weights);
	FREE(KERN.output.vec);
	if(KERN.hiddens!=NULL){
		for(idx=0;idx<KERN.n_hiddens;idx++){
			FREE(KERN.hiddens[idx].weights);
			FREE(KERN.hiddens[idx].vec);
		}
		FREE(KERN.hiddens);
	}
	if(KERN.dw!=NULL){
		for(idx=0;idx<KERN.n_hiddens+1;idx++) FREE(KERN.dw[idx]);
		FREE(KERN.dw);
	}
	FREE(KERN.tmp_cpu);
#endif /*_CUDA*/
	KERN.n_inputs=0;
	KERN.n_hiddens=0;
	KERN.n_outputs=0;
	KERN.max_index=0;
	return TRUE;
}
/*------------------------*/
/*+++ alloc ANN kernel +++*/
/*------------------------*/
BOOL ann_kernel_allocate(kernel_ann *kernel,UINT n_inputs,UINT n_hiddens,
						 UINT *h_neurons, UINT n_outputs){
	UINT64 allocate=0;
#ifdef _CUDA
	uint64_t g_allocate=0;
	cudastreams *cudas=_NN(get,cudas)();
	UINT n_gpu,jdx;
#endif /*_CUDA*/
	UINT idx;
	if(kernel==NULL) return FALSE;
	/*fill all the dimensions of the kernel*/
	KERN.n_inputs=n_inputs;
	KERN.n_hiddens=n_hiddens;
	KERN.n_outputs=n_outputs;
	KERN.output.n_neurons=n_outputs;
	if(h_neurons==NULL) return FALSE;
	KERN.output.n_inputs=h_neurons[n_hiddens-1];
	/*calculate the max index*/
	KERN.max_index=n_inputs;
	if(KERN.max_index<n_outputs) KERN.max_index=n_outputs;
	if(KERN.max_index<h_neurons[0]) KERN.max_index=h_neurons[0];
	/*alloc n_hiddens on CPU first*/
	ALLOC_REPORT(KERN.hiddens,n_hiddens,layer_ann,allocate);
	KERN.hiddens[0].n_inputs=n_inputs;
	KERN.hiddens[0].n_neurons=h_neurons[0];
	for(idx=1;idx<n_hiddens;idx++){
		if(KERN.max_index<h_neurons[idx]) KERN.max_index=h_neurons[idx];
		KERN.hiddens[idx].n_inputs=h_neurons[idx-1];
		KERN.hiddens[idx].n_neurons=h_neurons[idx];
	}
	/*allocate temporary CPU array*/
	ALLOC_REPORT(KERN.tmp_cpu,KERN.max_index,DOUBLE,allocate);
#ifndef  _CUDA
	/*CPU only*/
	ALLOC_REPORT(KERN.in,n_inputs,DOUBLE,allocate);
	for(idx=0;idx<n_hiddens;idx++){
		ALLOC_REPORT(KERN.hiddens[idx].weights,
			KERN.hiddens[idx].n_inputs*KERN.hiddens[idx].n_neurons,
			DOUBLE,allocate);
		ALLOC_REPORT(KERN.hiddens[idx].vec,KERN.hiddens[idx].n_neurons,
			DOUBLE,allocate);
	}
	ALLOC_REPORT(KERN.output.weights,
		KERN.output.n_inputs*KERN.output.n_neurons,DOUBLE,allocate);
	ALLOC_REPORT(KERN.output.vec,KERN.output.n_neurons,DOUBLE,allocate);
#else  /*_CUDA*/
	_NN(get,n_gpu)(&n_gpu);
if(n_gpu>1){
	ALLOC_REPORT(KERN.kerns,n_gpu,kernel_ann *,allocate);
	if(cudas->mem_model==CUDA_MEM_EXP){
		kernel_ann *kx;
		/*we are going to need n_gpu kernels*/
		KERN.kerns[0]=kernel;/*1st one is self*/
		for(idx=1;idx<n_gpu;idx++){
			ALLOC_REPORT(kx,1,kernel_ann,allocate);
#define _CP(name) (*kx).name = KERN.name
			_CP(n_inputs);
			_CP(n_hiddens);
			_CP(n_outputs);
			_CP(output.n_neurons);
			_CP(output.n_inputs);
			_CP(max_index);
			ALLOC_REPORT((*kx).hiddens,n_hiddens,layer_ann,allocate);
			for(jdx=1;jdx<n_hiddens;jdx++){
				_CP(hiddens[jdx].n_inputs);
				_CP(hiddens[jdx].n_neurons);
			}
			(*kx).tmp_cpu=NULL;/*we won't need that*/
#undef _CP
			KERN.kerns[idx]=kx;
			kx=NULL;
		}
	}else{
		/*hack: each kerns points to the same kernel*/
		for(idx=0;idx<n_gpu;idx++) KERN.kerns[idx]=kernel;
	}
}else{
	KERN.kerns=NULL;
}
	g_allocate=scuda_ann_allocate(kernel,cudas);
#endif /*_CUDA*/
#ifdef _MPI
	NN_OUT(stdout,"For each MPI thread:\n");
#endif /*_MPI*/
	NN_OUT(stdout,"[CPU] ANN total allocation: %lu (bytes)\n",allocate);
#ifdef _CUDA
	NN_OUT(stdout,"[GPU] ANN total allocation: %lu (bytes)\n",g_allocate);
#endif /*_CUDA*/
	return TRUE;
}
/*---------------------------------*/
/*+++ load ANN kernel from file +++*/
/*---------------------------------*/
kernel_ann *ann_load(CHAR *f_kernel){
#define FAIL load_kernel_fail
	PREP_READLINE();
	CHAR *line=NULL;
	CHAR *ptr,*ptr2;
	kernel_ann *kernel;
	UINT *parameter;
	UINT64 allocate;
	FILE  *fp;
	UINT  idx;
	UINT  jdx;
	UINT  kdx;
	CHAR *name;
	UINT  n_in;
	UINT n_out;
	UINT n_hid;
	UINT n_par;
	DOUBLE *w_ptr;
#ifdef _CUDA
	cudastreams *cudas=_NN(get,cudas)();
#endif /*_CUDA*/
	/*init*/
	n_in =0;
	n_out=0;
	n_hid=0;
	n_par=0;
	name=NULL;
	kernel=NULL;
	parameter=NULL;
	/*mpi*/
#ifdef _MPI
	int bailout=0;
	UINT N,M,ndx;
	UINT n_streams,stream;
	_NN(get,mpi_tasks)(&n_streams);
	_NN(get,curr_mpi_task)(&stream);
#define MPI_BAIL_SEND for(ndx=1;ndx<n_streams;ndx++) \
	MPI_Send(&bailout,1,MPI_INT,ndx,10,MPI_COMM_WORLD)
#define MPI_BAIL_RECV MPI_Recv(&bailout,1,MPI_INT,0,10,\
	MPI_COMM_WORLD,MPI_STATUS_IGNORE)
#else /*_MPI*/
#define MPI_BAIL_SEND 
#define MPI_BAIL_RECV
#endif /*_MPI*/
	/**/
#ifdef _MPI
if(stream==0) {
#endif /*_MPI*/
	fp=fopen(f_kernel,"r");
	if(!fp){
		NN_ERROR(stderr,"Error opening kernel file: %s\n",f_kernel);
		MPI_BAIL_SEND;
		return NULL;
	}
	READLINE(fp,line);/*line 1: name (SKIP)*/
	ptr=STRFIND("[name]",line);
	if(ptr==NULL){
		NN_ERROR(stderr,"kernel file should start with [name] keyword!\n");
		goto FAIL;
	}
	ptr+=6;SKIP_BLANK(ptr);
	allocate=0;
	STRDUP_REPORT(ptr,name,allocate);
	if(name==NULL) STRDUP_REPORT("noname",name,allocate);
	/*strip extra '\n'*/
	ptr=&(name[0]);
	while(*ptr!='\0'){
		if(*ptr=='\n') *ptr='\0';
		ptr++;
	}
	do{
		ptr=STRFIND("[param]",line);
		if(ptr!=NULL){
			/*found parameters*/
			ptr=&(line[0]);SKIP_BLANK(ptr);
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			if(!ISDIGIT(*ptr)) {
				NN_ERROR(stderr,"kernel read: malformed parameter line!\n");
				goto FAIL;
			}
			/*now we need to get each parameters*/
			/*counting*/
			ptr2=ptr;
			do{
				GET_UINT(n_hid,ptr,ptr2);
				if((*ptr2=='\n')||(*ptr2=='\0')) ptr=ptr2;
				else ptr=ptr2+1;SKIP_BLANK(ptr);
				n_par++;
			}while((*ptr!='\0')&&(*ptr!='\n')&&(ptr2!=NULL));
			/*there is n_par-2 hidden layers and 1 output*/
			n_par--;
			if(n_par<2){
				NN_ERROR(stderr,
					"kernel read: parameter line has too few parameters!\n");
				goto FAIL;
			}
			n_hid=n_par-1;
			/*get number of input*/
			ptr=&(line[0]);SKIP_BLANK(ptr);
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			GET_UINT(n_in,ptr,ptr2);ptr=ptr2+1;SKIP_BLANK(ptr);
			ALLOC(parameter,n_par,UINT);
			jdx=1;
			for(idx=0;idx<n_par;idx++) {
				GET_UINT(parameter[idx],ptr,ptr2);
				jdx*=(parameter[idx]!=0);
				ptr=ptr2+1;SKIP_BLANK(ptr);
			}
			if(jdx==0){
				NN_ERROR(stderr,"kernel read: zero in parameter line!\n");
				goto FAIL;
			}
			n_out=parameter[n_par-1];
			break;
		}
		READLINE(fp,line);
	}while(!feof(fp));
	if(n_in==0) {
		NN_ERROR(stderr,"kernel read: missing parameter line!\n");
		goto FAIL;
	}
	if(n_out<1){
		NN_ERROR(stderr,"kernel read: wrong parameter n_output<1!\n");
		goto FAIL;
	}
	if(n_hid<1){
		NN_ERROR(stderr,"kernel read: wrong parameter n_hiddens<1!\n");
		goto FAIL;
	}
	/*allocate everything*/
	ALLOC_REPORT(kernel,1,kernel_ann,allocate);
	ann_kernel_allocate(kernel,n_in,n_hid,parameter,n_out);
	KERN.name=name;name=NULL;
#ifndef _MPI
	FREE(parameter);/*we need to send parameter w/ MPI*/
#endif /*_MPI*/
	/*getting weights when available*/
	rewind(fp);
	/*1- find [hidden]*/
	do{
		ptr=STRFIND("[hidden",line);
		if(ptr!=NULL){
//			[hidden X] Y -> hidden layer X has Y neurons
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			if(!ISDIGIT(*ptr)) {
				NN_ERROR(stderr,
					"kernel read: malformed hidden layer definition\n");
				goto FAIL;
			}
			GET_UINT(idx,ptr,ptr2);/*this is hidden index*/
			if(ptr2==NULL) {
				NN_ERROR(stderr,
					"kernel read: malformed hidden layer index definition!\n");
				goto FAIL;
			}
			if(idx==0){
				NN_ERROR(stderr,
					"kernel read: wrong hidden layer index (=0)!\n");
				goto FAIL;
			}
			idx--;/*start counting from 1*/
			if(idx>n_hid){
				NN_ERROR(stderr,
					"kernel read: wrong hidden layer index (> n_hiddens)!\n");
				goto FAIL;
			}
			/*check neuron number for consistency*/
			ptr=ptr2+1;
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			GET_UINT(jdx,ptr,ptr2);
			if(jdx!=KERN.hiddens[idx].n_neurons){
				NN_ERROR(stderr,
						 "kernel read: inconsistent neuron number!\n");
				NN_ERROR(stderr,"-> layer %i n_neurons=%i (expected %i)\n",
					idx+1,jdx,KERN.hiddens[idx].n_neurons);
				goto FAIL;
			}
/*now let's fetch neurons*/
READLINE(fp,line);
jdx=0;
/*prepare array*/
#ifdef   _CUDA
	if(cudas->mem_model!=CUDA_MEM_CMM){
		ALLOC(w_ptr,
			  KERN.hiddens[idx].n_inputs*KERN.hiddens[idx].n_neurons,DOUBLE);
	}else{
		/*CMM memory can be access by CPU directly*/
		/*TODO: should we prefetch?*/
		w_ptr=KERN.hiddens[idx].weights;
	}
#else  /*_CUDA*/
	w_ptr=KERN.hiddens[idx].weights;
#endif /*_CUDA*/
do{
	ptr=STRFIND("[neuron",line);
	if(ptr==NULL){
		NN_ERROR(stderr,"kernel read: neuron definition missing!\n");
		NN_ERROR(stderr,"-> hidden layer %i, neuron %i\n",idx+1,jdx+1);
		goto FAIL;
	}
	while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
	if(!ISDIGIT(*ptr)) {
		NN_ERROR(stderr,"kernel read: missing neuron number!\n");
		NN_ERROR(stderr,"-> hidden layer %i, neuron %i\n",idx+1,jdx+1);
		goto FAIL;
	}
	GET_UINT(n_par,ptr,ptr2);/*this is neuron number*/
	if(n_par<1) {
		NN_ERROR(stderr,"kernel read: neuron number<1\n");
		NN_ERROR(stderr,"-> hidden layer %i, neuron %i\n",idx+1,jdx+1);
		goto FAIL;
	}
	ptr=ptr2+1;SKIP_BLANK(ptr);
	if(!ISDIGIT(*ptr)) {
		NN_ERROR(stderr,"kernel read: neuron has no input number!\n");
		NN_ERROR(stderr,"-> hidden layer %i, neuron %i\n",idx+1,jdx+1);
		goto FAIL;
	}
	GET_UINT(n_par,ptr,ptr2);/*this is number of inputs*/
	if(n_par<1) {
		NN_ERROR(stderr,"kernel read: neuron has less that 1 input!\n");
		NN_ERROR(stderr,"-> hidden layer %i, neuron %i\n",idx+1,jdx+1);
		goto FAIL;
	}
	if(n_par>KERN.hiddens[idx].n_inputs){
		NN_ERROR(stderr,"kernel read: neuron inconsistent input number!\n");
		NN_ERROR(stderr,"-> n_input=%i (expected %i)!\n",
				 n_par,KERN.hiddens[idx].n_inputs);
		NN_ERROR(stderr,"-> hidden layer %i, neuron %i\n",idx+1,jdx+1);
		goto FAIL;
	}
	READLINE(fp,line);/*weights line*/
	ptr=&(line[0]);SKIP_BLANK(ptr);
	for(kdx=0;kdx<n_par;kdx++){
		/*read weights*/
		GET_DOUBLE(w_ptr[_2D_IDX(n_par,jdx,kdx)],ptr,ptr2);
		ASSERT_GOTO(ptr2,FAIL);
		ptr=ptr2+1;SKIP_BLANK(ptr);
	}
#ifdef   _CUDA
	/*transfer to GPU*/
	scuda_ann_weight_transfer_C2G(kernel,idx,w_ptr,cudas);
#endif /*_CUDA*/
	jdx++;
	READLINE(fp,line);
}while(jdx<KERN.hiddens[idx].n_neurons);
/*continue*/
#ifdef _CUDA
	if(cudas->mem_model!=CUDA_MEM_CMM) FREE(w_ptr);
#endif /*_CUDA*/
		} else READLINE(fp,line);
	}while(!feof(fp));
/*finally get the output weights*/
	rewind(fp);
	do{
		ptr=STRFIND("[output]",line);
		if(ptr!=NULL){
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			if(!ISDIGIT(*ptr)) {
				NN_ERROR(stderr,
					"kernel read: malformed output layer definition\n");
				goto FAIL;
			}
			/*check neuron number for consistency*/
			GET_UINT(idx,ptr,ptr2);/*this is the number of output*/
			if((ptr2==NULL)||(idx!=KERN.output.n_neurons)) {
				NN_ERROR(stderr,
					"kernel read: inconsistent neuron number for output!\n");
				NN_ERROR(stderr,"-> n_neurons=%i (expected %i)\n",
					idx,KERN.output.n_neurons);
				goto FAIL;
			}
/*now let's fetch neurons*/
READLINE(fp,line);
jdx=0;
/*prepare array*/
#ifdef   _CUDA
	if(cudas->mem_model!=CUDA_MEM_CMM){
		ALLOC(w_ptr,KERN.output.n_inputs*KERN.output.n_neurons,DOUBLE);
	}else{
		/*CMM memory can be access by CPU directly*/
		/*TODO: should we prefetch?*/
		w_ptr=KERN.output.weights;
	}
#else  /*_CUDA*/
	w_ptr=KERN.output.weights;
#endif /*_CUDA*/
do{
	ptr=STRFIND("[neuron",line);
	if(ptr==NULL){
		NN_ERROR(stderr,"kernel read: neuron definition missing!\n");
		NN_ERROR(stderr,"-> output layer, neuron %i\n",jdx+1);
		goto FAIL;
	}
	while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
	if(!ISDIGIT(*ptr)) {
		NN_ERROR(stderr,"kernel read: missing neuron number!\n");
		NN_ERROR(stderr,"-> output layer, neuron %i\n",jdx+1);
		goto FAIL;
	}
	GET_UINT(n_par,ptr,ptr2);/*this is hidden index*/
	if(n_par<1) {
		NN_ERROR(stderr,"kernel read: neuron number<1\n");
		NN_ERROR(stderr,"-> output layer, neuron %i\n",jdx+1);
		goto FAIL;
	}
	ptr=ptr2+1;SKIP_BLANK(ptr);
	if(!ISDIGIT(*ptr)) {
		NN_ERROR(stderr,"kernel read: neuron has no input number!\n");
		NN_ERROR(stderr,"-> output layer, neuron %i\n",jdx+1);
		goto FAIL;
	}
	GET_UINT(n_par,ptr,ptr2);/*this is number of inputs*/
	if(n_par<1) {
		NN_ERROR(stderr,"kernel read: neuron has less that 1 input!\n");
		NN_ERROR(stderr,"-> output layer, neuron %i\n",jdx+1);
		goto FAIL;
	}
	READLINE(fp,line);
	ptr=&(line[0]);SKIP_BLANK(ptr);
	for(kdx=0;kdx<n_par;kdx++){
		/*read weights*/
		GET_DOUBLE(w_ptr[_2D_IDX(n_par,jdx,kdx)],ptr,ptr2);
		ASSERT_GOTO(ptr2,FAIL);
		ptr=ptr2+1;SKIP_BLANK(ptr);
	}
#ifdef   _CUDA
	/*transfer to GPU (KERN.n_hiddens -> output)*/
	scuda_ann_weight_transfer_C2G(kernel,KERN.n_hiddens,w_ptr,cudas);
#endif /*_CUDA*/
	jdx++;
	READLINE(fp,line);
}while(jdx<KERN.output.n_neurons);
#ifdef _CUDA
		if(cudas->mem_model!=CUDA_MEM_CMM) FREE(w_ptr);
#endif /*_CUDA*/
/*continue*/
		}
		READLINE(fp,line);
	}while(!feof(fp));
	/*end*/
	FREE(line);
	fclose(fp);
#ifdef _MPI
	for(ndx=1;ndx<n_streams;ndx++) 
		MPI_Send(&bailout,1,MPI_INT,ndx,10,MPI_COMM_WORLD);
}/*end of master load*/
else{/*slaves*/
	MPI_BAIL_RECV;
	if(bailout) return NULL;/*try to fail nicely*/
}
/*master -> slaves*/
MPI_Bcast(&n_in,1,MPI_INT,0,MPI_COMM_WORLD);
MPI_Bcast(&n_hid,1,MPI_INT,0,MPI_COMM_WORLD);
MPI_Bcast(&n_out,1,MPI_INT,0,MPI_COMM_WORLD);
MPI_Bcast(&n_par,1,MPI_INT,0,MPI_COMM_WORLD);
if(stream!=0) ALLOC(parameter,n_par-1,UINT);
MPI_Bcast(parameter,n_par-1,MPI_INT,0,MPI_COMM_WORLD);
if(stream!=0){/*slaves*/
	/*allocate everything - NO NEED to report*/
	ALLOC(kernel,1,kernel_ann);
	ann_kernel_allocate(kernel,n_in,n_hid,parameter,n_out);
	KERN.name=name;name=NULL;
}
FREE(parameter);
/*broadcast hidden weights*/
for(idx=0;idx<n_hid;idx++){
	N=KERN.hiddens[idx].n_neurons;
	M=KERN.hiddens[idx].n_inputs;
	/*That one will broadcast CPU values over MPI*/
#ifndef  _CUDA
	MPI_Bcast(KERN.hiddens[idx].weights,M*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
#else  /*_CUDA*/
	if(cudas->mem_model!=CUDA_MEM_CMM){
		ALLOC(w_ptr,M*N,DOUBLE);/*CPU MPI ARRAY*/
	}else{
		w_ptr=KERN.hiddens[idx].weights;
	}
	if(stream==0) scuda_ann_weight_transfer_G2C(kernel,idx,&w_ptr,cudas);
	MPI_Bcast(w_ptr,M*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	/*now, everyone received CPU... put it back on GPU*/
	if(stream!=0){
		/*master already have it*/
		scuda_ann_weight_transfer_C2G(kernel,idx,w_ptr,cudas);
	}
	if(cudas->mem_model!=CUDA_MEM_CMM) FREE(w_ptr);
#endif /*_CUDA*/
}
N=KERN.output.n_neurons;
M=KERN.output.n_inputs;
/*same broadcast type as before*/
#ifndef  _CUDA
MPI_Bcast(KERN.output.weights,N*M,MPI_DOUBLE,0,MPI_COMM_WORLD);
#else  /*_CUDA*/
	if(cudas->mem_model!=CUDA_MEM_CMM){
		ALLOC(w_ptr,M*N,DOUBLE);/*CPU MPI ARRAY*/
	}else{
		w_ptr=KERN.output.weights;
	}
	if(stream==0)
		scuda_ann_weight_transfer_G2C(kernel,KERN.n_hiddens,&w_ptr,cudas);
	MPI_Bcast(w_ptr,M*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	/*now, everyone received CPU... put it back on GPU*/
	if(stream!=0){
		/*master already have it*/
		scuda_ann_weight_transfer_C2G(kernel,KERN.n_hiddens,w_ptr,cudas);
	}
	if(cudas->mem_model!=CUDA_MEM_CMM) FREE(w_ptr);
#endif /*_CUDA*/
#endif /*_MPI*/
	return kernel;
load_kernel_fail:
	MPI_BAIL_SEND;
	/*un-allocate kernel*/
	ann_kernel_free(kernel);
	FREE(kernel);
	FREE(name);
	FREE(parameter);
	FREE(line);
	fclose(fp);
	return NULL;
#undef FAIL
}
kernel_ann *ann_generate(UINT *seed,UINT n_inputs,UINT n_hiddens,
						 UINT n_outputs,UINT *hiddens){
	kernel_ann *kernel;
	UINT64 allocate;
	UINT   idx, jdx;
	DOUBLE temp_rnd;
	UINT N,M;
#ifdef _CUDA
	cudastreams *cudas=_NN(get,cudas)();
#endif
	DOUBLE *w_ptr;
#ifdef _MPI
	UINT N,M;
	UINT n_streams,stream;
	_NN(get,mpi_tasks)(&n_streams);
	_NN(get,curr_mpi_task)(&stream);
#endif /*_MPI*/
#ifdef _MPI
if(stream==0){/*master kernel generation*/
#endif /*_MPI*/
	/*this should _NOT_ be performed in parallel (random seed consistency)*/
	allocate=0.;
	if(*seed==0) *seed=time(NULL);
	srandom(*seed);
	/*allocation*/
	ALLOC_REPORT(kernel,1,kernel_ann,allocate);
	ann_kernel_allocate(kernel,n_inputs,n_hiddens,hiddens,n_outputs);
	/*randomly fill hidden weights*/
	for(idx=0;idx<n_hiddens;idx++){
		N=KERN.hiddens[idx].n_neurons;
		M=KERN.hiddens[idx].n_inputs;
		/*prepare array*/
#ifdef   _CUDA
		if(cudas->mem_model!=CUDA_MEM_CMM){
			ALLOC(w_ptr,N*M,DOUBLE);
		}else{
			/*CMM memory can be access by CPU directly*/
			/*TODO: should we prefetch?*/
			w_ptr=KERN.hiddens[idx].weights;
		}
#else  /*_CUDA*/
		w_ptr=KERN.hiddens[idx].weights;
#endif /*_CUDA*/
		for(jdx=0;jdx<(N*M);jdx++){
			temp_rnd=(DOUBLE) random() / RAND_MAX;
//			KERN.hiddens[idx].weights[jdx]=2.0*(temp_rnd-0.5)/sqrt((DOUBLE)M);
			w_ptr[jdx]=2.0*(temp_rnd-0.5)/sqrt((DOUBLE)M);
		}
#ifdef   _CUDA
		/*transfer back w_ptr to GPU*/
		scuda_ann_weight_transfer_C2G(kernel,idx,w_ptr,cudas);
		if(cudas->mem_model!=CUDA_MEM_CMM) FREE(w_ptr);
#endif /*_CUDA*/
	}
	/*randomly fill output weight*/
	N=KERN.output.n_neurons;
	M=KERN.output.n_inputs;
#ifdef   _CUDA
	if(cudas->mem_model!=CUDA_MEM_CMM){
		ALLOC(w_ptr,N*M,DOUBLE);
	}else{
		/*CMM memory can be access by CPU directly*/
		/*TODO: should we prefetch?*/
		w_ptr=KERN.output.weights;
	}
#else  /*_CUDA*/
	w_ptr=KERN.output.weights;
#endif /*_CUDA*/
	for(jdx=0;jdx<(N*M);jdx++){
		temp_rnd=(DOUBLE) random() / RAND_MAX;
//		KERN.output.weights[jdx]=2.0*(temp_rnd-0.5)/sqrt((DOUBLE)M);
		w_ptr[jdx]=2.0*(temp_rnd-0.5)/sqrt((DOUBLE)M);
	}
#ifdef   _CUDA
	/*transfer to GPU (KERN.n_hiddens -> output)*/
	scuda_ann_weight_transfer_C2G(kernel,KERN.n_hiddens,w_ptr,cudas);
	if(cudas->mem_model!=CUDA_MEM_CMM) FREE(w_ptr);
#endif /*_CUDA*/
#ifdef _MPI
}/*end of master-only*/
/*master -> slave(s)*/
if(stream!=0){/*slave(s)*/
	/*allocation - NO NEED TO REPORT*/
	ALLOC(kernel,1,kernel_ann);
	ann_kernel_allocate(kernel,n_in,n_hid,parameter,n_out);
}
for(idx=0;idx<n_hiddens;idx++){
	N=KERN.hiddens[idx].n_neurons;
	M=KERN.hiddens[idx].n_inputs;
	/*That one will broadcast CPU values over MPI*/
#ifndef  _CUDA
	MPI_Bcast(KERN.hiddens[idx].weights,M*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
#else  /*_CUDA*/
	if(cudas->mem_model!=CUDA_MEM_CMM){
		ALLOC(w_ptr,M*N,DOUBLE);/*CPU MPI ARRAY*/
	}else{
		w_ptr=KERN.hiddens[idx].weights;
	}
	/*master transfer weights*/
	if(stream==0) scuda_ann_weight_transfer_G2C(kernel,idx,&w_ptr,cudas);
	MPI_Bcast(w_ptr,M*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	/*now, everyone received CPU... put it back on GPU*/
	if(stream!=0){
		/*master already have it*/
		scuda_ann_weight_transfer_C2G(kernel,idx,w_ptr,cudas);
	}
	if(cudas->mem_model!=CUDA_MEM_CMM) FREE(w_ptr);
#endif /*_CUDA*/
}
N=KERN.output.n_neurons;
M=KERN.output.n_inputs;
/*same broadcast type as before*/
#ifndef  _CUDA
MPI_Bcast(KERN.output.weights,N*M,MPI_DOUBLE,0,MPI_COMM_WORLD);
#else  /*_CUDA*/
	if(cudas->mem_model!=CUDA_MEM_CMM){
		ALLOC(w_ptr,M*N,DOUBLE);/*CPU MPI ARRAY*/
	}else{
		w_ptr=KERN.output.weights;
	}
	/*master transfer weights*/
	if(stream==0)
		scuda_ann_weight_transfer_G2C(kernel,KERN.n_hiddens,&w_ptr,cudas);
	MPI_Bcast(w_ptr,M*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	/*now, everyone received CPU... put it back on GPU*/
	if(stream!=0){
		/*master already have it*/
		scuda_ann_weight_transfer_C2G(kernel,KERN.n_hiddens,w_ptr,cudas);
	}
	if(cudas->mem_model!=CUDA_MEM_CMM) FREE(w_ptr);
	MPI_Barrier(MPI_COMM_WORLD);/*everyone WAIT each other*/
#endif /*_CUDA*/
#endif /*_MPI*/
	return kernel;
}
/*---------------------*/
/*+++ OUTPUT KERNEL +++*/
/*---------------------*/
void ann_dump(kernel_ann *kernel,FILE *out){
	UINT idx;
	UINT jdx;
	UINT kdx;
	UINT N,M;
	DOUBLE *w_ptr;
#ifdef _CUDA
	cudastreams *cudas=_NN(get,cudas)();
#endif
#ifdef _MPI
	UINT n_streams,stream;
	_NN(get,mpi_tasks)(&n_streams);
	_NN(get,curr_mpi_task)(&stream);
	if (kernel==NULL) {
		NN_ERROR(stderr,"CAN'T SAVE KERNEL! kernel=NULL\n");
		return;
	}
if(stream==0){/*only master writes*/
#else /*_MPI*/
	if (kernel==NULL) {
		NN_ERROR(stderr,"CAN'T SAVE KERNEL! kernel=NULL\n");
		return;
	}
#endif /*_MPI*/
/*before dumping, we need to sync*/
	NN_WRITE(out,"[name] %s\n",KERN.name);
	NN_WRITE(out,"[param] %i",KERN.n_inputs);
	for(idx=0;idx<KERN.n_hiddens;idx++) 
		NN_WRITE(out," %i",KERN.hiddens[idx].n_neurons);
	NN_WRITE(out," %i\n",KERN.output.n_neurons);
	NN_WRITE(out,"[input] %i\n",KERN.n_inputs);
	for(idx=0;idx<KERN.n_hiddens;idx++) {
		N=KERN.hiddens[idx].n_neurons;
		M=KERN.hiddens[idx].n_inputs;
		/*prepare array*/
#ifdef   _CUDA
		if(cudas->mem_model!=CUDA_MEM_CMM){
			ALLOC(w_ptr,N*M,DOUBLE);
			scuda_ann_weight_transfer_G2C(kernel,idx,&w_ptr,cudas);
		}else{
			/*CMM memory can be access by CPU directly*/
			/*TODO: should we prefetch?*/
			w_ptr=KERN.hiddens[idx].weights;
		}
#else  /*_CUDA*/
		w_ptr=KERN.hiddens[idx].weights;
#endif /*_CUDA*/
		NN_WRITE(out,"[hidden %i] %i\n",idx+1,N);
		for(jdx=0;jdx<N;jdx++){
			NN_WRITE(out,"[neuron %i] %i\n",jdx+1,M);
			NN_WRITE(out,"%17.15f",w_ptr[_2D_IDX(M,jdx,0)]);
			for(kdx=1;kdx<M;kdx++)
				NN_WRITE(out," %17.15f",w_ptr[_2D_IDX(M,jdx,kdx)]);
			NN_WRITE(out,"\n");
		}
	}
	N=KERN.output.n_neurons;
	M=KERN.output.n_inputs;
#ifdef   _CUDA
	if(cudas->mem_model!=CUDA_MEM_CMM){
		FREE(w_ptr);
		ALLOC(w_ptr,N*M,DOUBLE);
		scuda_ann_weight_transfer_G2C(kernel,KERN.n_hiddens,&w_ptr,cudas);
	}else{
		/*CMM memory can be access by CPU directly*/
		/*TODO: should we prefetch?*/
		w_ptr=KERN.output.weights;
	}
#else  /*_CUDA*/
	w_ptr=KERN.output.weights;
#endif /*_CUDA*/
	NN_WRITE(out,"[output] %i\n",N);
	for(jdx=0;jdx<N;jdx++){
		NN_WRITE(out,"[neuron %i] %i\n",jdx+1,M);
		NN_WRITE(out,"%17.15f",w_ptr[_2D_IDX(M,jdx,0)]);
		for(kdx=1;kdx<M;kdx++)
			NN_WRITE(out," %17.15f",w_ptr[_2D_IDX(M,jdx,kdx)]);
		NN_WRITE(out,"\n");
	}
#ifdef _CUDA
	if(cudas->mem_model!=CUDA_MEM_CMM) FREE(w_ptr);
#endif
#ifdef _MPI
	/*end of master*/
	}
	MPI_Barrier(MPI_COMM_WORLD);/*everyone WAIT for master*/
#endif /*_MPI*/
}
/*-------------------------------------*/
/*+++ validate parameters of kernel +++*/
/* (to appease the static analysis) +++*/
/*-------------------------------------*/
BOOL ann_validate_kernel(kernel_ann *kernel){
	UINT idx;
	if(KERN.n_inputs<1) return FALSE;
	if(KERN.n_outputs<1) return FALSE;
	if(KERN.n_hiddens<1) return FALSE;
	if(KERN.in==NULL) return FALSE;
	for(idx=0;idx<KERN.n_hiddens;idx++){
		if(KERN.hiddens[idx].n_neurons<1) return FALSE;
		if(KERN.hiddens[idx].n_inputs<1) return FALSE;
		if(KERN.hiddens[idx].weights==NULL) return FALSE;
		if(KERN.hiddens[idx].vec==NULL) return FALSE;
	}
	if(KERN.output.n_neurons<1) return FALSE;
	if(KERN.output.n_inputs<1) return FALSE;
	if(KERN.output.weights==NULL) return FALSE;
	if(KERN.output.vec==NULL) return FALSE;
	return TRUE;
}
/*----------------------------*/
/*+++ activation functions +++*/
/*----------------------------*/
DOUBLE ann_act(DOUBLE x){
	return 2.0/(1.0+exp(-1.0*x))-1.0;
}
DOUBLE ann_dact(DOUBLE y){
	return -0.5*(y*y-1.0);
}
/*------------------------*/
/*+++ feed-forward run +++*/
/*------------------------*/
void ann_kernel_run(kernel_ann *kernel){
#ifdef   _CUDA
	/*the _NN(run,kernel) is now in charge of transfer(s)*/
	scuda_ann_forward(kernel,_NN(get,cudas)());
#else  /*_CUDA*/
	/*simple, one pass kernel*/
	UINT idx,jdx,M,N;
#if !defined (PBLAS) && !defined (SBLAS)
	UINT kdx;
#endif
#ifdef _MPI
	UINT n_streams,stream;
	UINT red,rem;
	_NN(get,mpi_tasks)(&n_streams);
	_NN(get,curr_mpi_task)(&stream);
#endif /*_MPI*/
	/*simple, one pass kernel*/
/*+++ I - input +++*/
	N=KERN.hiddens[0].n_neurons;
	M=KERN.hiddens[0].n_inputs;
#ifdef _MPI
	red=N/n_streams;
	rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
	cblas_dgemv(CblasRowMajor,CblasNoTrans,red,M,
		1.0,KERN.hiddens[0].weights+stream*M*red,M,KERN.in,1,0.,KERN.hiddens[0].vec+stream*red,1);
#define OP_ACT(ix) KERN.hiddens[0].vec[ix+stream*red]=ann_act(KERN.hiddens[0].vec[ix+stream*red])
	UNROLL_OMP_FOR(0,red,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
	/*do the remaining ops without MPI*/
	if(rem>0){
		cblas_dgemv(CblasRowMajor,CblasNoTrans,rem,M,
			1.0,KERN.hiddens[0].weights+n_streams*M*red,M,KERN.in,1,0.,KERN.hiddens[0].vec+n_streams*red,1);
#define OP_ACT(ix) KERN.hiddens[0].vec[ix+n_streams*red]=ann_act(KERN.hiddens[0].vec[ix+n_streams*red])
		UNROLL_OMP_FOR(0,rem,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
	}
#else /*_MPI*/
	cblas_dgemv(CblasRowMajor,CblasNoTrans,N,M,1.0,KERN.hiddens[0].weights,M,KERN.in,1,0.,KERN.hiddens[0].vec,1);
#define OP_ACT(ix) KERN.hiddens[0].vec[ix]=ann_act(KERN.hiddens[0].vec[ix])
	UNROLL_OMP_FOR(0,N,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
//DMP_DBG(KERN.hiddens[0].vec,N);
#endif /*_MPI*/
#elif defined(SBLAS)
	/*move the parallel mv into a series of vv*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<red;jdx++){
_HT;
		KERN.hiddens[0].vec[jdx+stream*red]=cblas_ddot(
		M,&(KERN.hiddens[0].weights[M*(jdx+stream*red)]),1,KERN.in,1);
		KERN.hiddens[0].vec[jdx+stream*red]=ann_act(KERN.hiddens[0].vec[jdx+stream*red]);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
if(rem>0){
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<rem;jdx++){
_HT;
		KERN.hiddens[0].vec[jdx+n_streams*red]=cblas_ddot(
		M,&(KERN.hiddens[0].weights[M*(jdx+n_streams*red)]),1,KERN.in,1);
		KERN.hiddens[0].vec[jdx+n_streams*red]=ann_act(KERN.hiddens[0].vec[jdx+n_streams*red]);
	}
}
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<N;jdx++){
_HT;
		KERN.hiddens[0].vec[jdx]=cblas_ddot(
		M,&(KERN.hiddens[0].weights[_2D_IDX(M,jdx,0)]),1,KERN.in,1);
		KERN.hiddens[0].vec[jdx]=ann_act(KERN.hiddens[0].vec[jdx]);
	}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<red;jdx++){
		KERN.hiddens[0].vec[jdx+stream*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[0].vec[jdx+stream*red]+=KERN.hiddens[0].weights[M*(jdx+stream*red)+ix]*KERN.in[ix]
		UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
		KERN.hiddens[0].vec[jdx+stream*red]=ann_act(KERN.hiddens[0].vec[jdx+stream*red]);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<rem;jdx++){
			KERN.hiddens[0].vec[jdx+n_streams*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[0].vec[jdx+n_streams*red]+=KERN.hiddens[0].weights[M*(jdx+n_streams*red)+ix]*KERN.in[ix]
			UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
			KERN.hiddens[0].vec[jdx+n_streams*red]=ann_act(KERN.hiddens[0].vec[jdx+n_streams*red]);
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<N;jdx++){
		KERN.hiddens[0].vec[jdx]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[0].vec[jdx]+=KERN.hiddens[0].weights[_2D_IDX(M,jdx,ix)]*KERN.in[ix]
		UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
		KERN.hiddens[0].vec[jdx]=ann_act(KERN.hiddens[0].vec[jdx]);
	}
#endif /*_MPI*/
#endif /*PBLAS*/
/*+++ II - hiddens +++*/
	for(idx=1;idx<KERN.n_hiddens;idx++){
		N=KERN.hiddens[idx].n_neurons;
		M=KERN.hiddens[idx].n_inputs;
#ifdef _MPI
		red=N/n_streams;
		rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
		cblas_dgemv(CblasRowMajor,CblasNoTrans,red,M,
		1.0,KERN.hiddens[idx].weights+stream*M*red,M,KERN.hiddens[idx-1].vec,1,0.,KERN.hiddens[idx].vec+stream*red,1);
#define OP_ACT(ix) KERN.hiddens[idx].vec[ix+stream*red]=ann_act(KERN.hiddens[idx].vec[ix+stream*red])
		UNROLL_OMP_FOR(0,red,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
		MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rem>0){
			cblas_dgemv(CblasRowMajor,CblasNoTrans,rem,M,
			1.0,KERN.hiddens[idx].weights+n_streams*M*red,M,KERN.hiddens[idx-1].vec,1,0.,KERN.hiddens[idx].vec+n_streams*red,1);
#define OP_ACT(ix) KERN.hiddens[idx].vec[ix+n_streams*red]=ann_act(KERN.hiddens[idx].vec[ix+n_streams*red])
			UNROLL_OMP_FOR(0,rem,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
		}
#else /*_MPI*/
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N,M,
			1.0,KERN.hiddens[idx].weights,M,KERN.hiddens[idx-1].vec,1,0.,KERN.hiddens[idx].vec,1);
#define OP_ACT(ix) KERN.hiddens[idx].vec[ix]=ann_act(KERN.hiddens[idx].vec[ix])
		UNROLL_OMP_FOR(0,N,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
#endif /*_MPI*/
#elif defined(SBLAS)
		/*move the parallel mv into a series of vv*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
		for(jdx=0;jdx<red;jdx++){
_HT;
			KERN.hiddens[idx].vec[jdx+stream*red]=cblas_ddot(
			M,&(KERN.hiddens[idx].weights[M*(jdx+stream*red)]),1,KERN.hiddens[idx-1].vec,1);
			KERN.hiddens[idx].vec[jdx+stream*red]=ann_act(KERN.hiddens[idx].vec[jdx+stream*red]);
		}
		MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rem>0){
#pragma omp parallel for private(jdx) _NT
			for(jdx=0;jdx<rem;jdx++){
_HT;
				KERN.hiddens[idx].vec[jdx+n_streams*red]=cblas_ddot(
				M,&(KERN.hiddens[idx].weights[M*(jdx+n_streams*red)]),1,KERN.hiddens[idx-1].vec,1);
				KERN.hiddens[idx].vec[jdx+n_streams*red]=ann_act(KERN.hiddens[idx].vec[jdx+n_streams*red]);
			}
		}
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
		for(jdx=0;jdx<N;jdx++){
_HT;
			KERN.hiddens[idx].vec[jdx]=cblas_ddot(
			M,&(KERN.hiddens[idx].weights[_2D_IDX(M,jdx,0)]),1,KERN.hiddens[idx-1].vec,1);
			KERN.hiddens[idx].vec[jdx]=ann_act(KERN.hiddens[idx].vec[jdx]);
		}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
		#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<red;jdx++){
			KERN.hiddens[idx].vec[jdx+stream*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[idx].vec[jdx+stream*red]+=KERN.hiddens[idx].weights[M*(jdx+stream*red)+ix]*KERN.hiddens[idx-1].vec[ix]
			UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
			KERN.hiddens[idx].vec[jdx+stream*red]=ann_act(KERN.hiddens[idx].vec[jdx+stream*red]);
		}
		MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rem>0){
#pragma omp parallel for private(jdx) _NT
			for(jdx=0;jdx<rem;jdx++){
				KERN.hiddens[idx].vec[jdx+n_streams*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[idx].vec[jdx+n_streams*red]+=KERN.hiddens[idx].weights[M*(jdx+n_streams*red)+ix]*KERN.hiddens[idx-1].vec[ix]
				UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
				KERN.hiddens[idx].vec[jdx+n_streams*red]=ann_act(KERN.hiddens[idx].vec[jdx+n_streams*red]);
			}
		}
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<N;jdx++){
			KERN.hiddens[idx].vec[jdx]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[idx].vec[jdx]+=KERN.hiddens[idx].weights[_2D_IDX(M,jdx,ix)]*KERN.hiddens[idx-1].vec[ix]
			UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
			KERN.hiddens[idx].vec[jdx]=ann_act(KERN.hiddens[idx].vec[jdx]);
		}
#endif /*_MPI*/
#endif /*PBLAS*/
	}
/*+++ III - output +++*/
	N=KERN.output.n_neurons;
	M=KERN.output.n_inputs;
#ifdef _MPI
	red=N/n_streams;
	rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
	cblas_dgemv(CblasRowMajor,CblasNoTrans,red,M,
	1.0,KERN.output.weights+stream*M*red,M,KERN.hiddens[KERN.n_hiddens-1].vec,1,0.,KERN.output.vec+stream*red,1);
#define OP_ACT(ix) KERN.output.vec[ix+stream*red]=ann_act(KERN.output.vec[ix+stream*red])
	UNROLL_OMP_FOR(0,red,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
		cblas_dgemv(CblasRowMajor,CblasNoTrans,rem,M,
		1.0,KERN.output.weights+n_streams*M*red,M,KERN.hiddens[KERN.n_hiddens-1].vec,1,0.,KERN.output.vec+n_streams*red,1);
#define OP_ACT(ix) KERN.output.vec[ix+n_streams*red]=ann_act(KERN.output.vec[ix+n_streams*red])
		UNROLL_OMP_FOR(0,rem,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
	}
#else /*_MPI*/
	/*serial dgemv (no thread support here)*/
	cblas_dgemv(CblasRowMajor,CblasNoTrans,N,M,
		1.0,KERN.output.weights,M,KERN.hiddens[KERN.n_hiddens-1].vec,1,0.,KERN.output.vec,1);
#define OP_ACT(ix) KERN.output.vec[ix]=ann_act(KERN.output.vec[ix])
	UNROLL_OMP_FOR(0,N,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
#endif /*_MPI*/
#elif defined(SBLAS)
	/*move the mv into a series of vv*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<red;jdx++){
_HT;
		KERN.output.vec[jdx+stream*red]=cblas_ddot(
		M,&(KERN.output.weights[M*(jdx+stream*red)]),1,KERN.hiddens[KERN.n_hiddens-1].vec,1);
		KERN.output.vec[jdx+stream*red]=ann_act(KERN.output.vec[jdx+stream*red]);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(jdx) _NT
		for(jdx=0;jdx<rem;jdx++){
_HT;
			KERN.output.vec[jdx+n_streams*red]=cblas_ddot(
			M,&(KERN.output.weights[M*(jdx+n_streams*red)]),1,KERN.hiddens[KERN.n_hiddens-1].vec,1);
			KERN.output.vec[jdx+n_streams*red]=ann_act(KERN.output.vec[jdx+n_streams*red]);
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<N;jdx++){
_HT;
		KERN.output.vec[jdx]=cblas_ddot(
		M,&(KERN.output.weights[_2D_IDX(M,jdx,0)]),1,KERN.hiddens[KERN.n_hiddens-1].vec,1);
		KERN.output.vec[jdx]=ann_act(KERN.output.vec[jdx]);
	}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<red;jdx++){
		KERN.output.vec[jdx+stream*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.output.vec[jdx+stream*red]+=KERN.output.weights[M*(jdx+stream*red)+ix]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
		UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
		KERN.output.vec[jdx+stream*red]=ann_act(KERN.output.vec[jdx+stream*red]);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<rem;jdx++){
			KERN.output.vec[jdx+n_streams*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.output.vec[jdx+n_streams*red]+=KERN.output.weights[M*(jdx+n_streams*red)+ix]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
			UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
			KERN.output.vec[jdx+n_streams*red]=ann_act(KERN.output.vec[jdx+n_streams*red]);
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<N;jdx++){
		KERN.output.vec[jdx]=0.;/*TRAP*/
#define OP_WI(ix) KERN.output.vec[jdx]+=KERN.output.weights[_2D_IDX(M,jdx,ix)]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
		UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
		KERN.output.vec[jdx]=ann_act(KERN.output.vec[jdx]);
	}
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
//	MPI_Barrier(MPI_COMM_WORLD);/*WAIT FOR ALL TASKS BEFORE LEAVING*/
#endif
	/*done*/
#endif /*_CUDA*/
}
/*-------------------------------*/
/*+++ Train Error Calculation +++*/
/*-------------------------------*/
DOUBLE ann_kernel_train_error(kernel_ann *kernel, const DOUBLE *train){
	DOUBLE Ep=0.;
	UINT idx,N;
#ifdef _MPI
	UINT red,rem;
	UINT n_streams,stream;
	_NN(get,mpi_tasks)(&n_streams);
	_NN(get,curr_mpi_task)(&stream);
#endif /*_MPI*/
	N=KERN.n_outputs;
#ifdef _MPI
	red=N/n_streams;
	rem=N%n_streams;
#pragma omp parallel for private(idx) reduction(+:Ep) _NT
	for(idx=0;idx<red;idx++)
			Ep+=(train[idx+stream*red]-KERN.output.vec[idx+stream*red])
			   *(train[idx+stream*red]-KERN.output.vec[idx+stream*red]);
	MPI_Allreduce(MPI_IN_PLACE,&Ep,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if(rem>0) {
		for(idx=0;idx<rem;idx++) 
			Ep+=(train[idx+n_streams*red]-KERN.output.vec[idx+n_streams*red])
			   *(train[idx+n_streams*red]-KERN.output.vec[idx+n_streams*red]);
	}
#else /*_MPI*/
#pragma omp parallel for private(idx) reduction(+:Ep) _NT
	for(idx=0;idx<N;idx++) Ep+=(train[idx]-KERN.output.vec[idx])*(train[idx]-KERN.output.vec[idx]);
#endif /*_MPI*/
	Ep*=0.5;
	return Ep;
}
/*------------------------*/
/*+++ Calculate deltas +++*/
/*------------------------*/
void ann_kernel_train_delta(kernel_ann *kernel,const DOUBLE *train, DOUBLE **delta_ptr){
#if !defined (PBLAS) && !defined (SBLAS)
        UINT kdx;
#endif
	UINT N,M;
        UINT idx, jdx;
#ifdef _MPI
	UINT red, rem;
	UINT n_streams,stream;
	_NN(get,mpi_tasks)(&n_streams);
	_NN(get,curr_mpi_task)(&stream);
#endif /*_MPI*/
/*^^^ output*/
	N=KERN.output.n_neurons;
#ifdef _MPI
	red=N/n_streams;
	rem=N%n_streams;
#define OP_DELTA(ix) delta_ptr[KERN.n_hiddens][ix+stream*red]=\
	(train[ix+stream*red]-KERN.output.vec[ix+stream*red])*ann_dact(KERN.output.vec[ix+stream*red])
	UNROLL_OMP_FOR(0,red,ANN_UNROLL,DELTA,idx);
#undef OP_DELTA
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[KERN.n_hiddens],red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#define OP_DELTA(ix) delta_ptr[KERN.n_hiddens][ix+n_streams*red]=\
	(train[ix+n_streams*red]-KERN.output.vec[ix+n_streams*red])*ann_dact(KERN.output.vec[ix+n_streams*red])
		UNROLL_OMP_FOR(0,rem,ANN_UNROLL,DELTA,idx);
#undef OP_DELTA
	}
#else /*_MPI*/
#define OP_DELTA(ix) delta_ptr[KERN.n_hiddens][ix]=(train[ix]-KERN.output.vec[ix])*ann_dact(KERN.output.vec[ix])
	UNROLL_OMP_FOR(0,N,ANN_UNROLL,DELTA,idx);
#undef OP_DELTA
#endif /*_MPI*/
/*^^^ output to hidden*/
	N=KERN.output.n_neurons;
	M=KERN.output.n_inputs;
#ifdef _MPI
	red=M/n_streams;
	rem=M%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
	cblas_dgemv(CblasRowMajor,CblasTrans,N,red,
		1.0,KERN.output.weights+stream*red,M,delta_ptr[KERN.n_hiddens],1,0.,delta_ptr[KERN.n_hiddens-1]+stream*red,1);
#define OP_DACT(ix) delta_ptr[KERN.n_hiddens-1][ix+stream*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[ix+stream*red])
	UNROLL_OMP_FOR(0,red,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[KERN.n_hiddens-1],red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
		cblas_dgemv(CblasRowMajor,CblasTrans,N,rem,
			1.0,KERN.output.weights+n_streams*red,M,delta_ptr[KERN.n_hiddens],1,0.,delta_ptr[KERN.n_hiddens-1]+n_streams*red,1);
#define OP_DACT(ix) delta_ptr[KERN.n_hiddens-1][ix+n_streams*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[ix+n_streams*red])
		UNROLL_OMP_FOR(0,rem,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
	}
#else /*_MPI*/
	/*! transposed*/
	cblas_dgemv(CblasRowMajor,CblasTrans,N,M,1.0,KERN.output.weights,M,delta_ptr[KERN.n_hiddens],1,0.,delta_ptr[KERN.n_hiddens-1],1);
#define OP_DACT(ix) delta_ptr[KERN.n_hiddens-1][ix]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[ix])
	UNROLL_OMP_FOR(0,M,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
#endif /*_MPI*/
#elif defined(SBLAS)
	/*move the mv into a series of vv*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<red;jdx++){
_HT;
		delta_ptr[KERN.n_hiddens-1][jdx+stream*red]=cblas_ddot(
			N,&(KERN.output.weights[jdx+stream*red]),M,delta_ptr[KERN.n_hiddens],1);
		delta_ptr[KERN.n_hiddens-1][jdx+stream*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx+stream*red]);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[KERN.n_hiddens-1],red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(jdx) _NT
		for(jdx=0;jdx<rem;jdx++){
_HT;
			delta_ptr[KERN.n_hiddens-1][jdx+n_streams*red]=cblas_ddot(
				N,&(KERN.output.weights[jdx+n_streams*red]),M,delta_ptr[KERN.n_hiddens],1);
			delta_ptr[KERN.n_hiddens-1][jdx+n_streams*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx+n_streams*red]);
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<M;jdx++){
_HT;
		/*since the matrix is transposed incX is the matrix stride!*/
		delta_ptr[KERN.n_hiddens-1][jdx]=cblas_ddot(
		N,&(KERN.output.weights[jdx]),M,&(delta_ptr[KERN.n_hiddens][0]),1);
		delta_ptr[KERN.n_hiddens-1][jdx]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx]);
	}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<red;jdx++){
#define OP_WD(ix) delta_ptr[KERN.n_hiddens-1][jdx+stream*red]+=\
	KERN.output.weights[_2D_IDX(M,ix,jdx+stream*red)]*delta_ptr[KERN.n_hiddens][ix]
		UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
		delta_ptr[KERN.n_hiddens-1][jdx+stream*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx+stream*red]);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[KERN.n_hiddens-1],red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<rem;jdx++){
#define OP_WD(ix) delta_ptr[KERN.n_hiddens-1][jdx+n_streams*red]+=\
	KERN.output.weights[_2D_IDX(M,ix,jdx+n_streams*red)]*delta_ptr[KERN.n_hiddens][ix]
			UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
			delta_ptr[KERN.n_hiddens-1][jdx+n_streams*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx+n_streams*red]);
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<M;jdx++){
#define OP_WD(ix) delta_ptr[KERN.n_hiddens-1][jdx]+=KERN.output.weights[_2D_IDX(M,ix,jdx)]*delta_ptr[KERN.n_hiddens][ix]
		UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
		delta_ptr[KERN.n_hiddens-1][jdx]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx]);
	}
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
#endif /*_MPI*/
/*^^^ hidden to hidden (if any)*/
	if(KERN.n_hiddens>1){
		for(idx=(KERN.n_hiddens-2);idx>0;idx--){
			N=KERN.hiddens[idx+1].n_neurons;
			M=KERN.hiddens[idx+1].n_inputs;
#ifdef _MPI
			red=M/n_streams;
			rem=M%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
			cblas_dgemv(CblasRowMajor,CblasTrans,N,red,
				 1.0,KERN.hiddens[idx+1].weights+stream*red,M,delta_ptr[idx+1],1,0.,delta_ptr[idx]+stream*red,1);
#define OP_DACT(ix) delta_ptr[idx][ix+stream*red]*=ann_dact(KERN.hiddens[idx].vec[ix+stream*red])
			UNROLL_OMP_FOR(0,red,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
			MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[idx],red,MPI_DOUBLE,MPI_COMM_WORLD);
			if(rem>0){
				cblas_dgemv(CblasRowMajor,CblasTrans,N,rem,
					1.0,KERN.hiddens[idx+1].weights+n_streams*red,M,delta_ptr[idx+1],1,0.,delta_ptr[idx]+n_streams*red,1);
#define OP_DACT(ix) delta_ptr[idx][ix+n_streams*red]*=ann_dact(KERN.hiddens[idx].vec[ix+n_streams*red])
				UNROLL_OMP_FOR(0,rem,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
			}
#else /*_MPI*/
			/*! transposed*/
			cblas_dgemv(CblasRowMajor,CblasTrans,N,M,1.0,KERN.hiddens[idx+1].weights,M,delta_ptr[idx+1],1,0.,delta_ptr[idx],1);
#define OP_DACT(ix) delta_ptr[idx][ix]*=ann_dact(KERN.hiddens[idx].vec[ix])
			UNROLL_OMP_FOR(0,M,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
#endif /*_MPI*/
#elif defined(SBLAS)
			/*move the mv into a series of vv*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
			for(jdx=0;jdx<red;jdx++){
_HT;
				/*since the matrix is transposed incX is the matrix stride!*/
				delta_ptr[idx][jdx+stream*red]=cblas_ddot(
				N,&(KERN.hiddens[idx+1].weights[jdx+stream*red]),M,delta_ptr[idx+1],1);
				delta_ptr[idx][jdx+stream*red]*=ann_dact(KERN.hiddens[idx].vec[jdx+stream*red]);
			}
			MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[idx],red,MPI_DOUBLE,MPI_COMM_WORLD);
			if(rem>0){
#pragma omp parallel for private(jdx) _NT
				for(jdx=0;jdx<rem;jdx++){
_HT;
					/*since the matrix is transposed incX is the matrix stride!*/
					delta_ptr[idx][jdx+n_streams*red]=cblas_ddot(
					N,&(KERN.hiddens[idx+1].weights[jdx+n_streams*red]),M,delta_ptr[idx+1],1);
					delta_ptr[idx][jdx+n_streams*red]*=ann_dact(KERN.hiddens[idx].vec[jdx+n_streams*red]);
				}
			}
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
			for(jdx=0;jdx<M;jdx++){
_HT;
				/*since the matrix is transposed incX is the matrix stride!*/
				delta_ptr[idx][jdx]=cblas_ddot(
				N,&(KERN.hiddens[idx+1].weights[_2D_IDX(M,0,jdx)]),M,delta_ptr[idx+1],1);
				delta_ptr[idx][jdx]*=ann_dact(KERN.hiddens[idx].vec[jdx]);
			}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
			for(jdx=0;jdx<red;jdx++){
#define OP_WD(ix) delta_ptr[idx][jdx+stream*red]+=KERN.hiddens[idx+1].weights[_2D_IDX(M,ix,jdx+stream*red)]*delta_ptr[idx+1][ix]
				UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
				delta_ptr[idx][jdx+stream*red]*=ann_dact(KERN.hiddens[idx].vec[jdx+stream*red]);
			}
			MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[idx],red,MPI_DOUBLE,MPI_COMM_WORLD);
			if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
				for(jdx=0;jdx<rem;jdx++){
#define OP_WD(ix) delta_ptr[idx][jdx+n_streams*red]+=KERN.hiddens[idx+1].weights[_2D_IDX(M,ix,jdx+n_streams*red)]*delta_ptr[idx+1][ix]
					UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
					delta_ptr[idx][jdx+n_streams*red]*=ann_dact(KERN.hiddens[idx].vec[jdx+n_streams*red]);
				}
			}
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
			for(jdx=0;jdx<M;jdx++){
#define OP_WD(ix) delta_ptr[idx][jdx]+=KERN.hiddens[idx+1].weights[_2D_IDX(M,ix,jdx)]*delta_ptr[idx+1][ix]
				UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
				delta_ptr[idx][jdx]*=ann_dact(KERN.hiddens[idx].vec[jdx]);
			}
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
#endif /*_MPI*/
		}
		/*add zero*/
		N=KERN.hiddens[1].n_neurons;
		M=KERN.hiddens[1].n_inputs;
#ifdef _MPI
		red=M/n_streams;
		rem=M%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
		/*! transposed*/
#ifdef _MPI
		cblas_dgemv(CblasRowMajor,CblasTrans,N,red,
			1.0,KERN.hiddens[1].weights+stream*red,M,delta_ptr[1],1,0.,delta_ptr[0]+stream*red,1);
#define OP_DACT(ix) delta_ptr[0][ix+stream*red]*=ann_dact(KERN.hiddens[0].vec[ix+stream*red])
		UNROLL_OMP_FOR(0,red,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
		MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[0],red,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rem>0){
			cblas_dgemv(CblasRowMajor,CblasTrans,N,rem,
			1.0,KERN.hiddens[1].weights+n_streams*red,M,delta_ptr[1],1,0.,delta_ptr[0]+n_streams*red,1);
#define OP_DACT(ix) delta_ptr[0][ix+n_streams*red]*=ann_dact(KERN.hiddens[0].vec[ix+n_streams*red])
			UNROLL_OMP_FOR(0,rem,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
		}
#else /*_MPI*/
		cblas_dgemv(CblasRowMajor,CblasTrans,N,M,
		1.0,KERN.hiddens[1].weights,M,delta_ptr[1],1,0.,delta_ptr[0],1);
#define OP_DACT(ix) delta_ptr[0][ix]*=ann_dact(KERN.hiddens[0].vec[ix])
		UNROLL_OMP_FOR(0,M,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
#endif /*_MPI*/
#elif defined(SBLAS)
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
		for(jdx=0;jdx<red;jdx++){
_HT;
			/*since the matrix is transposed incX is the matrix stride!*/
			delta_ptr[0][jdx+stream*red]=cblas_ddot(
			N,&(KERN.hiddens[1].weights[_2D_IDX(M,0,jdx+stream*red)]),N,&(delta_ptr[1][0]),1);
			delta_ptr[0][jdx+stream*red]*=ann_dact(KERN.hiddens[0].vec[jdx+stream*red]);
		}
		MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[0],red,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rem>0){
#pragma omp parallel for private(jdx) _NT
			for(jdx=0;jdx<rem;jdx++){
_HT;
				/*since the matrix is transposed incX is the matrix stride!*/
				delta_ptr[0][jdx+n_streams*red]=cblas_ddot(
				N,&(KERN.hiddens[1].weights[_2D_IDX(M,0,jdx+n_streams*red)]),N,&(delta_ptr[1][0]),1);
				delta_ptr[0][jdx+n_streams*red]*=ann_dact(KERN.hiddens[0].vec[jdx+n_streams*red]);
			}
		}
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
		for(jdx=0;jdx<M;jdx++){
_HT;
			/*since the matrix is transposed incX is the matrix stride!*/
			delta_ptr[0][jdx]=cblas_ddot(
			N,&(KERN.hiddens[1].weights[_2D_IDX(M,0,jdx)]),M,&(delta_ptr[1][0]),1);
			delta_ptr[0][jdx]*=ann_dact(KERN.hiddens[0].vec[jdx]);
		}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<red;jdx++){
#define OP_WD(ix) delta_ptr[0][jdx+stream*red]+=KERN.hiddens[1].weights[_2D_IDX(M,ix,jdx+stream*red)]*delta_ptr[1][ix]
			UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
			delta_ptr[0][jdx+stream*red]*=ann_dact(KERN.hiddens[0].vec[jdx+stream*red]);
		}
		MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[0],red,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
			for(jdx=0;jdx<rem;jdx++){
#define OP_WD(ix) delta_ptr[0][jdx+n_streams*red]+=KERN.hiddens[1].weights[_2D_IDX(M,ix,jdx+n_streams*red)]*delta_ptr[1][ix]
				UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
				delta_ptr[0][jdx+n_streams*red]*=ann_dact(KERN.hiddens[0].vec[jdx+n_streams*red]);
			}
		}
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<M;jdx++){
#define OP_WD(ix) delta_ptr[0][jdx]+=KERN.hiddens[1].weights[_2D_IDX(M,ix,jdx)]*delta_ptr[1][ix]
			UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
			delta_ptr[0][jdx]*=ann_dact(KERN.hiddens[0].vec[jdx]);
		}
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
#endif /*_MPI*/
	}
}
/*------------------------*/
/*+++ back-propagation +++*/
/*------------------------*/
DOUBLE ann_kernel_train(kernel_ann *kernel,const DOUBLE *train){
#define LEARN_RATE 0.01
#if !defined (PBLAS) && !defined (SBLAS)
	UINT kdx;
#endif
	UINT N,M;
	DOUBLE **delta_ptr;
	UINT idx;
#ifndef PBLAS
	UINT jdx;
#endif
	DOUBLE Ep =0.;
	DOUBLE Epr=0.;
#ifdef _MPI
	UINT red, rem;
	UINT n_streams,stream;
	_NN(get,mpi_tasks)(&n_streams);
	_NN(get,curr_mpi_task)(&stream);
#endif /*_MPI*/
	/*keep a track of mem*/
	UINT64 allocate=0.;
	ALLOC_REPORT(delta_ptr,KERN.n_hiddens+1,DOUBLE *,allocate);/*+1 for OUTPUT*/
	ALLOC_REPORT(delta_ptr[KERN.n_hiddens],KERN.n_outputs,DOUBLE,allocate);
	for(idx=0;idx<KERN.n_hiddens;idx++)
		ALLOC_REPORT(delta_ptr[idx],KERN.hiddens[idx].n_neurons,DOUBLE,allocate);
/*+++ I - forward is _supposed_ to be done already +++*/
	Ep=ann_kernel_train_error(kernel,train);
//	NN_DBG(stdout,"TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - calculate deltas +++*/
	ann_kernel_train_delta(kernel,train,delta_ptr);
/*+++ III - back propagation +++*/
/*^^^ output*/
	N=KERN.output.n_neurons;
	M=KERN.output.n_inputs;
#ifdef _MPI
	red=N/n_streams;
	rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
	cblas_dger(CblasRowMajor,red,M,LEARN_RATE,delta_ptr[KERN.n_hiddens]+stream*red,
	1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.output.weights+stream*M*red,M);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
		cblas_dger(CblasRowMajor,rem,M,LEARN_RATE,delta_ptr[KERN.n_hiddens]+n_streams*red,
		1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.output.weights+n_streams*M*red,M);
	}
#else /*_MPI*/
	cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[KERN.n_hiddens],1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.output.weights,M);
#endif /*_MPI*/
#elif defined(SBLAS)
	/*move the ger into a series of axpy*/
#ifdef _MPI
#pragma omp parallel for private(idx) _NT
	for(idx=0;idx<red;idx++){
_HT;
		cblas_daxpy(
		M,delta_ptr[KERN.n_hiddens][idx+stream*red]*LEARN_RATE,
		&(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,&(KERN.output.weights[_2D_IDX(M,idx+stream*red,0)]),1);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(idx) _NT
		for(idx=0;idx<rem;idx++){
_HT;
			cblas_daxpy(
			M,delta_ptr[KERN.n_hiddens][idx+n_streams*red]*LEARN_RATE,
			&(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,
			&(KERN.output.weights[_2D_IDX(M,idx+n_streams*red,0)]),1);
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(idx) _NT
	for(idx=0;idx<N;idx++){
_HT;
		cblas_daxpy(
		M,delta_ptr[KERN.n_hiddens][idx]*LEARN_RATE,
		&(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,
		&(KERN.output.weights[_2D_IDX(M,idx,0)]),1);
	}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(idx,jdx) _NT
	for(idx=0;idx<red;idx++){
#define OP_DH(ix) KERN.output.weights[_2D_IDX(M,idx+stream*red,ix)]+=\
	LEARN_RATE*delta_ptr[KERN.n_hiddens][idx+stream*red]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
		UNROLL_FOR(0,M,ANN_UNROLL,DH,jdx);
#undef OP_DH
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(idx,jdx) _NT
		for(idx=0;idx<rem;idx++){
#define OP_DH(ix) KERN.output.weights[_2D_IDX(M,idx+n_streams*red,ix)]+=\
	LEARN_RATE*delta_ptr[KERN.n_hiddens][idx+n_streams*red]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
			UNROLL_FOR(0,M,ANN_UNROLL,DH,jdx);
#undef OP_DH
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(idx,jdx) _NT
	for(idx=0;idx<N;idx++){
#define OP_DH(ix) KERN.output.weights[_2D_IDX(M,idx,ix)]+=\
	LEARN_RATE*delta_ptr[KERN.n_hiddens][idx]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
		UNROLL_FOR(0,M,ANN_UNROLL,DH,jdx);
#undef OP_DH
	}
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
//	MPI_Barrier(MPI_COMM_WORLD);//WAIT FOR ALL TASKS
#endif /*_MPI*/
/*^^^ hiddens*/
	for(idx=(KERN.n_hiddens-1);idx>0;idx--){
		N=KERN.hiddens[idx].n_neurons;
		M=KERN.hiddens[idx].n_inputs;
#ifdef _MPI
		red=N/n_streams;
		rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
		cblas_dger(CblasRowMajor,red,M,LEARN_RATE,
		delta_ptr[idx]+stream*red,1,
		KERN.hiddens[idx-1].vec,1,
		KERN.hiddens[idx].weights+stream*M*red,M);
		MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rem>0){
			cblas_dger(CblasRowMajor,rem,M,LEARN_RATE,
				delta_ptr[idx]+n_streams*red,1,
				KERN.hiddens[idx-1].vec,1,
				KERN.hiddens[idx].weights+n_streams*M*red,M);
		}
#else /*_MPI*/
		cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[idx],1,KERN.hiddens[idx-1].vec,1,KERN.hiddens[idx].weights,M);
#endif /*_MPI*/
#elif defined(SBLAS)
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
		for(jdx=0;jdx<red;jdx++){
_HT;
			cblas_daxpy(M,delta_ptr[idx][jdx+stream*red]*LEARN_RATE,
				&(KERN.hiddens[idx-1].vec[0]),1,&(KERN.hiddens[idx].weights[_2D_IDX(M,jdx+stream*red,0)]),1);
		}
		MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rem>0){
#pragma omp parallel for private(jdx) _NT
			for(jdx=0;jdx<rem;jdx++){
_HT;
				cblas_daxpy(M,delta_ptr[idx][jdx+n_streams*red]*LEARN_RATE,
				&(KERN.hiddens[idx-1].vec[0]),1,&(KERN.hiddens[idx].weights[_2D_IDX(M,jdx+n_streams*red,0)]),1);
			}
		}
#else /*_MPI*/
		/*move the ger into a series of axpy*/
#pragma omp parallel for private(jdx) _NT
		for(jdx=0;jdx<N;jdx++){
_HT;
			cblas_daxpy(M,delta_ptr[idx][jdx]*LEARN_RATE,
				&(KERN.hiddens[idx-1].vec[0]),1,&(KERN.hiddens[idx].weights[_2D_IDX(M,jdx,0)]),1);
		}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<red;jdx++){
#define OP_DH(ix) KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx+stream*red,ix)]+=\
	LEARN_RATE*delta_ptr[idx][jdx+stream*red]*KERN.hiddens[idx-1].vec[ix]
			UNROLL_FOR(0,M,ANN_UNROLL,DH,kdx);
#undef OP_DH
		}
		MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
			for(jdx=0;jdx<rem;jdx++){
#define OP_DH(ix) KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx+n_streams*red,ix)]+=\
	LEARN_RATE*delta_ptr[idx][jdx+n_streams*red]*KERN.hiddens[idx-1].vec[ix]
				UNROLL_FOR(0,M,ANN_UNROLL,DH,kdx);
#undef OP_DH
			}
		}
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<N;jdx++){
#define OP_DH(ix) KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,ix)]+=\
	LEARN_RATE*delta_ptr[idx][jdx]*KERN.hiddens[idx-1].vec[ix]
			UNROLL_FOR(0,M,ANN_UNROLL,DH,kdx);
#undef OP_DH
		}
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
//	MPI_Barrier(MPI_COMM_WORLD);/*WAIT FOR ALL TASKS*/
#endif /*_MPI*/
	}
	/*add zero*/
	N=KERN.hiddens[0].n_neurons;
	M=KERN.hiddens[0].n_inputs;
#ifdef _MPI
	red=N/n_streams;
	rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
	cblas_dger(CblasRowMajor,red,M,LEARN_RATE,delta_ptr[0]+stream*red,1,KERN.in,1,KERN.hiddens[0].weights+stream*M*red,M);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
		cblas_dger(CblasRowMajor,rem,M,
			LEARN_RATE,delta_ptr[0]+n_streams*red,1,
			KERN.in,1,
			KERN.hiddens[0].weights+n_streams*M*red,M);
	}
#else /*_MPI*/
	cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[0],1,KERN.in,1,KERN.hiddens[0].weights,M);
#endif /*_MPI*/
#elif defined(SBLAS)
	/*move the ger into a series of axpy*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<red;jdx++){
		cblas_daxpy(M,LEARN_RATE*delta_ptr[0][jdx+stream*red],KERN.in,1,&(KERN.hiddens[0].weights[_2D_IDX(M,jdx+stream*red,0)]),1);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
		for(jdx=0;jdx<rem;jdx++){
			cblas_daxpy(M,LEARN_RATE*delta_ptr[0][jdx+n_streams*red],
			KERN.in,1,&(KERN.hiddens[0].weights[_2D_IDX(M,jdx+n_streams*red,0)]),1);
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<N;jdx++){
		cblas_daxpy(M,LEARN_RATE*delta_ptr[0][jdx],KERN.in,1,&(KERN.hiddens[0].weights[_2D_IDX(M,jdx,0)]),1);
	}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<red;jdx++){
#define OP_DI(ix) KERN.hiddens[0].weights[_2D_IDX(M,jdx+stream*red,ix)]+=LEARN_RATE*delta_ptr[0][jdx+stream*red]*KERN.in[ix]
		UNROLL_FOR(0,M,ANN_UNROLL,DI,kdx);
#undef OP_DI
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<rem;jdx++){
#define OP_DI(ix) KERN.hiddens[0].weights[_2D_IDX(M,jdx+n_streams*red,ix)]+=LEARN_RATE*delta_ptr[0][jdx+n_streams*red]*KERN.in[ix]
			UNROLL_FOR(0,M,ANN_UNROLL,DI,kdx);
#undef OP_DI
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<N;jdx++){
#define OP_DI(ix) KERN.hiddens[0].weights[_2D_IDX(M,jdx,ix)]+=LEARN_RATE*delta_ptr[0][jdx]*KERN.in[ix]
		UNROLL_FOR(0,M,ANN_UNROLL,DI,kdx);
#undef OP_DI
	}
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
//	MPI_Barrier(MPI_COMM_WORLD);//WAIT FOR ALL TASKS
#endif /*_MPI*/
/*+++ IV - update error +++*/
	ann_kernel_run(kernel);
	Epr=ann_kernel_train_error(kernel,train);
//	NN_DBG(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ V - cleanup +++*/
	for(idx=0;idx<(KERN.n_hiddens+1);idx++){
		FREE(delta_ptr[idx]);
		delta_ptr[idx]=NULL;
	}
	FREE(delta_ptr);
	return Ep-Epr;
}
/*----------------------------*/
/*+++ init momentum arrays +++*/
/*----------------------------*/
void ann_momentum_init(kernel_ann *kernel){
	UINT idx;
	UINT64 allocate=0;
	/*common CPU part*/
	ALLOC_REPORT(KERN.dw,KERN.n_hiddens+1,DOUBLE *,allocate);
#ifndef _CUDA
	for(idx=0;idx<KERN.n_hiddens;idx++){
		ALLOC_REPORT(KERN.dw[idx],
			KERN.hiddens[idx].n_inputs*KERN.hiddens[idx].n_neurons,
			DOUBLE,allocate);
	}
	ALLOC_REPORT(KERN.dw[idx],
		KERN.output.n_inputs*KERN.output.n_neurons,
		DOUBLE,allocate);
#else  /*_CUDA*/
	UINT64 g_allocate=0;
	g_allocate=scuda_ann_allocate_momentum(kernel,_NN(get,cudas)());
#endif /*_CUDA*/
	NN_OUT(stdout,"[CPU] MOMENTUM ALLOC: %lu (bytes)\n",allocate);
#ifdef   _CUDA
	NN_OUT(stdout,"[GPU] MOMENTUM ALLOC: %lu (bytes)\n",g_allocate);
#endif /*_CUDA*/
}
/*------------------------------*/
/*+++ zeroes momentum arrays +++*/
/*------------------------------*/
void ann_raz_momentum(kernel_ann *kernel){
	UINT idx;
#ifndef  _CUDA
	for(idx=0;idx<KERN.n_hiddens;idx++){
		memset(KERN.dw[idx],0,sizeof(DOUBLE)*
		(KERN.hiddens[idx].n_inputs*KERN.hiddens[idx].n_neurons));
	}
	memset(KERN.dw[KERN.n_hiddens],0,sizeof(DOUBLE)*
		(KERN.output.n_inputs*KERN.output.n_neurons));
#else  /*_CUDA*/
	scuda_ann_raz_momentum(kernel,_NN(get,cudas)());
#endif /*_CUDA*/
}
/*----------------------------*/
/*+++ FREE momentum arrays +++*/
/*----------------------------*/
void ann_momentum_free(kernel_ann *kernel){
	UINT idx;
	/*FREE everything*/
#ifndef  _CUDA
	for(idx=0;idx<KERN.n_hiddens;idx++) FREE(KERN.dw[idx]);
	FREE(KERN.dw[KERN.n_hiddens]);
#else  /*_CUDA*/
	/*allocate everything in CUDA*/
	scuda_ann_free_momentum(kernel,_NN(get,cudas)());
#endif /*_CUDA*/
	FREE(KERN.dw);
}
/*---------------------------------*/
/*+++ momentum back-propagation +++*/
/*---------------------------------*/
DOUBLE ann_kernel_train_momentum(kernel_ann *kernel,const DOUBLE *train,DOUBLE alpha){
	UINT idx,N,M;
#ifdef _MPI
	UINT red, rem;
	UINT n_streams,stream;
	_NN(get,mpi_tasks)(&n_streams);
	_NN(get,curr_mpi_task)(&stream);
#endif /*_MPI*/
#if !defined (PBLAS) && !defined (SBLAS)
	UINT kdx;
#endif
#ifndef PBLAS
	UINT jdx;
#endif
	DOUBLE Ep=0.;
	DOUBLE Epr=0.;
	DOUBLE **delta_ptr;
	/*keep a track of mem*/
	if(!ann_validate_kernel(kernel)) return 0.;
	UINT64 allocate=0.;
	ALLOC_REPORT(delta_ptr,KERN.n_hiddens+1,DOUBLE *,allocate);/*+1 for OUTPUT*/
	ALLOC_REPORT(delta_ptr[KERN.n_hiddens],KERN.n_outputs,DOUBLE,allocate);
	for(idx=0;idx<KERN.n_hiddens;idx++)
		ALLOC_REPORT(delta_ptr[idx],KERN.hiddens[idx].n_neurons,DOUBLE,allocate);
/*+++ I - forward is _supposed_ to be done already +++*/
	Ep=ann_kernel_train_error(kernel,train);
//	NN_DBG(stdout,"TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - calculate deltas +++*/
	ann_kernel_train_delta(kernel,train,delta_ptr);
/*+++ III - back propagation +++*/
/*^^^ output*/
	N=KERN.output.n_neurons;
	M=KERN.output.n_inputs;
#ifdef _MPI
	red=N/n_streams;
	rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
	cblas_dger(CblasRowMajor,red,M,LEARN_RATE,delta_ptr[KERN.n_hiddens]+stream*red,
	1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.dw[KERN.n_hiddens]+stream*M*red,M);
	cblas_daxpy(red*M,1.0,KERN.dw[KERN.n_hiddens]+stream*M*red,1,KERN.output.weights+stream*M*red,1);
	cblas_dscal(red*M,alpha,KERN.dw[KERN.n_hiddens]+stream*M*red,1);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[KERN.n_hiddens],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
		cblas_dger(CblasRowMajor,rem,M,LEARN_RATE,delta_ptr[KERN.n_hiddens]+n_streams*red,
		1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.dw[KERN.n_hiddens]+n_streams*M*red,M);
		cblas_daxpy(rem*M,1.0,KERN.dw[KERN.n_hiddens]+n_streams*M*red,1,KERN.output.weights+n_streams*M*red,1);
		cblas_dscal(rem*M,alpha,KERN.dw[KERN.n_hiddens]+n_streams*M*red,1);
	}
#else /*_MPI*/
	/*unfortunately dger output can't be scaled*/
	cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[KERN.n_hiddens],
	1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.dw[KERN.n_hiddens],M);
	cblas_daxpy(N*M,1.0,KERN.dw[KERN.n_hiddens],1,KERN.output.weights,1);
	cblas_dscal(N*M,alpha,KERN.dw[KERN.n_hiddens],1);
#endif /*_MPI*/
#elif defined(SBLAS)
	/*move the ger into a series of axpy*/
#ifdef _MPI
#pragma omp parallel for private(idx) _NT
	for(idx=0;idx<red;idx++){
_HT;
		cblas_daxpy(M,delta_ptr[KERN.n_hiddens][idx+stream*red]*LEARN_RATE,
		&(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,
		&(KERN.dw[KERN.n_hiddens][(idx+stream*red)*M]),1);
		cblas_daxpy(M,1.0,
		&(KERN.dw[KERN.n_hiddens][(idx+stream*red)*M]),1,
		&(KERN.output.weights[_2D_IDX(M,idx+stream*red,0)]),1);
		cblas_dscal(M,alpha,&(KERN.dw[KERN.n_hiddens][(idx+stream*red)*M]),1);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[KERN.n_hiddens],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(idx) _NT
		for(idx=0;idx<rem;idx++){
_HT;
			cblas_daxpy(M,delta_ptr[KERN.n_hiddens][idx+n_streams*red]*LEARN_RATE,
			&(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,
			&(KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M]),1);
			cblas_daxpy(M,1.0,
			&(KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M]),1,
			&(KERN.output.weights[_2D_IDX(M,idx+n_streams*red,0)]),1);
			cblas_dscal(M,alpha,&(KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M]),1);
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(idx) _NT
	for(idx=0;idx<N;idx++){
_HT;
		//dw += LEARN_RATE*delta*y
		cblas_daxpy(M,delta_ptr[KERN.n_hiddens][idx]*LEARN_RATE,
		&(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,
		&(KERN.dw[KERN.n_hiddens][idx*M]),1);
		//W += dw
		cblas_daxpy(M,1.0,
		&(KERN.dw[KERN.n_hiddens][idx*M]),1,
		&(KERN.output.weights[_2D_IDX(M,idx,0)]),1);
		//dw *= alpha
		cblas_dscal(M,alpha,&(KERN.dw[KERN.n_hiddens][idx*M]),1);
	}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(idx,jdx) _NT
	for(idx=0;idx<red;idx++){
		for(jdx=0;jdx<M;jdx++){
			KERN.dw[KERN.n_hiddens][(idx+stream*red)*M+jdx]+=
				LEARN_RATE*delta_ptr[KERN.n_hiddens][idx+stream*red]*KERN.hiddens[KERN.n_hiddens-1].vec[jdx];
			KERN.output.weights[(idx+stream*red)*M+jdx]+=KERN.dw[KERN.n_hiddens][(idx+stream*red)*M+jdx];
			KERN.dw[KERN.n_hiddens][(idx+stream*red)*M+jdx]*=alpha;
		}
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[KERN.n_hiddens],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(idx,jdx) _NT
		for(idx=0;idx<rem;idx++){
			for(jdx=0;jdx<M;jdx++){
				KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M+jdx]+=
					LEARN_RATE*delta_ptr[KERN.n_hiddens][idx+n_streams*red]*KERN.hiddens[KERN.n_hiddens-1].vec[jdx];
				KERN.output.weights[(idx+n_streams*red)*M+jdx]+=KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M+jdx];
				KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M+jdx]*=alpha;
			}
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(idx,jdx) _NT
	for(idx=0;idx<N;idx++){
		for(jdx=0;jdx<M;jdx++){
			KERN.dw[KERN.n_hiddens][idx*M+jdx]+=LEARN_RATE*delta_ptr[KERN.n_hiddens][idx]*KERN.hiddens[KERN.n_hiddens-1].vec[jdx];
			KERN.output.weights[_2D_IDX(M,idx,jdx)]+=KERN.dw[KERN.n_hiddens][idx*M+jdx];
			KERN.dw[KERN.n_hiddens][idx*M+jdx]*=alpha;
		}
	}
#endif /*_MPI*/
#endif /*PBLAS*/
/*^^^ hiddens*/
	for(idx=(KERN.n_hiddens-1);idx>0;idx--){
		N=KERN.hiddens[idx].n_neurons;
		M=KERN.hiddens[idx].n_inputs;
#ifdef _MPI
		red=N/n_streams;
		rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
	cblas_dger(CblasRowMajor,red,M,LEARN_RATE,
		delta_ptr[idx]+stream*red,1,KERN.hiddens[idx-1].vec,1,KERN.dw[idx]+stream*M*red,M);
	cblas_daxpy(N*M,1.0,KERN.dw[idx]+stream*M*red,1,KERN.hiddens[idx].weights+stream*M*red,1);
	cblas_dscal(N*M,alpha,KERN.dw[idx]+stream*M*red,1);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[idx],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
		cblas_dger(CblasRowMajor,red,M,LEARN_RATE,
			delta_ptr[idx]+n_streams*red,1,KERN.hiddens[idx-1].vec,1,KERN.dw[idx]+n_streams*M*red,M);
		cblas_daxpy(N*M,1.0,KERN.dw[idx]+n_streams*M*red,1,KERN.hiddens[idx].weights+n_streams*M*red,1);
		cblas_dscal(N*M,alpha,KERN.dw[idx]+n_streams*M*red,1);
	}
#else /*_MPI*/
	cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[idx],1,KERN.hiddens[idx-1].vec,1,KERN.dw[idx],M);
	cblas_daxpy(N*M,1.0,KERN.dw[idx],1,KERN.hiddens[idx].weights,1);
	cblas_dscal(N*M,alpha,KERN.dw[idx],1);
#endif /*_MPI*/
#elif defined(SBLAS)
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<red;jdx++){
_HT;
		cblas_daxpy(M,delta_ptr[idx][jdx+stream*red]*LEARN_RATE,
			KERN.hiddens[idx-1].vec,1,&(KERN.dw[idx][(jdx+stream*red)*M]),1);
		cblas_daxpy(M,1.0,&(KERN.dw[idx][(jdx+stream*red)*M]),1,&(KERN.hiddens[idx].weights[(jdx+stream*red)*M]),1);
		cblas_dscal(M,alpha,&(KERN.dw[idx][(jdx+stream*red)*M]),1);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[idx],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(jdx) _NT
		for(jdx=0;jdx<rem;jdx++){
_HT;
			cblas_daxpy(M,delta_ptr[idx][jdx+n_streams*red]*LEARN_RATE,
				KERN.hiddens[idx-1].vec,1,&(KERN.dw[idx][(jdx+n_streams*red)*M]),1);
			cblas_daxpy(M,1.0,&(KERN.dw[idx][(jdx+n_streams*red)*M]),1,&(KERN.hiddens[idx].weights[(jdx+n_streams*red)*M]),1);
			cblas_dscal(M,alpha,&(KERN.dw[idx][(jdx+n_streams*red)*M]),1);
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<N;jdx++){
_HT;
		//dw += LEARN_RATE*delta*y
		cblas_daxpy(M,delta_ptr[idx][jdx]*LEARN_RATE,KERN.hiddens[idx-1].vec,1,&(KERN.dw[idx][_2D_IDX(M,jdx,0)]),1);
		//W += dw
		cblas_daxpy(M,1.0,&(KERN.dw[idx][_2D_IDX(M,jdx,0)]),1,&(KERN.hiddens[idx].weights[_2D_IDX(M,jdx,0)]),1);
		//dw *= alpha
		cblas_dscal(M,alpha,&(KERN.dw[idx][_2D_IDX(M,jdx,0)]),1);
	}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<red;jdx++){
		for(kdx=0;kdx<M;kdx++){
			KERN.dw[idx][(jdx+stream*red)*M+kdx]+=
				LEARN_RATE*delta_ptr[idx][jdx+stream*red]*KERN.hiddens[idx-1].vec[kdx];
			KERN.hiddens[idx].weights[(jdx+stream*red)*M+kdx]+=KERN.dw[idx][(jdx+stream*red)*M+kdx];
			KERN.dw[idx][(jdx+stream*red)*M+kdx]*=alpha;
		}
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[idx],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
		for(jdx=0;jdx<rem;jdx++){
			for(kdx=0;kdx<M;kdx++){
				KERN.dw[idx][(jdx+n_streams*red)*M+kdx]+=
					LEARN_RATE*delta_ptr[idx][jdx+n_streams*red]*KERN.hiddens[idx-1].vec[kdx];
				KERN.hiddens[idx].weights[(jdx+n_streams*red)*M+kdx]+=KERN.dw[idx][(jdx+n_streams*red)*M+kdx];
				KERN.dw[idx][(jdx+n_streams*red)*M+kdx]*=alpha;
			}
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<N;jdx++){
		for(kdx=0;kdx<M;kdx++){
			KERN.dw[idx][_2D_IDX(M,jdx,kdx)]+=LEARN_RATE*delta_ptr[idx][jdx]*KERN.hiddens[idx-1].vec[kdx];
			KERN.hiddens[idx].weights[_2D_IDX(M,jdx,kdx)]+=KERN.dw[idx][_2D_IDX(M,jdx,kdx)];
			KERN.dw[idx][_2D_IDX(M,jdx,kdx)]*=alpha;
		}
	}
#endif /*_MPI*/
#endif /*PBLAS*/
	}/*idx: hiddens*/
	/*add zero*/
	N=KERN.hiddens[0].n_neurons;
	M=KERN.hiddens[0].n_inputs;
#ifdef _MPI
	red=N/n_streams;
	rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
	cblas_dger(CblasRowMajor,red,M,LEARN_RATE,delta_ptr[0]+stream*red,1,KERN.in,1,KERN.dw[0]+stream*M*red,M);
	cblas_daxpy(red*M,1.0,KERN.dw[0]+stream*M*red,1,KERN.hiddens[0].weights+stream*M*red,1);
	cblas_dscal(red*M,alpha,KERN.dw[0]+stream*M*red,1);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[0],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
		cblas_dger(CblasRowMajor,rem,M,LEARN_RATE,delta_ptr[0]+n_streams*red,1,KERN.in,1,KERN.dw[0]+n_streams*M*red,M);
		cblas_daxpy(rem*M,1.0,KERN.dw[0]+n_streams*M*red,1,KERN.hiddens[0].weights+n_streams*M*red,1);
		cblas_dscal(rem*M,alpha,KERN.dw[0]+n_streams*M*red,1);
	}
#else /*_MPI*/
	cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[0],1,KERN.in,1,KERN.dw[0],M);
	cblas_daxpy(N*M,1.0,KERN.dw[0],1,KERN.hiddens[0].weights,1);
	cblas_dscal(N*M,alpha,KERN.dw[0],1);
#endif /*_MPI*/
#elif defined(SBLAS)
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<red;jdx++){
_HT;
		cblas_daxpy(M,delta_ptr[0][jdx+stream*red]*LEARN_RATE,KERN.in,1,&(KERN.dw[0][(jdx+stream*red)*M]),1);
		cblas_daxpy(M,1.0,&(KERN.dw[0][(jdx+stream*red)*M]),1,&(KERN.hiddens[0].weights[(jdx+stream*red)*M]),1);
		cblas_dscal(M,alpha,&(KERN.dw[0][(jdx+stream*red)*M]),1);
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[0],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
		for(jdx=0;jdx<rem;jdx++){
_HT;
			cblas_daxpy(M,delta_ptr[0][jdx+n_streams*red]*LEARN_RATE,KERN.in,1,&(KERN.dw[0][(jdx+n_streams*red)*M]),1);
			cblas_daxpy(M,1.0,&(KERN.dw[0][(jdx+n_streams*red)*M]),1,&(KERN.hiddens[0].weights[(jdx+n_streams*red)*M]),1);
			cblas_dscal(M,alpha,&(KERN.dw[0][(jdx+n_streams*red)*M]),1);
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<N;jdx++){
_HT;
		//dw += LEARN_RATE*delta*y
		cblas_daxpy(M,delta_ptr[0][jdx]*LEARN_RATE,KERN.in,1,&(KERN.dw[0][_2D_IDX(M,jdx,0)]),1);
		//W += dw
		cblas_daxpy(M,1.0,&(KERN.dw[0][_2D_IDX(M,jdx,0)]),1,&(KERN.hiddens[0].weights[_2D_IDX(M,jdx,0)]),1);
		//dw *= alpha
		cblas_dscal(M,alpha,&(KERN.dw[0][_2D_IDX(M,jdx,0)]),1);
	}
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
	for(jdx=0;jdx<red;jdx++){
		for(kdx=0;kdx<M;kdx++){
			KERN.dw[0][(jdx+stream*red)*M+kdx]+=LEARN_RATE*delta_ptr[0][jdx+stream*red]*KERN.in[kdx];
			KERN.hiddens[0].weights[(jdx+stream*red)*M+kdx]+=KERN.dw[0][(jdx+stream*red)*M+kdx];
			KERN.dw[0][(jdx+stream*red)*M+kdx]*=alpha;
		}
	}
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[0],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
	if(rem>0){
#pragma omp parallel for private(jdx) _NT
		for(jdx=0;jdx<rem;jdx++){
			for(kdx=0;kdx<M;kdx++){
				KERN.dw[0][(jdx+n_streams*red)*M+kdx]+=LEARN_RATE*delta_ptr[0][jdx+n_streams*red]*KERN.in[kdx];
				KERN.hiddens[0].weights[(jdx+n_streams*red)*M+kdx]+=KERN.dw[0][(jdx+n_streams*red)*M+kdx];
				KERN.dw[0][(jdx+n_streams*red)*M+kdx]*=alpha;
			}
		}
	}
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
	for(jdx=0;jdx<N;jdx++){
		for(kdx=0;kdx<M;kdx++){
			KERN.dw[0][_2D_IDX(M,jdx,kdx)]+=LEARN_RATE*delta_ptr[0][jdx]*KERN.in[kdx];
			KERN.hiddens[0].weights[_2D_IDX(M,jdx,kdx)]+=KERN.dw[0][_2D_IDX(M,jdx,kdx)];
			KERN.dw[0][_2D_IDX(M,jdx,kdx)]*=alpha;
		}
	}
#endif /*_MPI*/
#endif /*PBLAS*/
/*+++ IV - update error +++*/
	ann_kernel_run(kernel);
	Epr=ann_kernel_train_error(kernel,train);
//	NN_DBG(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ IV - cleanup +++*/
	for(idx=0;idx<(KERN.n_hiddens+1);idx++){
		FREE(delta_ptr[idx]);
		delta_ptr[idx]=NULL;
	}
	FREE(delta_ptr);
	return Ep-Epr;
}

/*--------------------------*/
/* train ANN sample with BP */
/*--------------------------*/
DOUBLE ann_train_BP(kernel_ann *kernel,DOUBLE *train_in,DOUBLE *train_out,DOUBLE delta){
/*typical values delta=0.000001*/
	BOOL is_ok;
	UINT   idx;
	UINT  iter;
	UINT max_p;
	UINT p_trg;
	DOUBLE dEp;
	DOUBLE *ptr;
	DOUBLE probe;
#ifdef _CUDA
	cudastreams *cudas=_NN(get,cudas)();
	DOUBLE *train_gpu;
	cudaSetDevice(0);/*make sure all transfer happen to gpu[0]*/
	CUDA_C2G_CP(train_in,KERN.in,KERN.n_inputs,DOUBLE);
	CUDA_ALLOC(train_gpu,KERN.n_outputs,DOUBLE);
	CUDA_C2G_CP(train_out,train_gpu,KERN.n_outputs,DOUBLE);
	scuda_ann_forward(kernel,cudas);
	dEp=scuda_ann_error(kernel,train_gpu,cudas);
#else /*_CUDA*/
	/*copy input*/
	ARRAY_CP(train_in,KERN.in,KERN.n_inputs);
	dEp=0.;
	ann_kernel_run(kernel);/*also FILL vec*/
	for(idx=0;idx<kernel->n_outputs;idx++)
		dEp+=(train_out[idx]-kernel->output.vec[idx])*(train_out[idx]-kernel->output.vec[idx]);
	dEp*=0.5;
#endif /*_CUDA*/
	NN_COUT(stdout," init=%15.10f",dEp);
	iter=0;
	do{
		iter++;
#ifdef _CUDA
		dEp=(DOUBLE)scuda_ann_train(kernel,train_gpu,cudas);
if(cudas->mem_model!=CUDA_MEM_CMM){
		/*we have to sync output.cuda_v -> out*/
		cudaSetDevice(0);/*make sure transfer happen from GPU[0]*/
		CUDA_G2C_CP(kernel->tmp_cpu,kernel->output.vec,KERN.n_outputs,DOUBLE);
		cudaDeviceSynchronize();/*<-useful?*/
		ptr=kernel->tmp_cpu;
}else{
		/*CMM can use GPU memory directly*/
		cudaMemPrefetchAsync(kernel->output.vec,
			kernel->n_outputs*sizeof(DOUBLE),cudaCpuDeviceId,NULL);
		ptr=kernel->output.vec;
}
#else /*_CUDA*/
		dEp=ann_kernel_train(kernel,train_out);
		ptr=kernel->output.vec;
#endif /*_CUDA*/
		/*1- determine max_p, p_trg*/
		probe=-1.0;max_p=0;p_trg=0;
		for(idx=0;idx<KERN.n_outputs;idx++){
			if(probe<ptr[idx]){
				probe=ptr[idx];
				max_p=idx;
			}
			if(train_out[idx]==1.0) p_trg=idx;
		}
		/*2- match*/
		is_ok=(max_p==p_trg);
		if(iter==1){
			/*determine if we get a good answer at first try*/
			if(is_ok==TRUE) NN_COUT(stdout," OK");
			else NN_COUT(stdout," NO");
		}
		if(iter>10239) break;/*failsafe number of wrong iteration*/
	}while((dEp > delta)||(!(is_ok==TRUE)));
	NN_COUT(stdout," N_ITER=%8i",iter);
	if(is_ok==TRUE) NN_COUT(stdout," SUCCESS!\n");
	else NN_COUT(stdout," FAIL!\n");
	fflush(stdout);
#ifdef _CUDA
	CUDA_FREE(train_gpu);
#endif /*_CUDA*/
	return dEp;
}
/*---------------------------*/
/* train ANN sample with BPM */
/*---------------------------*/
DOUBLE ann_train_BPM(kernel_ann *kernel,DOUBLE *train_in,DOUBLE *train_out,DOUBLE alpha,DOUBLE delta){
/*typical values alpha=0.2 delta=0.00001*/
	BOOL is_ok;
	UINT   idx;
	UINT  iter;
	UINT max_p;
	UINT p_trg;
	DOUBLE dEp;
	DOUBLE *ptr;
	DOUBLE probe;
	ann_raz_momentum(kernel);
#ifdef _CUDA
	cudastreams *cudas=_NN(get,cudas)();
	DOUBLE *train_gpu;
	cudaSetDevice(0);/*make sure all transfer happen to gpu[0]*/
	CUDA_C2G_CP(train_in,KERN.in,KERN.n_inputs,DOUBLE);
	CUDA_ALLOC(train_gpu,KERN.n_outputs,DOUBLE);
	CUDA_C2G_CP(train_out,train_gpu,KERN.n_outputs,DOUBLE);
	scuda_ann_forward(kernel,cudas);
	dEp=scuda_ann_error(kernel,train_gpu,cudas);
#else /*_CUDA*/
	ARRAY_CP(train_in,KERN.in,KERN.n_inputs);
	dEp=0.;
	ann_kernel_run(kernel);/*also FILL vec*/
	for(idx=0;idx<kernel->n_outputs;idx++)
		dEp+=(train_out[idx]-kernel->output.vec[idx])*(train_out[idx]-kernel->output.vec[idx]);
	dEp*=0.5;
#endif /*_CUDA*/
	NN_COUT(stdout," init=%15.10f",dEp);
	iter=0;
	do{
		iter++;
#ifdef _CUDA
		dEp=(DOUBLE)scuda_ann_train_momentum(kernel,train_gpu,alpha,cudas);
if(cudas->mem_model!=CUDA_MEM_CMM){
		/*we have to sync output.cuda_v -> out*/
		cudaSetDevice(0);/*make sure transfer happen from GPU[0]*/
		CUDA_G2C_CP(kernel->tmp_cpu,kernel->output.vec,KERN.n_outputs,DOUBLE);
		cudaDeviceSynchronize();/*<-useful?*/
		ptr=kernel->tmp_cpu;
}else{
		/*CMM can use GPU memory directly*/
		cudaMemPrefetchAsync(kernel->output.vec,
			kernel->n_outputs*sizeof(DOUBLE),cudaCpuDeviceId,NULL);
		ptr=kernel->output.vec;
}
#else /*_CUDA*/
		dEp=ann_kernel_train_momentum(kernel,train_out,alpha);
		ptr=kernel->output.vec;
#endif /*_CUDA*/
		/*1- determine max_p, p_trg*/
		probe=-1.0;max_p=0;p_trg=0;
		for(idx=0;idx<KERN.n_outputs;idx++){
			if(probe < ptr[idx]){
				probe = ptr[idx];
				max_p = idx;
			}
			if(train_out[idx]==1.0) p_trg = idx;
		}
		/*2- match*/
		is_ok=(max_p == p_trg);
		if(iter==1){
			/*determine if we get a good answer at first try*/
			if(is_ok==TRUE) NN_COUT(stdout," OK");
			else NN_COUT(stdout," NO");
		}
		if(iter>10239) break;/*failsafe number of wrong iteration*/	
	}while((dEp > delta)||(!(is_ok==TRUE)));
	NN_COUT(stdout," N_ITER=%8i",iter);
	if(is_ok==TRUE) NN_COUT(stdout," SUCCESS!\n");
	else NN_COUT(stdout," FAIL!\n");
	fflush(stdout);
#ifdef _CUDA
	CUDA_FREE(train_gpu);
#endif /*_CUDA*/
	return dEp;
}

#undef KERN
