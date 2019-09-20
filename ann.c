#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>

#ifdef PBLAS
#include <cblas.h>
#endif

#ifdef _OMP
#include <omp.h>
#endif

#include "common.h"
#include "ann.h"

//int verbose;
//int dry_run;

/*GLOBAL VARIABLES*/
SHORT ann_verbose=0;
BOOL ann_dry=FALSE;
#ifdef _OMP
UINT ann_num_threads=1;
#endif

void ann_set_verbose(){
	ann_verbose++;
	if(ann_verbose>0) fprintf(stdout,"# NN: increasing verbosity\n");
}

void ann_set_dry(){
	ann_dry^=ann_dry;
}

int ann_init(){
#ifdef _OMP
	#pragma omp parallel default(shared)
	{
		ann_num_threads = omp_get_num_threads();
	}
	fprintf(stdout,"ANN started with %i threads.\n",ann_num_threads);
#endif /*_OMP*/
	return 0;
}

_kernel *ann_load(CHAR *f_kernel){
#define FAIL load_kernel_fail
#define KERN (*kernel)
	PREP_READLINE();
        CHAR *line=NULL;
        CHAR *ptr,*ptr2;
	_kernel *kernel;
	UINT *parameter;
	UINT64 allocate;
//	int is_ok;
	FILE  *fp;
	UINT  idx;
	UINT  jdx;
	UINT  kdx;
	CHAR *name;
	UINT  n_in;
	UINT n_out;
	UINT n_hid;
	UINT n_par;
	/*init*/
	n_in =0;
	n_out=0;
	n_hid=0;
	n_par=0;
	kernel=NULL;
	parameter=NULL;
	/**/
	fp=fopen(f_kernel,"r");
	if(!fp){
		fprintf(stderr,"Error opening kernel file: %s\n",f_kernel);
		return NULL;
	}
	READLINE(fp,line);/*line 1: name (SKIP)*/
	ptr=STRFIND("[name]",line);
	if(ptr==NULL){
		fprintf(stderr,"ANN kernel ERROR: kernel file should start with [name] keyword!\n");
		goto FAIL;
	}
	ptr+=6;SKIP_BLANK(ptr);
	allocate=0;
	STRDUP_REPORT(ptr,name,allocate);
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
				fprintf(stderr,"ANN kernel ERROR: malformed parameter line!\n");
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
				fprintf(stderr,"ANN kernel ERROR: parameter line has too few parameters!\n");
				goto FAIL;
			}
			n_hid=n_par-1;
			/*get number of input*/
			ptr=&(line[0]);SKIP_BLANK(ptr);
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			GET_UINT(n_in,ptr,ptr2);ptr=ptr2+1;SKIP_BLANK(ptr);
			ALLOC(parameter,n_par-1,UINT);
			jdx=1;
			for(idx=0;idx<n_par;idx++) {
				GET_UINT(parameter[idx],ptr,ptr2);
				jdx*=(parameter[idx]!=0);
				ptr=ptr2+1;SKIP_BLANK(ptr);
			}
			if(jdx==0){
				fprintf(stderr,"ANN kernel ERROR: zero in parameter line!\n");
				goto FAIL;
			}
			n_out=parameter[n_par-1];
			
			break;
		}
		READLINE(fp,line);
	}while(!feof(fp));
	if(n_in==0) {
		fprintf(stderr,"ANN kernel ERROR: missing parameter line!\n");
		goto FAIL;
	}
	/*allocate everything*/
	ALLOC_REPORT(kernel,1,_kernel,allocate);
	KERN.name=name;name=NULL;
	KERN.n_inputs=n_in;
	KERN.n_hiddens=n_hid;
	KERN.n_outputs=n_out;
	ALLOC_REPORT(KERN.in,n_in,DOUBLE,allocate);
	ALLOC_REPORT(KERN.hiddens,n_hid,_layer,allocate);
	ALLOC_REPORT(KERN.out,n_out,DOUBLE,allocate);
	/*first hidden layer*/
	KERN.hiddens[0].n_neurons=parameter[0];
	KERN.hiddens[0].n_inputs=n_in;
	ALLOC_REPORT(KERN.hiddens[0].weights,n_in*KERN.hiddens[0].n_neurons,DOUBLE,allocate);
	/*remaining hidden layers*/
	for(idx=1;idx<n_hid;idx++){
		KERN.hiddens[idx].n_neurons=parameter[idx];
		KERN.hiddens[idx].n_inputs=parameter[idx-1];
		ALLOC_REPORT(KERN.hiddens[idx].weights,parameter[idx]*parameter[idx-1],DOUBLE,allocate);
	}
	/*output*/
	KERN.output.n_neurons=n_out;
	KERN.output.n_inputs=parameter[n_par-2];
	ALLOC_REPORT(KERN.output.weights,n_out*parameter[n_par-2],DOUBLE,allocate);
	/*end of allocations*/
fprintf(stdout,"ANN total allocation: %lu (bytes)\n",allocate);
fprintf(stdout,"n_input=%i ",n_in);
for(jdx=0;jdx<n_par-1;jdx++) fprintf(stdout,"n_hidden[%i]=%i ",jdx,parameter[jdx]);
fprintf(stdout,"n_output=%i\n",n_out);
	FREE(parameter);
	/*getting weights when available*/
	rewind(fp);
	/*1- find [hidden]*/
	do{
		ptr=STRFIND("[hidden",line);
		if(ptr!=NULL){
//			[hidden X] Y -> hidden layer X has Y neurons
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			if(!ISDIGIT(*ptr)) {
				fprintf(stderr,"ANN kernel ERROR: malformed hidden layer definition\n");
				goto FAIL;
			}
			GET_UINT(idx,ptr,ptr2);/*this is hidden index*/
			if((ptr2==NULL)||(idx<1)) {
				fprintf(stderr,"ANN kernel ERROR: malformed hidden layer index definition\n");
				goto FAIL;
			}
			idx--;/*start counting from 1*/
			/*check neuron number for consistency*/
			ptr=ptr2+1;
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			GET_UINT(jdx,ptr,ptr2);
			if(jdx!=KERN.hiddens[idx].n_neurons){
				fprintf(stderr,"ANN kernel ERROR: inconsistent neuron number - layer %i n_neurons=%i (expected %i)\n",
					idx+1,jdx,KERN.hiddens[idx].n_neurons);
				goto FAIL;
			}
/*now let's fetch neurons*/
READLINE(fp,line);
jdx=0;
do{
	ptr=STRFIND("[neuron",line);
	if(ptr==NULL){
		fprintf(stderr,"ANN kernel ERROR: neuron definition missing! (hidden layer %i, neuron %i)\n",idx+1,jdx+1);
		goto FAIL;
	}
	while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
	if(!ISDIGIT(*ptr)) {
		fprintf(stderr,"ANN kernel ERROR: missing neuron number! (hidden layer %i, neuron %i)\n",idx+1,jdx+1);
		goto FAIL;
	}
	GET_UINT(n_par,ptr,ptr2);/*this is neuron number*/
	if(n_par<1) {
		fprintf(stderr,"ANN kernel ERROR: neuron number<1 (hidden layer %i, neuron %i)\n",idx+1,jdx+1);
		goto FAIL;
	}
	ptr=ptr2+1;SKIP_BLANK(ptr);
	if(!ISDIGIT(*ptr)) {
		fprintf(stderr,"ANN kernel ERROR: neuron has no input number! (hidden layer %i, neuron %i)\n",idx+1,jdx+1);
		goto FAIL;
	}
	GET_UINT(n_par,ptr,ptr2);/*this is number of inputs*/
	if(n_par<1) {
		fprintf(stderr,"ANN kernel ERROR: neuron has less that 1 input! (hidden layer %i, neuron %i)\n",idx+1,jdx+1);
		goto FAIL;
	}
	READLINE(fp,line);/*weights line*/
	ptr=&(line[0]);SKIP_BLANK(ptr);
	for(kdx=0;kdx<n_par;kdx++){
		/*read weights*/
		GET_DOUBLE(KERN.hiddens[idx].weights[_2D_IDX(n_par,jdx,kdx)],ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
		ptr=ptr2+1;SKIP_BLANK(ptr);
	}
	jdx++;
	READLINE(fp,line);
}while(jdx<KERN.hiddens[idx].n_neurons);
/*continue*/
		} else READLINE(fp,line);
	}while(!feof(fp));
/*finally get the output weights*/
	rewind(fp);
	do{
		ptr=STRFIND("[output]",line);
		if(ptr!=NULL){
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			if(!ISDIGIT(*ptr)) {
				fprintf(stderr,"ANN kernel ERROR: malformed output layer definition\n");
				goto FAIL;
			}
			/*check neuron number for consistency*/
			GET_UINT(idx,ptr,ptr2);/*this is the number of output*/
			if((ptr2==NULL)||(idx!=KERN.output.n_neurons)) {
				fprintf(stderr,"ANN kernel ERROR: inconsistent neuron number for output - n_neurons=%i (expected %i)\n",
					idx,KERN.output.n_neurons);
				goto FAIL;
			}
/*now let's fetch neurons*/
READLINE(fp,line);
jdx=0;
do{
        ptr=STRFIND("[neuron",line);
        if(ptr==NULL){
                fprintf(stderr,"ANN kernel ERROR: neuron definition missing! (output layer, neuron %i)\n",jdx+1);
                goto FAIL;
        }
        while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
        if(!ISDIGIT(*ptr)) {
                fprintf(stderr,"ANN kernel ERROR: missing neuron number! (output layer, neuron %i)\n",jdx+1);
                goto FAIL;
        }
        GET_UINT(n_par,ptr,ptr2);/*this is hidden index*/
        if(n_par<1) {
                fprintf(stderr,"ANN kernel ERROR: neuron number<1 (output layer, neuron %i)\n",jdx+1);
                goto FAIL;
        }
        ptr=ptr2+1;SKIP_BLANK(ptr);
        if(!ISDIGIT(*ptr)) {
                fprintf(stderr,"ANN kernel ERROR: neuron has no input number! (output layer, neuron %i)\n",jdx+1);
                goto FAIL;
        }
        GET_UINT(n_par,ptr,ptr2);/*this is number of inputs*/
        if(n_par<1) {
                fprintf(stderr,"ANN kernel ERROR: neuron has less that 1 input! (output layer, neuron %i)\n",jdx+1);
                goto FAIL;
        }
        READLINE(fp,line);
        ptr=&(line[0]);SKIP_BLANK(ptr);
        for(kdx=0;kdx<n_par;kdx++){
                /*read weights*/
                GET_DOUBLE(KERN.output.weights[_2D_IDX(n_par,jdx,kdx)],ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
                ptr=ptr2+1;SKIP_BLANK(ptr);
        }
        jdx++;
	READLINE(fp,line);
}while(jdx<KERN.output.n_neurons);
/*continue*/
		}
		READLINE(fp,line);
	}while(!feof(fp));
	/*end*/
	FREE(line);
	fclose(fp);
	return kernel;
load_kernel_fail:
	FREE(parameter);
	FREE(line);
	fclose(fp);
	return NULL;
#undef FAIL
}
_kernel *ann_generate(UINT *seed,UINT n_inputs,UINT n_hiddens,UINT n_outputs,UINT *hiddens){
	_kernel *kernel;
	UINT64 allocate;
	UINT   idx, jdx;
	DOUBLE temp_rnd;
	/*this generation should _NOT_ be performed in parallel*/
	allocate=0.;
	if(*seed==0) *seed=time(NULL);
	srandom(*seed);
	/*allocation*/
	ALLOC_REPORT(kernel,1,_kernel,allocate);
	KERN.n_inputs=n_inputs;
	KERN.n_hiddens=n_hiddens;
	KERN.n_outputs=n_outputs;
	ALLOC_REPORT(KERN.in,n_inputs,DOUBLE,allocate);
	ALLOC_REPORT(KERN.hiddens,n_hiddens,_layer,allocate);
	ALLOC_REPORT(KERN.out,n_outputs,DOUBLE,allocate);
	/*first layer*/
	KERN.hiddens[0].n_inputs=n_inputs;
	KERN.hiddens[0].n_neurons=hiddens[0];
	ALLOC_REPORT(KERN.hiddens[0].weights,n_inputs*hiddens[0],DOUBLE,allocate);
	/*remaining hidden layers*/
	for(idx=1;idx<n_hiddens;idx++){
		KERN.hiddens[idx].n_neurons=hiddens[idx];
		KERN.hiddens[idx].n_inputs=hiddens[idx-1];
		ALLOC_REPORT(KERN.hiddens[idx].weights,hiddens[idx]*hiddens[idx-1],DOUBLE,allocate);
	}
	/*output*/
	KERN.output.n_neurons=n_outputs;
	KERN.output.n_inputs=hiddens[n_hiddens-1];
	ALLOC_REPORT(KERN.output.weights,n_outputs*hiddens[n_hiddens-1],DOUBLE,allocate);
	fprintf(stdout,"ANN total allocation: %lu (bytes)\n",allocate);
	/*randomly fill hidden weights*/
	for(idx=0;idx<n_hiddens;idx++){
		for(jdx=0;jdx<KERN.hiddens[idx].n_inputs*KERN.hiddens[idx].n_neurons;jdx++){
			temp_rnd=(DOUBLE) random() / RAND_MAX;
			KERN.hiddens[idx].weights[jdx]=2.0*(temp_rnd-0.5)/sqrt(KERN.hiddens[idx].n_inputs);
		}
	}
	/*randomly fill output weight*/
	for(jdx=0;jdx<KERN.output.n_inputs*KERN.output.n_neurons;jdx++){
		temp_rnd=(DOUBLE) random() / RAND_MAX;
		KERN.output.weights[jdx]=2.0*(temp_rnd-0.5)/sqrt(KERN.output.n_inputs);
	}
	return kernel;
}



void ann_dump(_kernel *kernel,FILE *out){
	UINT idx;
	UINT jdx;
	UINT kdx;
	if (kernel==NULL) return;
	fprintf(out,"[name] %s\n",KERN.name);
	fprintf(out,"[param] %i",KERN.n_inputs);
	for(idx=0;idx<KERN.n_hiddens;idx++) fprintf(out," %i",KERN.hiddens[idx].n_neurons);
	fprintf(out," %i\n",KERN.output.n_neurons);
	fprintf(out,"[input] %i\n",KERN.n_inputs);
	for(idx=0;idx<KERN.n_hiddens;idx++) {
		fprintf(out,"[hidden %i] %i\n",idx+1,KERN.hiddens[idx].n_neurons);
		for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++){
			fprintf(out,"[neuron %i] %i\n",jdx+1,KERN.hiddens[idx].n_inputs);
			fprintf(out,"%17.15f",KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,0)]);
			for(kdx=1;kdx<KERN.hiddens[idx].n_inputs;kdx++)
				fprintf(out," %17.15f",KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)]);
			fprintf(out,"\n");
		}
	}
	fprintf(out,"[output] %i\n",KERN.n_outputs);
	for(jdx=0;jdx<KERN.output.n_neurons;jdx++){
		fprintf(out,"[neuron %i] %i\n",jdx+1,KERN.output.n_inputs);
		fprintf(out,"%17.15f",KERN.output.weights[_2D_IDX(KERN.output.n_inputs,jdx,0)]);
		for(kdx=1;kdx<KERN.output.n_inputs;kdx++)
			fprintf(out," %17.15f",KERN.output.weights[_2D_IDX(KERN.output.n_inputs,jdx,kdx)]);
		fprintf(out,"\n");
	}
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
void ann_kernel_run(_kernel *kernel){
	UINT idx,jdx;//kdx;
	DOUBLE  *in , *out;
#ifndef PBLAS
	UINT kdx;
#endif
	/*simple, one pass kernel*/
	ALLOC(in,KERN.n_inputs,DOUBLE);
	ARRAY_CP(KERN.in,in,KERN.n_inputs);
/*+++ I - hiddens +++*/
	for(idx=0;idx<KERN.n_hiddens;idx++){
		ALLOC(out,KERN.hiddens[idx].n_neurons,DOUBLE);
#ifdef PBLAS
		cblas_dgemv(CblasRowMajor,CblasNoTrans,KERN.hiddens[idx].n_neurons,KERN.hiddens[idx].n_inputs,
			1.0,KERN.hiddens[idx].weights,KERN.hiddens[idx].n_inputs,in,1,0.,out,1);
#define OP_ACT(ix) out[ix]=ann_act(out[ix])
		UNROLL_OMP_FOR(0,KERN.hiddens[idx].n_neurons,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
#else /*PBLAS*/
for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++){
#define OP_WI(ix) out[jdx]+=KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,ix)]*in[ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[idx].n_inputs,ANN_UNROLL,WI,kdx);
#undef OP_WI
	out[jdx]=ann_act(out[jdx]);
}
//#pragma omp parallel for private(jdx)
//for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++){
//	for(kdx=0;kdx<KERN.hiddens[idx].n_inputs;kdx++){
//		out[jdx]+=KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)]*in[kdx];
//	}
//	out[jdx]=ann_act(out[jdx]);
//}
#endif /*PBLAS*/
		FREE(in);
		in=out;out=NULL;
	}
/*+++ II - output +++*/
#ifdef PBLAS
	/*serial dgemv (no thread support here)*/
	cblas_dgemv(CblasRowMajor,CblasNoTrans,KERN.output.n_neurons,KERN.output.n_inputs,
		1.0,KERN.output.weights,KERN.output.n_inputs,in,1,0.,KERN.out,1);
#define OP_ACT(ix) KERN.out[ix]=ann_act(KERN.out[ix])
	UNROLL_OMP_FOR(0,KERN.output.n_neurons,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
#else /*PBLAS*/
for(jdx=0;jdx<KERN.output.n_neurons;jdx++){
	KERN.out[jdx]=0.;/*TRAP*/
#define OP_WI(ix) KERN.out[jdx]+=KERN.output.weights[_2D_IDX(KERN.output.n_inputs,jdx,ix)]*in[ix]
	UNROLL_OMP_FOR(0,KERN.output.n_inputs,ANN_UNROLL,WI,kdx);
#undef OP_WI
	KERN.out[jdx]=ann_act(KERN.out[jdx]);
}
//#pragma omp parallel for private(jdx)
//	for(jdx=0;jdx<KERN.output.n_neurons;jdx++){
//		KERN.out[jdx]=0.;
//		for(kdx=0;kdx<KERN.output.n_inputs;kdx++){
//			KERN.out[jdx]+=KERN.output.weights[_2D_IDX(KERN.output.n_inputs,jdx,kdx)]*in[kdx];
//		}
//		KERN.out[jdx]=ann_act(KERN.out[jdx]);
//	}
#endif /*PBLAS*/
	FREE(in);
	/*done*/
}
/*------------------------*/
/*+++ back-propagation +++*/
/*------------------------*/
DOUBLE ann_kernel_train(_kernel *kernel,const DOUBLE *train){
#define LEARN_RATE 0.01
	UINT idx, jdx;
#ifndef PBLAS
	UINT kdx;
#endif
	DOUBLE Ep=0.;
	DOUBLE Epr=0.;
	DOUBLE **hidden_vector_ptr;
	DOUBLE **delta_ptr;
	/*keep a track of mem*/
	UINT64 allocate=0.;
/*+++ I - forward +++*/
	ALLOC_REPORT(hidden_vector_ptr,KERN.n_hiddens,DOUBLE *,allocate);
	for(idx=0;idx<KERN.n_hiddens;idx++){
		ALLOC_REPORT(hidden_vector_ptr[idx],KERN.hiddens[idx].n_neurons,DOUBLE,allocate);
	}
	ALLOC_REPORT(delta_ptr,KERN.n_hiddens+1,DOUBLE *,allocate);/*+1 for OUTPUT*/
	ALLOC_REPORT(delta_ptr[KERN.n_hiddens],KERN.n_outputs,DOUBLE,allocate);
	for(idx=0;idx<KERN.n_hiddens;idx++)
		ALLOC_REPORT(delta_ptr[idx],KERN.hiddens[idx].n_neurons,DOUBLE,allocate);
//fprintf(stdout,"TRAINING MEM ALLOC: %lu (bytes)\n",allocate);
/*^^^ input to hidden*/
#ifdef PBLAS
	cblas_dgemv(CblasRowMajor,CblasNoTrans,KERN.hiddens[0].n_neurons,KERN.hiddens[0].n_inputs,
		1.0,KERN.hiddens[0].weights,KERN.hiddens[0].n_inputs,KERN.in,1,0.,hidden_vector_ptr[0],1);
#define OP_ACT(ix) hidden_vector_ptr[0][ix]=ann_act(hidden_vector_ptr[0][ix])
	UNROLL_OMP_FOR(0,KERN.hiddens[0].n_neurons,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
/*serial*/
//	for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++)
//		hidden_vector_ptr[0][jdx]=ann_act(hidden_vector_ptr[0][jdx]);
#else /*PBLAS*/
for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++){
#define OP_WI(ix) hidden_vector_ptr[0][jdx]+=KERN.hiddens[0].weights[_2D_IDX(KERN.hiddens[0].n_inputs,jdx,ix)]*KERN.in[ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[0].n_inputs,ANN_UNROLL,WI,kdx);
#undef OP_WI
	hidden_vector_ptr[0][jdx]=ann_act(hidden_vector_ptr[0][jdx]);
}
//#pragma omp parallel for private(jdx)
//	for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++){
//		for(kdx=0;kdx<KERN.hiddens[0].n_inputs;kdx++){
//			hidden_vector_ptr[0][jdx]+=KERN.hiddens[0].weights[_2D_IDX(KERN.hiddens[0].n_inputs,jdx,kdx)]*KERN.in[kdx];
//		}
//		hidden_vector_ptr[0][jdx]=ann_act(hidden_vector_ptr[0][jdx]);
//	}
/*serial*/
//	for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++){
//		for(kdx=0;kdx<KERN.hiddens[0].n_inputs;kdx++)
//			hidden_vector_ptr[0][jdx]+=KERN.hiddens[0].weights[_2D_IDX(KERN.hiddens[0].n_inputs,jdx,kdx)]*KERN.in[kdx];
//		hidden_vector_ptr[0][jdx]=ann_act(hidden_vector_ptr[0][jdx]);
//	}
#endif /*PBLAS*/
/*^^^ hidden to hidden (if any)*/
	for(idx=1;idx<KERN.n_hiddens;idx++){
#ifdef PBLAS
		cblas_dgemv(CblasRowMajor,CblasNoTrans,KERN.hiddens[idx].n_neurons,KERN.hiddens[idx].n_inputs,
			1.0,KERN.hiddens[idx].weights,KERN.hiddens[idx].n_inputs,hidden_vector_ptr[idx-1],1,0.,hidden_vector_ptr[idx],1);
#define OP_ACT(ix) hidden_vector_ptr[idx][ix]=ann_act(hidden_vector_ptr[idx][ix])
	UNROLL_OMP_FOR(0,KERN.hiddens[idx].n_neurons,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
/*serial*/
//		for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++)
//			hidden_vector_ptr[idx][jdx]=ann_act(hidden_vector_ptr[idx][jdx]);
#else /*PBLAS*/
for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++){
#define OP_WH(ix) hidden_vector_ptr[idx][jdx]+=KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,ix)]*hidden_vector_ptr[idx-1][ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[idx].n_inputs,ANN_UNROLL,WH,kdx);
#undef OP_WH
	hidden_vector_ptr[idx][jdx]=ann_act(hidden_vector_ptr[idx][jdx]);
}
//#pragma omp parallel for private(jdx)
//	for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++){
//		for(kdx=0;kdx<KERN.hiddens[idx].n_inputs;kdx++){
//			hidden_vector_ptr[idx][jdx]+=
//				KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)]*hidden_vector_ptr[idx-1][kdx];
//		}
//		hidden_vector_ptr[idx][jdx]=ann_act(hidden_vector_ptr[idx][jdx]);
//	}
/*serial*/
//		for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++){
//			for(kdx=0;kdx<KERN.hiddens[idx].n_inputs;kdx++)
//hidden_vector_ptr[idx][jdx]+=KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)]*hidden_vector_ptr[idx-1][kdx];
//			hidden_vector_ptr[idx][jdx]=ann_act(hidden_vector_ptr[idx][jdx]);
//		}
#endif /*PBLAS*/
	}
/*^^^ hidden to output*/
#ifdef PBLAS
	cblas_dgemv(CblasRowMajor,CblasNoTrans,KERN.output.n_neurons,KERN.output.n_inputs,
		1.0,KERN.output.weights,KERN.output.n_inputs,hidden_vector_ptr[KERN.n_hiddens-1],1,0.,KERN.out,1);
#define OP_ACT(ix) KERN.out[ix]=ann_act(KERN.out[ix])
	UNROLL_OMP_FOR(0,KERN.output.n_neurons,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
/*serial*/
//	for(jdx=0;jdx<KERN.output.n_neurons;jdx++) KERN.out[jdx]=ann_act(KERN.out[jdx]);
#else /*PBLAS*/
for(jdx=0;jdx<KERN.output.n_neurons;jdx++){
#define OP_WH(ix) KERN.out[jdx]+=KERN.output.weights[_2D_IDX(KERN.output.n_inputs,jdx,ix)]*hidden_vector_ptr[KERN.n_hiddens-1][ix]
	UNROLL_OMP_FOR(0,KERN.output.n_inputs,ANN_UNROLL,WH,kdx);
#undef OP_WH
	KERN.out[jdx]=ann_act(KERN.out[jdx]);
}
//#pragma omp parallel for private(jdx)
//	for(jdx=0;jdx<KERN.output.n_neurons;jdx++){
//		KERN.out[jdx]=0.;
//		for(kdx=0;kdx<KERN.output.n_inputs;kdx++){
//			KERN.out[jdx]+=
//				KERN.output.weights[_2D_IDX(KERN.output.n_inputs,jdx,kdx)]*hidden_vector_ptr[KERN.n_hiddens-1][kdx];
//		}
//		KERN.out[jdx]=ann_act(KERN.out[jdx]);
//	}
/*serial*/
//	for(jdx=0;jdx<KERN.output.n_neurons;jdx++){
//		for(kdx=0;kdx<KERN.output.n_inputs;kdx++)
//			KERN.out[jdx]+=KERN.output.weights[_2D_IDX(KERN.output.n_inputs,jdx,kdx)]*hidden_vector_ptr[KERN.n_hiddens-1][kdx];
//		KERN.out[jdx]=ann_act(KERN.out[jdx]);
//	}
#endif /*PBLAS*/
	/*all done, calculate a preliminary error*/
	for(idx=0;idx<KERN.n_outputs;idx++) Ep+=(train[idx]-KERN.out[idx])*(train[idx]-KERN.out[idx]);
	Ep*=0.5;
//	fprintf(stdout,"TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - calculate deltas +++*/
/*^^^ output*/
#define OP_DELTA(ix) delta_ptr[KERN.n_hiddens][ix]=(train[ix]-KERN.out[ix])*ann_dact(KERN.out[ix])
	UNROLL_OMP_FOR(0,KERN.output.n_neurons,ANN_UNROLL,DELTA,idx);
#undef OP_DELTA
/*serial*/
//	for(idx=0;idx<KERN.n_outputs;idx++)
//		delta_ptr[KERN.n_hiddens][idx]=(train[idx]-KERN.out[idx])*ann_dact(KERN.out[idx]);
/*^^^ output to hidden*/
#ifdef PBLAS
	/*! transposed*/
	cblas_dgemv(CblasRowMajor,CblasTrans,KERN.output.n_neurons,KERN.output.n_inputs,
	1.0,KERN.output.weights,KERN.output.n_inputs,delta_ptr[KERN.n_hiddens],1,0.,delta_ptr[KERN.n_hiddens-1],1);
#define OP_DACT(ix) delta_ptr[KERN.n_hiddens-1][ix]*=ann_dact(hidden_vector_ptr[KERN.n_hiddens-1][ix])
	UNROLL_OMP_FOR(0,KERN.output.n_inputs,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
/*serial*/
//	for(jdx=0;jdx<KERN.output.n_inputs;jdx++)
//		delta_ptr[KERN.n_hiddens-1][jdx]*=ann_dact(hidden_vector_ptr[KERN.n_hiddens-1][jdx]);
#else /*PBLAS*/
for(jdx=0;jdx<KERN.output.n_inputs;jdx++){
#define OP_WD(ix) delta_ptr[KERN.n_hiddens-1][jdx]+=KERN.output.weights[_2D_IDX(KERN.output.n_inputs,ix,jdx)]*delta_ptr[KERN.n_hiddens][ix]
	UNROLL_OMP_FOR(0,KERN.output.n_neurons,ANN_UNROLL,WD,kdx);
#undef OP_WD
	delta_ptr[KERN.n_hiddens-1][jdx]*=ann_dact(hidden_vector_ptr[KERN.n_hiddens-1][jdx]);
}
//#pragma omp parallel for private(jdx)
//	for(jdx=0;jdx<KERN.output.n_inputs;jdx++){
//		for(kdx=0;kdx<KERN.output.n_neurons;kdx++){
//			delta_ptr[KERN.n_hiddens-1][jdx]+=
//				KERN.output.weights[_2D_IDX(KERN.output.n_inputs,kdx,jdx)]*delta_ptr[KERN.n_hiddens][kdx];
//		}
//		delta_ptr[KERN.n_hiddens-1][jdx]*=ann_dact(hidden_vector_ptr[KERN.n_hiddens-1][jdx]);
//	}
/*serial*/
//	for(jdx=0;jdx<KERN.output.n_inputs;jdx++){
//for(kdx=0;kdx<KERN.output.n_neurons;kdx++)
//	delta_ptr[KERN.n_hiddens-1][jdx]+=KERN.output.weights[_2D_IDX(KERN.output.n_inputs,kdx,jdx)]*delta_ptr[KERN.n_hiddens][kdx];
//delta_ptr[KERN.n_hiddens-1][jdx]*=ann_dact(hidden_vector_ptr[KERN.n_hiddens-1][jdx]);
//}
#endif /*PBLAS*/
/*^^^ hidden to hidden (if any)*/
	if(KERN.n_hiddens>1){
#ifdef PBLAS
		for(idx=(KERN.n_hiddens-2);idx>1;idx--){
			/*! transposed*/
			cblas_dgemv(CblasRowMajor,CblasTrans,KERN.hiddens[idx+1].n_neurons,KERN.hiddens[idx+1].n_inputs,
			1.0,KERN.hiddens[idx+1].weights,KERN.hiddens[idx+1].n_inputs,delta_ptr[idx+1],1,0.,delta_ptr[idx],1);
#define OP_DACT(ix) delta_ptr[idx][ix]*=ann_dact(hidden_vector_ptr[idx][ix])
			UNROLL_OMP_FOR(0,KERN.hiddens[idx].n_neurons,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
/*serial*/
//			for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++)
//				delta_ptr[idx][jdx]*=ann_dact(hidden_vector_ptr[idx][jdx]);
		}
		/*add zero*/
		/*! transposed*/
		cblas_dgemv(CblasRowMajor,CblasTrans,KERN.hiddens[1].n_neurons,KERN.hiddens[1].n_inputs,
		1.0,KERN.hiddens[1].weights,KERN.hiddens[1].n_inputs,delta_ptr[1],1,0.,delta_ptr[0],1);
#define OP_DACT(ix) delta_ptr[0][ix]*=ann_dact(hidden_vector_ptr[0][ix])
		UNROLL_OMP_FOR(0,KERN.hiddens[0].n_neurons,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
/*serial*/
//		for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++)
//			delta_ptr[0][jdx]*=ann_dact(hidden_vector_ptr[0][jdx]);
#else /*PBLAS*/
		for(idx=(KERN.n_hiddens-2);idx>1;idx--){
for(jdx=0;jdx<KERN.hiddens[idx+1].n_inputs;jdx++){
#define OP_WD(ix) delta_ptr[idx][jdx]+=KERN.hiddens[idx+1].weights[_2D_IDX(KERN.hiddens[idx+1].n_inputs,ix,jdx)]*delta_ptr[idx+1][ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[idx+1].n_neurons,ANN_UNROLL,WD,kdx);
#undef OP_WD
	delta_ptr[idx][jdx]*=ann_dact(hidden_vector_ptr[idx][jdx]);
}
//#pragma omp parallel for private(jdx)
//			for(jdx=0;jdx<KERN.hiddens[idx+1].n_inputs;jdx++){
//				for(kdx=0;kdx<KERN.hiddens[idx+1].n_neurons;kdx++){
//					delta_ptr[idx][jdx]+=
//					KERN.hiddens[idx+1].weights[_2D_IDX(KERN.hiddens[idx+1].n_inputs,kdx,jdx)]*delta_ptr[idx+1][kdx];
//				}
//				delta_ptr[idx][jdx]*=ann_dact(hidden_vector_ptr[idx][jdx]);
//			}
/*serial*/
//			for(jdx=0;jdx<KERN.hiddens[idx+1].n_inputs;jdx++){
//for(kdx=0;kdx<KERN.hiddens[idx+1].n_neurons;kdx++)
//	delta_ptr[idx][jdx]+=KERN.hiddens[idx+1].weights[_2D_IDX(KERN.hiddens[idx+1].n_neurons,kdx,jdx)]*delta_ptr[idx+1][kdx];
//delta_ptr[idx][jdx]*=ann_dact(hidden_vector_ptr[idx][jdx]);
//			}
		}
		/*add zero*/
for(jdx=0;jdx<KERN.hiddens[1].n_inputs;jdx++){
#define OP_WD(ix) delta_ptr[0][jdx]+=KERN.hiddens[1].weights[_2D_IDX(KERN.hiddens[1].n_inputs,ix,jdx)]*delta_ptr[1][ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[1].n_neurons,ANN_UNROLL,WD,kdx);
#undef OP_WD
	delta_ptr[0][jdx]*=ann_dact(hidden_vector_ptr[0][jdx]);
}
//#pragma omp parallel for private(jdx)
//		for(jdx=0;jdx<KERN.hiddens[1].n_inputs;jdx++){
//			for(kdx=0;kdx<KERN.hiddens[1].n_neurons;kdx++){
//				delta_ptr[0][jdx]+=
//				KERN.hiddens[1].weights[_2D_IDX(KERN.hiddens[1].n_inputs,kdx,jdx)]*delta_ptr[1][kdx];
//			}
//			delta_ptr[0][jdx]*=ann_dact(hidden_vector_ptr[0][jdx]);
//		}
/*serial*/
//		for(jdx=0;jdx<KERN.hiddens[1].n_inputs;jdx++){
//for(kdx=0;kdx<KERN.hiddens[1].n_neurons;kdx++)
//	delta_ptr[0][jdx]+=KERN.hiddens[1].weights[_2D_IDX(KERN.hiddens[1].n_inputs,kdx,jdx)]*delta_ptr[1][kdx];
//delta_ptr[0][jdx]*=ann_dact(hidden_vector_ptr[0][jdx]);
//		}
#endif /*PBLAS*/
	}
/*+++ III - back propagation +++*/
/*^^^ output*/
#ifdef PBLAS
	cblas_dger(CblasRowMajor,KERN.output.n_neurons,KERN.output.n_inputs,LEARN_RATE,delta_ptr[KERN.n_hiddens],
	1,hidden_vector_ptr[KERN.n_hiddens-1],1,KERN.output.weights,KERN.output.n_inputs);
#else /*PBLAS*/
for(idx=0;idx<KERN.output.n_neurons;idx++){
#define OP_DH(ix) KERN.output.weights[_2D_IDX(KERN.output.n_inputs,idx,ix)]+=\
	LEARN_RATE*delta_ptr[KERN.n_hiddens][idx]*hidden_vector_ptr[KERN.n_hiddens-1][ix]
	UNROLL_OMP_FOR(0,KERN.output.n_inputs,ANN_UNROLL,DH,jdx);
#undef OP_DH
}
//#pragma omp parallel for private(idx)
//	for(idx=0;idx<KERN.output.n_neurons;idx++){
//		for(jdx=0;jdx<KERN.output.n_inputs;jdx++){
//			KERN.output.weights[_2D_IDX(KERN.output.n_inputs,idx,jdx)]+=
//			LEARN_RATE*delta_ptr[KERN.n_hiddens][idx]*hidden_vector_ptr[KERN.n_hiddens-1][jdx];
//		}
//	}
/*serial*/
//	for(idx=0;idx<KERN.n_outputs;idx++)
//		for(jdx=0;jdx<KERN.output.n_inputs;jdx++)
//KERN.output.weights[_2D_IDX(KERN.output.n_inputs,idx,jdx)]+=LEARN_RATE*delta_ptr[KERN.n_hiddens][idx]*hidden_vector_ptr[KERN.n_hiddens-1][jdx];
#endif /*PBLAS*/
/*^^^ hiddens*/
#ifdef PBLAS
/*serial*/
	for(idx=(KERN.n_hiddens-1);idx>1;idx--){
		cblas_dger(CblasRowMajor,KERN.hiddens[idx].n_neurons,KERN.hiddens[idx].n_inputs,LEARN_RATE,delta_ptr[idx],
		1,hidden_vector_ptr[idx-1],1,KERN.hiddens[idx].weights,KERN.hiddens[idx].n_inputs);
	}
	/*add zero*/
	cblas_dger(CblasRowMajor,KERN.hiddens[0].n_neurons,KERN.hiddens[0].n_inputs,LEARN_RATE,delta_ptr[0],
	1,KERN.in,1,KERN.hiddens[0].weights,KERN.hiddens[0].n_inputs);
#else /*PBLAS*/
for(idx=(KERN.n_hiddens-1);idx>1;idx--){
#define OP_DH(ix) KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,ix)]+=\
	LEARN_RATE*delta_ptr[idx][jdx]*hidden_vector_ptr[idx-1][ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[idx].n_inputs,ANN_UNROLL,DH,kdx);
#undef OP_DH
}
//#pragma omp parallel for private(jdx)
//		for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++){
//			for(kdx=0;kdx<KERN.hiddens[idx].n_inputs;kdx++){
//				KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)]+=
//				LEARN_RATE*delta_ptr[idx][jdx]*hidden_vector_ptr[idx-1][kdx];
//			}
//		}
//	}
/*serial*/
//	for(idx=(KERN.n_hiddens-1);idx>1;idx--)
//		for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++)
//			for(kdx=0;kdx<KERN.hiddens[idx].n_inputs;kdx++)
//KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)]+=LEARN_RATE*delta_ptr[idx][jdx]*hidden_vector_ptr[idx-1][kdx];
	/*add zero*/
for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++){
#define OP_DI(ix) KERN.hiddens[0].weights[_2D_IDX(KERN.hiddens[0].n_inputs,jdx,ix)]+=LEARN_RATE*delta_ptr[0][jdx]*KERN.in[ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[0].n_inputs,ANN_UNROLL,DI,kdx);
#undef OP_DI
}
//#pragma omp parallel for private(jdx)
//	for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++){
//		for(kdx=0;kdx<KERN.hiddens[0].n_inputs;kdx++){
//			KERN.hiddens[0].weights[_2D_IDX(KERN.hiddens[0].n_inputs,jdx,kdx)]+=LEARN_RATE*delta_ptr[0][jdx]*KERN.in[kdx];
//		}
//	}
/*serial*/
//	for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++)
//		for(kdx=0;kdx<KERN.hiddens[0].n_inputs;kdx++)
//KERN.hiddens[0].weights[_2D_IDX(KERN.hiddens[0].n_inputs,jdx,kdx)]+=LEARN_RATE*delta_ptr[0][jdx]*KERN.in[kdx];
#endif /*PBLAS*/
/*+++ IV - update error +++*/
	ann_kernel_run(kernel);
	for(idx=0;idx<KERN.n_outputs;idx++) Epr+=(train[idx]-KERN.out[idx])*(train[idx]-KERN.out[idx]);
	Epr*=0.5;
//	fprintf(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ IV - cleanup +++*/
	for(idx=0;idx<KERN.n_hiddens;idx++){
		FREE(hidden_vector_ptr[idx]);
		hidden_vector_ptr[idx]=NULL;
	}
	FREE(hidden_vector_ptr);
	for(idx=0;idx<(KERN.n_hiddens+1);idx++){
		FREE(delta_ptr[idx]);
		delta_ptr[idx]=NULL;
	}
	FREE(delta_ptr);
//	return sqrt((Ep-Epr)*(Ep-Epr));
	return Ep-Epr;
}
/*---------------------------------*/
/*+++ momentum back-propagation +++*/
/*---------------------------------*/
void ann_momentum_init(_kernel *kernel){
	UINT idx;
	UINT64 allocate=0.;
	/**/
	ALLOC_REPORT(KERN.dw,KERN.n_hiddens+1,DOUBLE *,allocate);
	ALLOC_REPORT(KERN.dw[KERN.n_hiddens],KERN.output.n_inputs*KERN.output.n_neurons,DOUBLE,allocate);
	for(idx=0;idx<KERN.n_hiddens;idx++){
		ALLOC_REPORT(KERN.dw[idx],KERN.hiddens[idx].n_inputs*KERN.hiddens[idx].n_neurons,DOUBLE,allocate);
	}
	fprintf(stdout,"TRAINING MOMENTUM ALLOC: %lu (bytes)\n",allocate);
}
void ann_raz_momentum(_kernel *kernel){
	UINT idx;
	memset(KERN.dw[0],0,sizeof(DOUBLE)*KERN.output.n_inputs*KERN.output.n_neurons);
	for(idx=0;idx<KERN.n_hiddens;idx++)
		memset(KERN.dw[idx],0,sizeof(DOUBLE)*KERN.hiddens[idx].n_inputs*KERN.hiddens[idx].n_neurons);
}
void ann_empty_momentum(_kernel *kernel){
	UINT idx;
	FREE(KERN.dw[KERN.n_hiddens]);
	for(idx=0;idx<KERN.n_hiddens;idx++){
		FREE(KERN.dw[idx]);
		KERN.dw[idx]=NULL;
	}
	FREE(KERN.dw);
	KERN.dw=NULL;
}
DOUBLE ann_kernel_train_momentum(_kernel *kernel,const DOUBLE *train,DOUBLE alpha){
	UINT idx, jdx;
#ifndef PBLAS
	UINT kdx;
#endif
	DOUBLE Ep=0.;
	DOUBLE Epr=0.;
	DOUBLE **hidden_vector_ptr;
	DOUBLE **delta_ptr;
	/*keep a track of mem*/
	UINT64 allocate=0.;
/*+++ I - forward +++*/
	ALLOC_REPORT(hidden_vector_ptr,KERN.n_hiddens,DOUBLE *,allocate);
	for(idx=0;idx<KERN.n_hiddens;idx++){
		ALLOC_REPORT(hidden_vector_ptr[idx],KERN.hiddens[idx].n_neurons,DOUBLE,allocate);
	}
	ALLOC_REPORT(delta_ptr,KERN.n_hiddens+1,DOUBLE *,allocate);/*+1 for OUTPUT*/
	ALLOC_REPORT(delta_ptr[KERN.n_hiddens],KERN.n_outputs,DOUBLE,allocate);
	for(idx=0;idx<KERN.n_hiddens;idx++)
		ALLOC_REPORT(delta_ptr[idx],KERN.hiddens[idx].n_neurons,DOUBLE,allocate);
//fprintf(stdout,"TRAINING MEM ALLOC: %lu (bytes)\n",allocate);
/*^^^ input to hidden*/
#ifdef PBLAS
	cblas_dgemv(CblasRowMajor,CblasNoTrans,KERN.hiddens[0].n_neurons,KERN.hiddens[0].n_inputs,
		1.0,KERN.hiddens[0].weights,KERN.hiddens[0].n_inputs,KERN.in,1,0.,hidden_vector_ptr[0],1);
#define OP_ACT(ix) hidden_vector_ptr[0][ix]=ann_act(hidden_vector_ptr[0][ix])
	UNROLL_OMP_FOR(0,KERN.hiddens[0].n_neurons,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
#else /*PBLAS*/
for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++){
#define OP_WI(ix) hidden_vector_ptr[0][jdx]+=KERN.hiddens[0].weights[_2D_IDX(KERN.hiddens[0].n_inputs,jdx,ix)]*KERN.in[ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[0].n_inputs,ANN_UNROLL,WI,kdx);
#undef OP_WI
	hidden_vector_ptr[0][jdx]=ann_act(hidden_vector_ptr[0][jdx]);
}
//#pragma omp parallel for private(jdx)
//	for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++){
//		for(kdx=0;kdx<KERN.hiddens[0].n_inputs;kdx++){
//			hidden_vector_ptr[0][jdx]+=KERN.hiddens[0].weights[_2D_IDX(KERN.hiddens[0].n_inputs,jdx,kdx)]*KERN.in[kdx];
//		}
//		hidden_vector_ptr[0][jdx]=ann_act(hidden_vector_ptr[0][jdx]);
//	}
#endif /*PBLAS*/
/*^^^ hidden to hidden (if any)*/
	for(idx=1;idx<KERN.n_hiddens;idx++){
#ifdef PBLAS
		cblas_dgemv(CblasRowMajor,CblasNoTrans,KERN.hiddens[idx].n_neurons,KERN.hiddens[idx].n_inputs,
			1.0,KERN.hiddens[idx].weights,KERN.hiddens[idx].n_inputs,hidden_vector_ptr[idx-1],1,0.,hidden_vector_ptr[idx],1);
#define OP_ACT(ix) hidden_vector_ptr[idx][ix]=ann_act(hidden_vector_ptr[idx][ix])
	UNROLL_OMP_FOR(0,KERN.hiddens[idx].n_neurons,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
#else /*PBLAS*/
for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++){
#define OP_WH(ix) hidden_vector_ptr[idx][jdx]+=KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,ix)]*hidden_vector_ptr[idx-1][ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[idx].n_inputs,ANN_UNROLL,WH,kdx);
#undef OP_WH
	hidden_vector_ptr[idx][jdx]=ann_act(hidden_vector_ptr[idx][jdx]);
}
//#pragma omp parallel for private(jdx)
//	for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++){
//		for(kdx=0;kdx<KERN.hiddens[idx].n_inputs;kdx++){
//			hidden_vector_ptr[idx][jdx]+=
//				KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)]*hidden_vector_ptr[idx-1][kdx];
//		}
//		hidden_vector_ptr[idx][jdx]=ann_act(hidden_vector_ptr[idx][jdx]);
//	}
#endif /*PBLAS*/
	}
/*^^^ hidden to output*/
#ifdef PBLAS
	cblas_dgemv(CblasRowMajor,CblasNoTrans,KERN.output.n_neurons,KERN.output.n_inputs,
		1.0,KERN.output.weights,KERN.output.n_inputs,hidden_vector_ptr[KERN.n_hiddens-1],1,0.,KERN.out,1);
#define OP_ACT(ix) KERN.out[ix]=ann_act(KERN.out[ix])
	UNROLL_OMP_FOR(0,KERN.output.n_neurons,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
#else /*PBLAS*/
for(jdx=0;jdx<KERN.output.n_neurons;jdx++){
	KERN.out[jdx]=0.;/*TRAP*/
#define OP_WH(ix) KERN.out[jdx]+=KERN.output.weights[_2D_IDX(KERN.output.n_inputs,jdx,ix)]*hidden_vector_ptr[KERN.n_hiddens-1][ix]
	UNROLL_OMP_FOR(0,KERN.output.n_inputs,ANN_UNROLL,WH,kdx);
#undef OP_WH
	KERN.out[jdx]=ann_act(KERN.out[jdx]);
}
//#pragma omp parallel for private(jdx)
//	for(jdx=0;jdx<KERN.output.n_neurons;jdx++){
//		KERN.out[jdx]=0.;
//		for(kdx=0;kdx<KERN.output.n_inputs;kdx++){
//			KERN.out[jdx]+=
//				KERN.output.weights[_2D_IDX(KERN.output.n_inputs,jdx,kdx)]*hidden_vector_ptr[KERN.n_hiddens-1][kdx];
//		}
//		KERN.out[jdx]=ann_act(KERN.out[jdx]);
//	}
#endif /*PBLAS*/
	/*all done, calculate a preliminary error*/
	for(idx=0;idx<KERN.n_outputs;idx++) Ep+=(train[idx]-KERN.out[idx])*(train[idx]-KERN.out[idx]);
	Ep*=0.5;
//	fprintf(stdout,"TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - calculate deltas +++*/
/*^^^ output*/
#define OP_DELTA(ix) delta_ptr[KERN.n_hiddens][ix]=(train[ix]-KERN.out[ix])*ann_dact(KERN.out[ix])
	UNROLL_OMP_FOR(0,KERN.output.n_neurons,ANN_UNROLL,DELTA,idx);
#undef OP_DELTA
/*^^^ output to hidden*/
#ifdef PBLAS
	/*! transposed*/
	cblas_dgemv(CblasRowMajor,CblasTrans,KERN.output.n_neurons,KERN.output.n_inputs,
	1.0,KERN.output.weights,KERN.output.n_inputs,delta_ptr[KERN.n_hiddens],1,0.,delta_ptr[KERN.n_hiddens-1],1);
#define OP_DACT(ix) delta_ptr[KERN.n_hiddens-1][ix]*=ann_dact(hidden_vector_ptr[KERN.n_hiddens-1][ix])
	UNROLL_OMP_FOR(0,KERN.output.n_inputs,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
#else /*PBLAS*/
for(jdx=0;jdx<KERN.output.n_inputs;jdx++){
#define OP_WD(ix) delta_ptr[KERN.n_hiddens-1][jdx]+=KERN.output.weights[_2D_IDX(KERN.output.n_inputs,ix,jdx)]*delta_ptr[KERN.n_hiddens][ix]
	UNROLL_OMP_FOR(0,KERN.output.n_neurons,ANN_UNROLL,WD,kdx);
#undef OP_WD
	delta_ptr[KERN.n_hiddens-1][jdx]*=ann_dact(hidden_vector_ptr[KERN.n_hiddens-1][jdx]);
}
//#pragma omp parallel for private(jdx)
//	for(jdx=0;jdx<KERN.output.n_inputs;jdx++){
//		for(kdx=0;kdx<KERN.output.n_neurons;kdx++){
//			delta_ptr[KERN.n_hiddens-1][jdx]+=
//				KERN.output.weights[_2D_IDX(KERN.output.n_inputs,kdx,jdx)]*delta_ptr[KERN.n_hiddens][kdx];
//		}
//		delta_ptr[KERN.n_hiddens-1][jdx]*=ann_dact(hidden_vector_ptr[KERN.n_hiddens-1][jdx]);
//	}
#endif /*PBLAS*/
/*^^^ hidden to hidden (if any)*/
	if(KERN.n_hiddens>1){
#ifdef PBLAS
		for(idx=(KERN.n_hiddens-2);idx>1;idx--){
			/*! transposed*/
			cblas_dgemv(CblasRowMajor,CblasTrans,KERN.hiddens[idx+1].n_neurons,KERN.hiddens[idx+1].n_inputs,
			1.0,KERN.hiddens[idx+1].weights,KERN.hiddens[idx+1].n_inputs,delta_ptr[idx+1],1,0.,delta_ptr[idx],1);
#define OP_DACT(ix) delta_ptr[idx][ix]*=ann_dact(hidden_vector_ptr[idx][ix])
			UNROLL_OMP_FOR(0,KERN.hiddens[idx].n_neurons,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
		}
		/*add zero*/
		/*! transposed*/
		cblas_dgemv(CblasRowMajor,CblasTrans,KERN.hiddens[1].n_neurons,KERN.hiddens[1].n_inputs,
		1.0,KERN.hiddens[1].weights,KERN.hiddens[1].n_inputs,delta_ptr[1],1,0.,delta_ptr[0],1);
#define OP_DACT(ix) delta_ptr[0][ix]*=ann_dact(hidden_vector_ptr[0][ix])
		UNROLL_OMP_FOR(0,KERN.hiddens[0].n_neurons,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
#else /*PBLAS*/
		for(idx=(KERN.n_hiddens-2);idx>1;idx--){
for(jdx=0;jdx<KERN.hiddens[idx+1].n_inputs;jdx++){
#define OP_WD(ix) delta_ptr[idx][jdx]+=KERN.hiddens[idx+1].weights[_2D_IDX(KERN.hiddens[idx+1].n_inputs,ix,jdx)]*delta_ptr[idx+1][ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[idx+1].n_neurons,ANN_UNROLL,WD,kdx);
#undef OP_WD
	delta_ptr[idx][jdx]*=ann_dact(hidden_vector_ptr[idx][jdx]);
}
//#pragma omp parallel for private(jdx)
//			for(jdx=0;jdx<KERN.hiddens[idx+1].n_inputs;jdx++){
//				for(kdx=0;kdx<KERN.hiddens[idx+1].n_neurons;kdx++){
//					delta_ptr[idx][jdx]+=
//					KERN.hiddens[idx+1].weights[_2D_IDX(KERN.hiddens[idx+1].n_inputs,kdx,jdx)]*delta_ptr[idx+1][kdx];
//				}
//				delta_ptr[idx][jdx]*=ann_dact(hidden_vector_ptr[idx][jdx]);
//			}
		}
		/*add zero*/
for(jdx=0;jdx<KERN.hiddens[1].n_inputs;jdx++){
#define OP_WD(ix) delta_ptr[0][jdx]+=KERN.hiddens[1].weights[_2D_IDX(KERN.hiddens[1].n_inputs,ix,jdx)]*delta_ptr[1][ix]
	UNROLL_OMP_FOR(0,KERN.hiddens[1].n_neurons,ANN_UNROLL,WD,kdx);
#undef OP_WD
	delta_ptr[0][jdx]*=ann_dact(hidden_vector_ptr[0][jdx]);
}
//#pragma omp parallel for private(jdx)
//		for(jdx=0;jdx<KERN.hiddens[1].n_inputs;jdx++){
//			for(kdx=0;kdx<KERN.hiddens[1].n_neurons;kdx++){
//				delta_ptr[0][jdx]+=
//				KERN.hiddens[1].weights[_2D_IDX(KERN.hiddens[1].n_inputs,kdx,jdx)]*delta_ptr[1][kdx];
//			}
//			delta_ptr[0][jdx]*=ann_dact(hidden_vector_ptr[0][jdx]);
//		}
#endif /*PBLAS*/
	}
/*+++ III - back propagation +++*/
/*^^^ output*/
#ifdef PBLAS
	/*unfortunately dger output can't be scaled*/
	cblas_dger(CblasRowMajor,KERN.output.n_neurons,KERN.output.n_inputs,LEARN_RATE,delta_ptr[KERN.n_hiddens],
	1,hidden_vector_ptr[KERN.n_hiddens-1],1,KERN.dw[KERN.n_hiddens],KERN.output.n_inputs);
	cblas_daxpy(KERN.output.n_inputs*KERN.output.n_neurons,1.0,KERN.dw[KERN.n_hiddens],1,KERN.output.weights,1);
	cblas_dscal(KERN.output.n_inputs*KERN.output.n_neurons,alpha,KERN.dw[KERN.n_hiddens],1);
#else /*PBLAS*/
#pragma omp parallel for private(idx)
	for(idx=0;idx<KERN.output.n_neurons;idx++){
		for(jdx=0;jdx<KERN.output.n_inputs;jdx++){
			KERN.dw[KERN.n_hiddens][idx*KERN.output.n_inputs+jdx]+=LEARN_RATE*delta_ptr[KERN.n_hiddens][idx]*hidden_vector_ptr[KERN.n_hiddens-1][jdx];
			KERN.output.weights[_2D_IDX(KERN.output.n_inputs,idx,jdx)]+=KERN.dw[KERN.n_hiddens][idx*KERN.output.n_inputs+jdx];
			KERN.dw[KERN.n_hiddens][idx*KERN.output.n_inputs+jdx]*=alpha;
		}
	}
#endif /*PBLAS*/
/*^^^ hiddens*/
#ifdef PBLAS
/*serial*/
	for(idx=(KERN.n_hiddens-1);idx>1;idx--){
		cblas_dger(CblasRowMajor,KERN.hiddens[idx].n_neurons,KERN.hiddens[idx].n_inputs,LEARN_RATE,delta_ptr[idx],
		1,hidden_vector_ptr[idx-1],1,KERN.dw[idx],KERN.hiddens[idx].n_inputs);
		cblas_daxpy(KERN.hiddens[idx].n_inputs*KERN.hiddens[idx].n_neurons,1.0,KERN.dw[idx],1,KERN.hiddens[idx].weights,1);
		cblas_dscal(KERN.hiddens[idx].n_inputs*KERN.hiddens[idx].n_neurons,alpha,KERN.dw[idx],1);
	}
	/*add zero*/
	cblas_dger(CblasRowMajor,KERN.hiddens[0].n_neurons,KERN.hiddens[0].n_inputs,LEARN_RATE,delta_ptr[0],
		1,KERN.in,1,KERN.dw[0],KERN.hiddens[0].n_inputs);
	cblas_daxpy(KERN.hiddens[0].n_inputs*KERN.hiddens[0].n_neurons,1.0,KERN.dw[0],1,KERN.hiddens[0].weights,1);
	cblas_dscal(KERN.hiddens[0].n_inputs*KERN.hiddens[0].n_neurons,alpha,KERN.dw[0],1);
#else /*PBLAS*/
	for(idx=(KERN.n_hiddens-1);idx>1;idx--){
#pragma omp parallel for private(jdx)
		for(jdx=0;jdx<KERN.hiddens[idx].n_neurons;jdx++){
			for(kdx=0;kdx<KERN.hiddens[idx].n_inputs;kdx++){
				KERN.dw[idx][_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)]+=
					LEARN_RATE*delta_ptr[idx][jdx]*hidden_vector_ptr[idx-1][kdx];
				KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)]+=
					KERN.dw[idx][_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)];
				KERN.dw[idx][_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,kdx)]*=alpha;
			}
		}
	}
	/*add zero*/
#pragma omp parallel for private(jdx)
	for(jdx=0;jdx<KERN.hiddens[0].n_neurons;jdx++){
		for(kdx=0;kdx<KERN.hiddens[0].n_inputs;kdx++){
			KERN.dw[0][_2D_IDX(KERN.hiddens[0].n_inputs,jdx,kdx)]+=LEARN_RATE*delta_ptr[0][jdx]*KERN.in[kdx];
			KERN.hiddens[0].weights[_2D_IDX(KERN.hiddens[0].n_inputs,jdx,kdx)]+=
				KERN.dw[0][_2D_IDX(KERN.hiddens[0].n_inputs,jdx,kdx)];
			KERN.dw[0][_2D_IDX(KERN.hiddens[0].n_inputs,jdx,kdx)]*=alpha;
		}
	}
#endif /*PBLAS*/
/*+++ IV - update error +++*/
	ann_kernel_run(kernel);
	for(idx=0;idx<KERN.n_outputs;idx++) Epr+=(train[idx]-KERN.out[idx])*(train[idx]-KERN.out[idx]);
	Epr*=0.5;
//	fprintf(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ IV - cleanup +++*/
	for(idx=0;idx<KERN.n_hiddens;idx++){
		FREE(hidden_vector_ptr[idx]);
		hidden_vector_ptr[idx]=NULL;
	}
	FREE(hidden_vector_ptr);
	for(idx=0;idx<(KERN.n_hiddens+1);idx++){
		FREE(delta_ptr[idx]);
		delta_ptr[idx]=NULL;
	}
	FREE(delta_ptr);
//	return sqrt((Ep-Epr)*(Ep-Epr));
	return Ep-Epr;
}





/*---------------------------*/
/* train ANN sample with BPM */
/*---------------------------*/
DOUBLE ann_train_BPM(_kernel *kernel,DOUBLE *train_in,DOUBLE *train_out,DOUBLE alpha,DOUBLE delta){
/*typical values alpha=0.2 delta=0.00001*/
	BOOL is_ok;
	UINT   idx;
	UINT  iter;
	DOUBLE dEp;
	DOUBLE probe;
	/*copy input*/
	ARRAY_CP(train_in,KERN.in,KERN.n_inputs);
	/**/
	ann_raz_momentum(kernel);
	ann_kernel_run(kernel);
	dEp=0.;
	for(idx=0;idx<kernel->n_outputs;idx++)
		dEp+=(train_out[idx]-kernel->out[idx])*(train_out[idx]-kernel->out[idx]);
	dEp*=0.5;
	fprintf(stdout," init=%15.10f",dEp);
	iter=0;
	do{
		dEp=ann_kernel_train_momentum(kernel,train_out,alpha);
		is_ok=FALSE;
		iter++;
		is_ok=TRUE;
		for(idx=0;idx<KERN.n_outputs;idx++){
			if(kernel->out[idx]>0.1) probe=1.0;
			else probe=-1.0;
			if(train_out[idx]!=probe) is_ok=FALSE;
		}
		if(iter==1){
			/*determine if we get a good answer at first try*/
			if(is_ok==TRUE) fprintf(stdout," OK");
			else fprintf(stdout," NO");
		}
		if(iter>10239) break;/*failsafe number of wrong iteration*/	
	}while((dEp > delta)||(!(is_ok==TRUE)));
	fprintf(stdout," N_ITER=%8i",iter);
	if(is_ok==TRUE) fprintf(stdout," SUCCESS!\n");
	else fprintf(stdout," FAIL!\n");
	fflush(stdout);
	return dEp;
}






