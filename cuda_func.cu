/* High Performance Neural Networks  -- OVHPA 2019
 * mail: hubert.valencia _at_ imass.nagoya-u.ac.jp
 * cuda_func.cu:  contains the CUDA implementation
 * of HPNN's ANN neural network routines.
*/

/*
This file is part of HPNN library.

    HPNN is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    HPNN is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common.h"
#include "ann.h"

/*This is a CUDA area for functions*/

#define _WP  32
#define _TPW 32
#define _TPB (_TPW*_WP)
#define _KG(n) ((n+_TPB-1)/(_TPB)),_TPB

__global__ 
void dbg_print(int n, double *x){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n){
		printf("val[%d]=%lf\n",i,x[i]);
	}
}

__global__
void sigmoid(int n, double *x){
#ifdef NO_THREADS
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n){
		x[i] = 2.0/(1.0+exp(-1.0*x[i]))-1.0;
	}
#else
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] = 2.0/(1.0+exp(-1.0*x[i]))-1.0;
#endif
}
__global__
void _dsigmoid(int n, double *in, double *out){
#ifdef NO_THREADS
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n){
                out[i] = (-0.5 * ( in[i] * in[i] - 1.0));
        }
#else
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i += stride)
                out[i] = (-0.5 * ( in[i] * in[i] - 1.0));
#endif
}
__global__
void dsigmoid(int n, double *in, double *out){
#ifdef NO_THREADS
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n){
		out[i] *= (-0.5 * ( in[i] * in[i] - 1.0));
	7}
#else
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		out[i] *= (-0.5 * ( in[i] * in[i] - 1.0));
#endif
}
__global__
void amb(int n, double *out, double *a, double *b){
#ifdef NO_THREADS
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n){
		out[i] = ( a[i] - b[i] ) * ( a[i] - b[i] );
	}
#else
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i += stride)
                out[i] = ( a[i] - b[i] ) * ( a[i] - b[i] );
#endif
}
__global__
void mul_diff(int n, double *t, double *o, double *y){
#ifdef NO_THREADS
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n){
		y[i] *= ( t[i] - o[i] );
	}
#else
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i += stride)
		y[i] *= ( t[i] - o[i] );
	
#endif
}
__global__
void zero_mv(int m,int n, double *mat,double *vec,double *res){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	double sum=0.;
	if(tid<m){
		for(int i=0; i<n; i++) sum += vec[i]*mat[(i*m)+tid];
		res[tid]=sum;
	}
}
__global__
void zero_tmv(int m,int n, double *mat,double *vec,double *res){
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        double sum=0.;
        if(tid<m){
                for(int i=0; i<m; i++) sum += vec[i] * mat[(tid*m)+i];
                res[tid]=sum;
        }
}
/*try*/
__global__
void fw_mv_acc(int m,int n, double *mat,double *vec,double *res){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	double sum=0.;
	if(tid<n){
		/*a full line*/
		for(int i=0; i<m; i++) sum += vec[i]*mat[(tid*m)+i];
		res[tid]=2.0/(1.0+exp(-1.0*sum))-1.0;
	}
}
__global__
void amb_acc(int n, double *out, double *a, double *b){
	extern __shared__ double sh_data[];
	int tid=threadIdx.x;
	int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
	sh_data[tid]=0.;
	if(i<n) sh_data[tid]=(a[i]-b[i])*(a[i]-b[i]) + (a[i+blockDim.x]-b[i+blockDim.x])*(a[i+blockDim.x]-b[i+blockDim.x]);
	__syncthreads();
	/*reduction in shared memory*/
	for(int s=blockDim.x/2;s>0;s>>=1){
		if(tid<s) sh_data[tid] += sh_data[tid+s];
		__syncthreads();
	}
	/*result*/
	if(tid==0) out[blockIdx.x]=sh_data[0];
}
__global__
void dsigmoid_mul_diff(int n, double *t, double *o, double *y){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i += stride){
		y[i] = ( t[i] - o[i] ) * (-0.5 * ( o[i] * o[i] - 1.0));
	}
	
}

__global__
void dsigmoid_mul_delta(int m,int n, double *w,double *d,double *h,double *res){
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        double sum=0.;
        if(tid<m){
                for(int i=0; i<m; i++) sum += d[i] * w[(tid*m)+i];
                res[tid]=sum * (-0.5 * ( h[tid] * h[tid] -1.0));
        }
}
/*TRY*/
__global__
void ger_acc(int m,int n, double *a,double *b,double *res){
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        if(tid<n){
		/*DGER, one line at a time*/
		for(int i=0; i<m; i++) res[(tid*m)+i]+=a[i]*b[tid];
        }
}

/*-----------------*/
/* The C interface */
/*-----------------*/
extern "C"{

#define _K (*kernel)

double cuda_array_dbg(cublasHandle_t cublas_handle,int n,double *gpu_in){
	double res=0.1;
	cublasDnrm2(cublas_handle,n,gpu_in,1,&res);
	return res;
}

void cuda_ann_forward_cublas(_kernel *kernel,cublasHandle_t cublas_handle){
        int idx;
        int M;
        int N;
        double *gpu_in;
#ifdef   _CUBLAS
	double _alpha=1.0;
	double _beta =0.0;
#endif /*_CUBLAS*/
#ifdef _TIMING
cudaEvent_t start, stop;
float time;
int eventflags = cudaEventBlockingSync;
cudaEventCreateWithFlags(&start,eventflags);
cudaEventCreateWithFlags(&stop,eventflags);
cudaEventRecord(start,0);
#endif
	CUDA_ALLOC(gpu_in,_K.max_index,DOUBLE);
	CUDA_G2G_CP(_K.cuda_in,gpu_in,_K.n_inputs,DOUBLE);
/*+++ I - hiddens +++*/
        for(idx=0;idx<_K.n_hiddens;idx++){
                /*GEMV + act*/
                N=_K.hiddens[idx].n_neurons;
                M=_K.hiddens[idx].n_inputs;
#ifdef   _CUBLAS
		cublasDgemv(cublas_handle,CUBLAS_OP_T,M,N,&_alpha,_K.hiddens[idx].cuda_w,M,gpu_in,1,&_beta,_K.tmp_gpu,1);
		CHK_ERR(cublas_1);
		sigmoid<<<_KG(N)>>>(N,_K.tmp_gpu);
                CHK_ERR(kernel_1);
#else  /*_CUBLAS*/
		fw_mv_acc<<<_KG(N)>>>(M,N,_K.hiddens[idx].cuda_w,gpu_in,_K.tmp_gpu);
		CHK_ERR(kernel_1);
#endif /*_CUBLAS*/
		CUDA_G2G_CP(_K.tmp_gpu,gpu_in,N,DOUBLE);
        }
/*+++ II - output +++*/
        N=_K.output.n_neurons;
        M=_K.output.n_inputs;
#ifdef   _CUBLAS
	cublasDgemv(cublas_handle,CUBLAS_OP_T,M,N,&_alpha,_K.output.cuda_w,M,gpu_in,1,&_beta,_K.cuda_out,1);
	CHK_ERR(cublas_2);
	sigmoid<<<_KG(N)>>>(N,_K.cuda_out);
        CHK_ERR(kernel_2);
#else  /*_CUBLAS*/
	fw_mv_acc<<<_KG(N)>>>(M,N,_K.output.cuda_w,gpu_in,_K.cuda_out);
	CHK_ERR(kernel_2);
#endif /*_CUBLAS*/
#ifdef _TIMING
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time,start,stop);
printf("cuda_ann_forward_cublas: time = %f\n",time);
#endif
//      cudaDeviceSynchronize();
}
void scuda_ann_forward_cublas(_kernel *kernel,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem;
	double *gpu_in;
#ifdef _CUBLAS
        double _alpha=1.0;
        double _beta =0.0;
#endif
#ifdef _TIMING
cudaEvent_t start, stop;
float time;
int eventflags = cudaEventBlockingSync;
cudaEventCreateWithFlags(&start,eventflags);
cudaEventCreateWithFlags(&stop,eventflags);
cudaEventRecord(start,0);
#endif
/*+++ I - hiddens +++*/
	CUDA_ALLOC(gpu_in,_K.max_index,DOUBLE);
	CUDA_G2G_CP(_K.cuda_in,gpu_in,_K.n_inputs,DOUBLE);
	for(idx=0;idx<_K.n_hiddens;idx++){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		red=N/cudas->cuda_n_streams;
		rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle,
				CUBLAS_OP_T,M,red,&_alpha,_K.hiddens[idx].cuda_w+jdx*M*red,M,
				gpu_in,1,&_beta,_K.tmp_gpu+jdx*red,1);
			CHK_ERR(cublas_1);
			sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,_K.tmp_gpu+jdx*red);
			CHK_ERR(kernel_1);
		}
		/*launch the last kernel*/
		jdx=cudas->cuda_n_streams-1;/*necessary?*/
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle,
			CUBLAS_OP_T,M,red+rem,&_alpha,_K.hiddens[idx].cuda_w+jdx*M*red,M,
			gpu_in,1,&_beta,_K.tmp_gpu+jdx*red,1);
		CHK_ERR(cublas_1);
		sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,_K.tmp_gpu+jdx*red);
		CHK_ERR(kernel_1);
		/*now wait for everyone*/
		cudaDeviceSynchronize();
#else  /*_CUBLAS*/
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,_K.hiddens[idx].cuda_w+jdx*M*red,gpu_in,_K.tmp_gpu+jdx*red);
			CHK_ERR(kernel_1);
		}
		jdx=cudas->cuda_n_streams-1;/*necessary?*/
		fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,_K.hiddens[idx].cuda_w+jdx*M*red,gpu_in,_K.tmp_gpu+jdx*red);
		CHK_ERR(kernel_1);
		/*now wait for everyone*/
		cudaDeviceSynchronize();
#endif /*_CUBLAS*/
		/*now copy back _K.tmp_gpu to gpu_in*/
		CUDA_G2G_CP(_K.tmp_gpu,gpu_in,N,DOUBLE);
		CHK_ERR(sync_1);
	}
//M=_K.output.n_inputs;
//dbg_print<<<(M+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M,gpu_in);
/*+++ II - output +++*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle,
			CUBLAS_OP_T,M,red,&_alpha,_K.output.cuda_w+jdx*M*red,M,
			gpu_in,1,&_beta,_K.cuda_out+jdx*red,1);
		CHK_ERR(cublas_2);
		sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,_K.cuda_out+jdx*red);
		CHK_ERR(kernel_2);
	}
	jdx=cudas->cuda_n_streams-1;/*necessary?*/
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle,
		CUBLAS_OP_T,M,red+rem,&_alpha,_K.output.cuda_w+jdx*M*red,M,
		gpu_in,1,&_beta,_K.cuda_out+jdx*red,1);
	CHK_ERR(cublas_2);
	sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,_K.cuda_out+jdx*red);
	CHK_ERR(kernel_2);
	/*now wait for everyone*/
	cudaDeviceSynchronize();
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.output.cuda_w+jdx*M*red,gpu_in,_K.cuda_out+jdx*red);
		CHK_ERR(kernel_2);
	}
	jdx=cudas->cuda_n_streams-1;/*necessary?*/
	fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_K.output.cuda_w+jdx*M*red,gpu_in,_K.cuda_out+jdx*red);
	CHK_ERR(kernel_2);
	/*now wait for everyone*/
	cudaDeviceSynchronize();
#endif /*_CUBLAS*/
	CUDA_FREE(gpu_in);
#ifdef _TIMING
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time,start,stop);
printf("scuda_ann_forward_cublas: streams = %i time = %f\n",cudas->cuda_n_streams,time);
#endif
}

#define LEARN_RATE 0.01
double cuda_ann_train_cublas(_kernel *kernel,double *train,cudastreams *cudas){
	cublasHandle_t cublas_handle=cudas->cuda_handle;
	int idx;
	int M;
	int N;
	double *tmp_gpu;
	double **hidden_vector_ptr;
	double **delta_ptr;
	double Ep =0.;
	double Epr=0.;
	/**/
	double _alpha=1.0;
	double _beta =0.0;
	/*allocate*/
	CUDA_ALLOC(tmp_gpu,_K.n_outputs,DOUBLE);
	ALLOC(hidden_vector_ptr,_K.n_hiddens,DOUBLE *);/*HOST*/
	for(idx=0;idx<_K.n_hiddens;idx++) CUDA_ALLOC(hidden_vector_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
	ALLOC(delta_ptr,_K.n_hiddens+1,DOUBLE *);/*HOST*/
	for(idx=0;idx<_K.n_hiddens;idx++) CUDA_ALLOC(delta_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
	CUDA_ALLOC(delta_ptr[_K.n_hiddens],_K.n_outputs,DOUBLE);/*DEVICE*/
/*+++ I - FORWARD +++*/
/*^^^ input to hidden +++*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	cublasDgemv(cublas_handle,CUBLAS_OP_T,M,N,&_alpha,_K.hiddens[0].cuda_w,M,_K.cuda_in,1,&_beta,hidden_vector_ptr[0],1);
	CHK_ERR(cublas_1);
	sigmoid<<<_KG(N)>>>(N,hidden_vector_ptr[0]);
	CHK_ERR(kernel_1);
/*^^^ hidden to hidden (if any)*/
	for(idx=1;idx<_K.n_hiddens;idx++){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		CUBLAS_ERR(cublasDgemv(cublas_handle,CUBLAS_OP_T,M,N,&_alpha,_K.hiddens[idx].cuda_w,M,hidden_vector_ptr[idx-1],1,&_beta,hidden_vector_ptr[idx],1));
		CHK_ERR(cublas_2);
		sigmoid<<<_KG(N)>>>(N,hidden_vector_ptr[idx]);
		CHK_ERR(kernel_2);
	}
/*^^^ hidden to output*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	CUBLAS_ERR(cublasDgemv(cublas_handle,CUBLAS_OP_T,M,N,&_alpha,_K.output.cuda_w,M,hidden_vector_ptr[_K.n_hiddens-1],1,&_beta,_K.cuda_out,1));
	CHK_ERR(cublas_3);
	sigmoid<<<_KG(N)>>>(N,_K.cuda_out);
	CHK_ERR(kernel_3);
	/*all done, calculate a preliminary error*/
	N=_K.n_outputs;
	amb<<<_KG(N)>>>(N,tmp_gpu,train,_K.cuda_out);
	CHK_ERR(kernel_4);
	cublasDasum(cublas_handle,N,tmp_gpu,1,&Ep);
	CHK_ERR(cublas_4);
	//cudaDeviceSynchronize();
	Ep*=0.5;
//	printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
/*^^^ output*/
	N=_K.n_outputs;
	_dsigmoid<<<_KG(N)>>>(N,_K.cuda_out,delta_ptr[_K.n_hiddens]);
	CHK_ERR(kernel_5);
	mul_diff<<<_KG(N)>>>(N,train,_K.cuda_out,delta_ptr[_K.n_hiddens]);
	CHK_ERR(kernel_6);
/*^^^ output to hidden*/
	/*! transposed (of the transposed might be transposed)*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	cublasDgemv(cublas_handle,CUBLAS_OP_N,M,N,&_alpha,_K.output.cuda_w,M,delta_ptr[_K.n_hiddens],1,&_beta,delta_ptr[_K.n_hiddens-1],1);
	CHK_ERR(cublas_5);
	dsigmoid<<<_KG(M)>>>(M,hidden_vector_ptr[_K.n_hiddens-1],delta_ptr[_K.n_hiddens-1]);
	CHK_ERR(kernel_7);
/*^^^ hidden to hidden (if any)*/
	if(_K.n_hiddens>1){
		for(idx=(_K.n_hiddens-2);idx>0;idx--){
			/*! transposed (of the transposed might be transposed)*/
			N=_K.hiddens[idx+1].n_neurons;
			M=_K.hiddens[idx+1].n_inputs;
			cublasDgemv(cublas_handle,CUBLAS_OP_N,M,N,&_alpha,_K.hiddens[idx+1].cuda_w,M,delta_ptr[idx+1],1,&_beta,delta_ptr[idx],1);
			CHK_ERR(cublas_6);
			dsigmoid<<<_KG(M)>>>(M,hidden_vector_ptr[idx],delta_ptr[idx]);
			CHK_ERR(kernel_8);
		}
		/*add zero*/
		/*! transposed (of the transposed might be transposed)*/
		N=_K.hiddens[1].n_neurons;
		M=_K.hiddens[1].n_inputs;
		cublasDgemv(cublas_handle,CUBLAS_OP_N,M,N,&_alpha,_K.hiddens[1].cuda_w,M,delta_ptr[1],1,&_beta,delta_ptr[0],1);
		CHK_ERR(cublas_7);
		dsigmoid<<<_KG(M)>>>(M,hidden_vector_ptr[0],delta_ptr[0]);
		CHK_ERR(kernel_9);
	}
/*+++ III - back propagation +++*/
/*^^^ output*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	_alpha=LEARN_RATE;
	cublasDger(cublas_handle,M,N,&_alpha,hidden_vector_ptr[_K.n_hiddens-1],1,delta_ptr[_K.n_hiddens],1,_K.output.cuda_w,M);
	CHK_ERR(cublas_8);
/*^^^ hiddens*/
	for(idx=(_K.n_hiddens-1);idx>0;idx--){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		cublasDger(cublas_handle,M,N,&_alpha,hidden_vector_ptr[idx-1],1,delta_ptr[idx],1,_K.hiddens[idx].cuda_w,M);
		CHK_ERR(cublas_9);
	}
	/*add zero*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	cublasDger(cublas_handle,M,N,&_alpha,_K.cuda_in,1,delta_ptr[0],1,_K.hiddens[0].cuda_w,M);
	CHK_ERR(cublas_10);
/*+++ IV - update error +++*/
	N=_K.n_outputs;
	/*>>> update cuda_out <<<*/
if(cudas->cuda_n_streams>1) scuda_ann_forward_cublas(kernel,cudas);
else cuda_ann_forward_cublas(kernel,cublas_handle);
	amb<<<_KG(N)>>>(N,tmp_gpu,train,_K.cuda_out);
	CHK_ERR(kernel_10);
	cublasDasum(cublas_handle,N,tmp_gpu,1,&Epr);
	CHK_ERR(cublas_11);
	Epr*=0.5;
//	cudaDeviceSynchronize();
//	fprintf(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ V - cleanup +++*/
	for(idx=0;idx<_K.n_hiddens;idx++){
		CUDA_FREE(hidden_vector_ptr[idx]);
		hidden_vector_ptr[idx]=NULL;
	}
	FREE(hidden_vector_ptr);
	for(idx=0;idx<(_K.n_hiddens+1);idx++){
		CUDA_FREE(delta_ptr[idx]);
		delta_ptr[idx]=NULL;
	}
	FREE(delta_ptr);
	CUDA_FREE(tmp_gpu);
	CHK_ERR(free_1);
	return Ep-Epr;
}

double scuda_ann_train_cublas(_kernel *kernel,double *train,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem;
	double **hidden_vector_ptr;
	double **delta_ptr;
	double Ep =0.;
	double Epr=0.;
	/**/
	double _alpha=1.0;
	double _beta =0.0;
	/*allocate*/
	ALLOC(hidden_vector_ptr,_K.n_hiddens,DOUBLE *);/*HOST*/
	for(idx=0;idx<_K.n_hiddens;idx++) CUDA_ALLOC(hidden_vector_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
	ALLOC(delta_ptr,_K.n_hiddens+1,DOUBLE *);/*HOST*/
	for(idx=0;idx<_K.n_hiddens;idx++) CUDA_ALLOC(delta_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
	CUDA_ALLOC(delta_ptr[_K.n_hiddens],_K.n_outputs,DOUBLE);/*DEVICE*/
/*+++ I - FORWARD +++*/
/*^^^ input to hidden +++*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle,CUBLAS_OP_T,M,red,
		&_alpha,_K.hiddens[0].cuda_w+jdx*M*red,M,_K.cuda_in,1,
		&_beta,hidden_vector_ptr[0]+jdx*red,1);
		CHK_ERR(cublas_1);
		sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,hidden_vector_ptr[0]+jdx*red);
		CHK_ERR(kernel_1);
	}
	jdx=cudas->cuda_n_streams-1;/*useful?*/
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle,CUBLAS_OP_T,M,red+rem,
	&_alpha,_K.hiddens[0].cuda_w+jdx*M*red,M,_K.cuda_in,1,
	&_beta,hidden_vector_ptr[0]+jdx*red,1);
	CHK_ERR(cublas_1);
	sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,hidden_vector_ptr[0]+jdx*red);
	CHK_ERR(kernel_1);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(M,red,
		_K.hiddens[0].cuda_w+jdx*M*red,_K.cuda_in,hidden_vector_ptr[0]+jdx*red);
		CHK_ERR(kernel_1);
	}
	jdx=cudas->cuda_n_streams-1;/*useful?*/
	fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(M,red+rem,
	_K.hiddens[0].cuda_w+jdx*M*red,_K.cuda_in,hidden_vector_ptr[0]+jdx*red);
	CHK_ERR(kernel_1);
#endif /*_CUBLAS*/
	/*now wait for everyone*/
	cudaDeviceSynchronize();/*TODO: check if necessary*/
/*^^^ hidden to hidden (if any)*/
	for(idx=1;idx<_K.n_hiddens;idx++){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		red=N/cudas->cuda_n_streams;
		rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle,CUBLAS_OP_T,M,red,
			&_alpha,_K.hiddens[idx].cuda_w+jdx*M*red,M,hidden_vector_ptr[idx-1],1,
			&_beta,hidden_vector_ptr[idx]+jdx*red,1);
			CHK_ERR(cublas_2);
			sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,hidden_vector_ptr[idx]+jdx*red);
			CHK_ERR(kernel_2);
		}
		jdx=cudas->cuda_n_streams-1;/*useful?*/
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle,CUBLAS_OP_T,M,red+rem,
		&_alpha,_K.hiddens[idx].cuda_w+jdx*M*red,M,hidden_vector_ptr[idx-1],1,
		&_beta,hidden_vector_ptr[idx]+jdx*red,1);
		CHK_ERR(cublas_2);
		sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,hidden_vector_ptr[idx]+jdx*red);
		CHK_ERR(kernel_2);
#else  /*_CUBLAS*/
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(M,red,
			_K.hiddens[idx].cuda_w+jdx*M*red,hidden_vector_ptr[idx-1],hidden_vector_ptr[idx]+jdx*red);
			CHK_ERR(kernel_2);
		}
		jdx=cudas->cuda_n_streams-1;/*useful?*/
		fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(M,red+rem,
		_K.hiddens[idx].cuda_w+jdx*M*red,hidden_vector_ptr[idx-1],hidden_vector_ptr[idx]+jdx*red);
		CHK_ERR(kernel_2);
#endif /*_CUBLAS*/
		/*now wait for everyone*/
		cudaDeviceSynchronize();/*TODO: check if necessary*/
	}
/*^^^ hidden to output*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle,CUBLAS_OP_T,M,red,
		&_alpha,_K.output.cuda_w+jdx*M*red,M,hidden_vector_ptr[_K.n_hiddens-1],1,
		&_beta,_K.cuda_out+jdx*red,1);
		CHK_ERR(cublas_3);
		sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,_K.cuda_out+jdx*red);
		CHK_ERR(kernel_3);
	}
	jdx=cudas->cuda_n_streams-1;/*useful?*/
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle,CUBLAS_OP_T,M,red+rem,
	&_alpha,_K.output.cuda_w+jdx*M*red,M,hidden_vector_ptr[_K.n_hiddens-1],1,
	&_beta,_K.cuda_out+jdx*red,1);
	CHK_ERR(cublas_3);
	sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,_K.cuda_out+jdx*red);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(M,red,
		_K.output.cuda_w+jdx*M*red,hidden_vector_ptr[_K.n_hiddens-1],_K.cuda_out+jdx*red);
		CHK_ERR(kernel_3);
	}
	jdx=cudas->cuda_n_streams-1;/*useful?*/
	fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(M,red+rem,
	_K.output.cuda_w+jdx*M*red,hidden_vector_ptr[_K.n_hiddens-1],_K.cuda_out+jdx*red);
	CHK_ERR(kernel_3);
#endif /*_CUBLAS*/
	/*all done, calculate a preliminary error*/
	N=_K.n_outputs;
	/*TODO: no streams for that part?*/
#ifdef   _CUBLAS
	amb<<<_KG(N)>>>(N,_K.tmp_gpu,train,_K.cuda_out);
	CHK_ERR(kernel_4);
	cublasDasum(cudas->cuda_handle,N,_K.tmp_gpu,1,&Ep);
	CHK_ERR(cublas_4);
#else  /*_CUBLAS*/
	amb_acc<<<1,_TPB/2,sizeof(double)*(_TPB)>>>(N,_K.tmp_gpu,train,_K.cuda_out);
	CHK_ERR(kernel_4b);
	CUDA_G2C_CP(&Ep,_K.tmp_gpu,1,double);
#endif /*_CUBLAS*/
	Ep*=0.5;
//	printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
/*^^^ output*/
	/*TODO: no streams for that part?*/
	N=_K.n_outputs;
	dsigmoid_mul_diff<<<_KG(N)>>>(N,train,_K.cuda_out,delta_ptr[_K.n_hiddens]);
	CHK_ERR(kernel_5);
/*^^^ output to hidden*/
	/*distribution over M due to transposed operations*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=M/cudas->cuda_n_streams;
	rem=M%cudas->cuda_n_streams;
#ifdef   _CUBLAS
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
//FIXME: WHAT?
		cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red,N,
		&_alpha,_K.output.cuda_w+jdx*N*red,red /*or M?*/,delta_ptr[_K.n_hiddens],1,
		&_beta,delta_ptr[_K.n_hiddens-1]+jdx*red,1);
		CHK_ERR(cublas_5);
		dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,hidden_vector_ptr[_K.n_hiddens-1],delta_ptr[_K.n_hiddens-1]+jdx*red);
		CHK_ERR(kernel_6);
	}
	jdx=cudas->cuda_n_streams-1;/*useful?*/
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
//FIXME: WHAT?
	cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red+rem,N,
	&_alpha,_K.output.cuda_w+jdx*N*red,red /*or M?*/,delta_ptr[_K.n_hiddens],1,
	&_beta,delta_ptr[_K.n_hiddens-1]+jdx*red,1);
	CHK_ERR(cublas_5);
	dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,hidden_vector_ptr[_K.n_hiddens-1],delta_ptr[_K.n_hiddens-1]+jdx*red);
	CHK_ERR(kernel_6);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		dsigmoid_mul_delta<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,N,_K.output.cuda_w+jdx*N*red,
		delta_ptr[_K.n_hiddens],hidden_vector_ptr[_K.n_hiddens-1],delta_ptr[_K.n_hiddens-1]+jdx*red);
		CHK_ERR(kernel_6b);
	}
	jdx=cudas->cuda_n_streams-1;/*useful?*/
	dsigmoid_mul_delta<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,N,_K.output.cuda_w+jdx*N*red,
	delta_ptr[_K.n_hiddens],hidden_vector_ptr[_K.n_hiddens-1],delta_ptr[_K.n_hiddens-1]+jdx*red);
	CHK_ERR(kernel_6b);
#endif /*_CUBLAS*/
	/*TODO: should we sync?*/
/*^^^ hidden to hidden (if any)*/
	if(_K.n_hiddens>1){
		for(idx=(_K.n_hiddens-2);idx>0;idx--){
			N=_K.hiddens[idx+1].n_neurons;
			M=_K.hiddens[idx+1].n_inputs;
			red=M/cudas->cuda_n_streams;
			rem=M%cudas->cuda_n_streams;
#ifdef   _CUBLAS
			for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
				cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
//FIXME: WHAT?
				cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red,N,
				&_alpha,_K.hiddens[idx+1].cuda_w+jdx*N*red,red /*or M?*/,delta_ptr[idx+1],1,
				&_beta,delta_ptr[idx]+jdx*red,1);
				CHK_ERR(cublas_6);
				dsigmoid<<<_KG(red)>>>(red,hidden_vector_ptr[idx],delta_ptr[idx]+jdx*red);
				CHK_ERR(kernel_7);
			}
			jdx=cudas->cuda_n_streams-1;/*useful?*/
			cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
//FIXME: WHAT?
			cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red+rem,N,
			&_alpha,_K.hiddens[idx+1].cuda_w+jdx*N*red,red /*or M?*/,delta_ptr[idx+1],1,
			&_beta,delta_ptr[idx]+jdx*red,1);
			CHK_ERR(cublas_6);
			dsigmoid<<<_KG(red+rem)>>>(red+rem,hidden_vector_ptr[idx],delta_ptr[idx]+jdx*red);
			CHK_ERR(kernel_7);
#else  /*_CUBLAS*/
			for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
				dsigmoid_mul_delta<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,N,_K.hiddens[idx+1].cuda_w+jdx*N*red,
				delta_ptr[idx+1],hidden_vector_ptr[idx],delta_ptr[idx]+jdx*red);
				CHK_ERR(kernel_7b);
			}
			jdx=cudas->cuda_n_streams-1;/*useful?*/
			dsigmoid_mul_delta<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,N,_K.hiddens[idx+1].cuda_w+jdx*N*red,
			delta_ptr[idx+1],hidden_vector_ptr[idx],delta_ptr[idx]+jdx*red);
			CHK_ERR(kernel_7b);
#endif /*_CUBLAS*/
		}
		/*add zero*/
		N=_K.hiddens[1].n_neurons;
		M=_K.hiddens[1].n_inputs;
		red=M/cudas->cuda_n_streams;
		rem=M%cudas->cuda_n_streams;
#ifdef   _CUBLAS
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
//FIXME: WHAT?
			cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red,N,
			&_alpha,_K.hiddens[1].cuda_w+jdx*N*red,red /*or M?*/,delta_ptr[1],1,
			&_beta,delta_ptr[0]+jdx*red,1);
			CHK_ERR(cublas_7);
			dsigmoid<<<_KG(red)>>>(red,hidden_vector_ptr[0],delta_ptr[0]+jdx*red);
			CHK_ERR(kernel_8);
		}
		jdx=cudas->cuda_n_streams-1;/*useful?*/
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
//FIXME: WHAT?
		cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red+rem,N,
		&_alpha,_K.hiddens[1].cuda_w+jdx*N*red,red /*or M?*/,delta_ptr[1],1,
		&_beta,delta_ptr[0]+jdx*red,1);
		CHK_ERR(cublas_7);
		dsigmoid<<<_KG(red+rem)>>>(red+rem,hidden_vector_ptr[0],delta_ptr[0]+jdx*red);
		CHK_ERR(kernel_8);
#else  /*_CUBLAS*/
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			dsigmoid_mul_delta<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,N,_K.hiddens[1].cuda_w+jdx*N*red,
			delta_ptr[1],hidden_vector_ptr[0],delta_ptr[0]+jdx*red);
			CHK_ERR(kernel_8b);
		}
		jdx=cudas->cuda_n_streams-1;/*useful?*/
		dsigmoid_mul_delta<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,N,_K.hiddens[1].cuda_w+jdx*N*red,
		delta_ptr[1],hidden_vector_ptr[0],delta_ptr[0]+jdx*red);
		CHK_ERR(kernel_8b);
#endif /*_CUBLAS*/
	}
/*+++ III - back propagation +++*/
/*^^^ output*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
	_alpha=LEARN_RATE;
#ifdef   _CUBLAS
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle,M,red,&_alpha,hidden_vector_ptr[_K.n_hiddens-1],1,
		delta_ptr[_K.n_hiddens]+jdx*red,1,_K.output.cuda_w+jdx*M*red,M);
		CHK_ERR(cublas_8);
	}
	jdx=cudas->cuda_n_streams-1;/*useful?*/
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle,M,red+rem,&_alpha,hidden_vector_ptr[_K.n_hiddens-1],1,
	delta_ptr[_K.n_hiddens]+jdx*red,1,_K.output.cuda_w+jdx*M*red,M);
	CHK_ERR(cublas_8);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
//TODO: CHECK
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,hidden_vector_ptr[_K.n_hiddens-1],delta_ptr[_K.n_hiddens]+jdx*red,_K.output.cuda_w+jdx*M*red);
		CHK_ERR(kernel_9);
	}
	jdx=cudas->cuda_n_streams-1;/*useful?*/
	ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,hidden_vector_ptr[_K.n_hiddens-1],delta_ptr[_K.n_hiddens]+jdx*red,_K.output.cuda_w+jdx*M*red);
	CHK_ERR(kernel_9);
#endif /*_CUBLAS*/
/*^^^ hiddens*/
	for(idx=(_K.n_hiddens-1);idx>0;idx--){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		red=N/cudas->cuda_n_streams;
		rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle,M,red,&_alpha,hidden_vector_ptr[idx-1],1,
			delta_ptr[idx]+jdx*red,1,_K.hiddens[idx].cuda_w+jdx*M*red,M);
			CHK_ERR(cublas_9);
		}
		jdx=cudas->cuda_n_streams-1;/*useful?*/
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle,M,red+rem,&_alpha,hidden_vector_ptr[idx-1],1,
		delta_ptr[idx]+jdx*red,1,_K.hiddens[idx].cuda_w+jdx*M*red,M);
#else  /*_CUBLAS*/
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
//TODO: CHECK
			ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,hidden_vector_ptr[idx-1],delta_ptr[idx]+jdx*red,_K.hiddens[idx].cuda_w+jdx*M*red);
			CHK_ERR(kernel_A);
		}
		jdx=cudas->cuda_n_streams-1;/*useful?*/
		ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,hidden_vector_ptr[idx-1],delta_ptr[idx]+jdx*red,_K.hiddens[idx].cuda_w+jdx*M*red);
		CHK_ERR(kernel_A);
#endif /*_CUBLAS*/
	}
	/*add zero*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle,M,red,&_alpha,_K.cuda_in,1,delta_ptr[0]+jdx*red,1,_K.hiddens[0].cuda_w+jdx*M*red,M);
		CHK_ERR(cublas_9);
	}
	jdx=cudas->cuda_n_streams-1;/*useful?*/
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle,M,red+rem,&_alpha,_K.cuda_in,1,delta_ptr[0]+jdx*red,1,_K.hiddens[0].cuda_w+jdx*M*red,M);
	CHK_ERR(cublas_9);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
//TODO: CHECK
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.cuda_in,delta_ptr[0]+jdx*red,_K.hiddens[0].cuda_w+jdx*M*red);
		CHK_ERR(kernel_A);
	}
	jdx=cudas->cuda_n_streams-1;/*useful?*/
	ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_K.cuda_in,delta_ptr[0]+jdx*red,_K.hiddens[0].cuda_w+jdx*M*red);
	CHK_ERR(kernel_A);
#endif /*_CUBLAS*/
/*+++ IV - update error +++*/
	N=_K.n_outputs;
	/*update cuda_out*/
	scuda_ann_forward_cublas(kernel,cudas);
	/*TODO: no streams for that part?*/
#ifdef   _CUBLAS
	amb<<<_KG(N)>>>(N,_K.tmp_gpu,train,_K.cuda_out);
	CHK_ERR(kernel_4);
	cublasDasum(cudas->cuda_handle,N,_K.tmp_gpu,1,&Epr);
	CHK_ERR(cublas_4);
#else  /*_CUBLAS*/
	amb_acc<<<1,_TPB/2,sizeof(double)*(_TPB)>>>(N,_K.tmp_gpu,train,_K.cuda_out);
	CHK_ERR(kernel_4b);
	CUDA_G2C_CP(&Epr,_K.tmp_gpu,1,double);
#endif /*_CUBLAS*/
	Epr*=0.5;
//	fprintf(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ V - cleanup +++*/
	return Ep-Epr;
}




void cuda_ann_act(double *out,int size){
	sigmoid<<<(size+255)/256, 256>>>(size, out);
	CHK_ERR(sigmoid);
}
void cuda_ann_dact(double *in,double *out,int size){
	dsigmoid<<<(size+255)/256, 256>>>(size, in, out);
	CHK_ERR(dsigmoid);
}
void cuda_ann_amb(double *out, double *a,double *b,int size){
	amb<<<(size+255)/256, 256>>>(size, out, a, b);
	CHK_ERR(amb);
}
void cuda_ann_mul_diff(double *train, double *out, double *res, int size){
	mul_diff<<<(size+255)/256, 256>>>(size,train,out,res);
	CHK_ERR(mul_diff);
}
void cuda_zero_mv(int m,int n,double *mat,double *vec, double *res){
	zero_mv<<<m/256+1, 256>>>(m,n,mat,vec,res);
	CHK_ERR(zero_mv);
}

void cuda_zero_tmv(int m,int n,double *mat,double *vec, double *res){
        zero_tmv<<<n/256+1, 256>>>(m,n,mat,vec,res);
	CHK_ERR(zero_tmv);
}






}/*extern "C"*/
