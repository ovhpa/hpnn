/*
+++ libhpnn - High Performance Neural Network library - file: cuda_ann.cu +++
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
#include <cuda_runtime.h>
#ifdef _CUBLAS
#include <cublas_v2.h>
#endif
#include <libhpnn/common.h>
#include <libhpnn/ann.h>
/*CUDA specific*/
#include <libhpnn/cuda_ann.h>

/*^^^ useful to launch kernels*/
#define _WP  32
#define _TPW 32
#define _TPB (_TPW*_WP)
#define _KG(n) ((n+_TPB-1)/(_TPB)),_TPB
/*---------------*/
/*+++ KERNELS +++*/
/*---------------*/
__global__
void sigmoid(int n, double *x){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] = 2.0/(1.0+exp(-1.0*x[i]))-1.0;
}
__global__
void _dsigmoid(int n, double *in, double *out){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i += stride)
                out[i] = (-0.5 * ( in[i] * in[i] - 1.0));
}
__global__
void dsigmoid(int n, double *in, double *out){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		out[i] *= (-0.5 * ( in[i] * in[i] - 1.0));
}
__global__
void amb(int n, double *out, double *a, double *b){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i += stride)
                out[i] = ( a[i] - b[i] ) * ( a[i] - b[i] );
}
__global__
void mul_diff(int n, double *t, double *o, double *y){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i += stride)
		y[i] *= ( t[i] - o[i] );
	
}
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
	double mySum = (i < n) ? (a[i]-b[i])*(a[i]-b[i]) : 0;
	if(i+blockDim.x < n) mySum += 
		(a[i+blockDim.x]-b[i+blockDim.x])*(a[i+blockDim.x]-b[i+blockDim.x]);
	sh_data[tid]=mySum;
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
void dsigmoid_mul_delta_T(int red,int m,int n,
						  double *w,double *d,double *h,double *res){
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        double sum=0.;
        if(tid<red){
                for(int i=0; i<n; i++) sum += d[i] * w[(i*m)+tid];
                res[tid]=sum * (-0.5 * ( h[tid] * h[tid] -1.0));
        }
}
__global__
void ger_acc(int m,int n,double alpha,double *d,double *h,double *w){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	double tmp;
	if(tid<n){
		tmp=alpha*d[tid];
		/*a full line*/
		for(int i=0; i<m; i++) w[(tid*m)+i] += tmp*h[i];
	}
}
__global__
void ger_dw_acc(int m,int n,double learn,double moment,
				double *d,double *v,double *dw,double *w){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	double tmp;
	if(tid<n){
		tmp=learn*d[tid];
		/*a full line*/
		for(int i=0; i<m; i++) {
			dw[(tid*m)+i] += tmp*v[i];
			w[(tid*m)+i]  += dw[(tid*m)+i];
			dw[(tid*m)+i] *= moment;
		}
	}
}
/*-----------------*/
/* The C interface */
/*-----------------*/
extern "C"{
#define _K (*kernel)
/*---------------------------------------*/
/*+++ de-allocate CUDA-part of kernel +++*/
/*---------------------------------------*/
void scuda_ann_deallocate(_kernel *kernel){
	int idx;
	cudaSetDevice(0);/*make sure all de-allocation happen on gpu[0]*/
	CUDA_FREE(_K.cuda_in);
	for(idx=0;idx<_K.n_hiddens;idx++){
		CUDA_FREE(_K.hiddens[idx].cuda_w);
		CUDA_FREE(_K.hiddens[idx].cuda_v);
	}
	CUDA_FREE(_K.output.cuda_w);
	CUDA_FREE(_K.output.cuda_v);
	CUDA_FREE(_K.tmp_gpu);
}
/*------------------------------------*/
/*+++ allocate CUDA-part of kernel +++*/
/*------------------------------------*/
void scuda_ann_allocate(_kernel *kernel,cudastreams *cudas){
	int allocate;
	int idx;
	/*allocate everything in CUDA*/
	switch(cudas->mem_model){
	case CUDA_MEM_P2P:
	case CUDA_MEM_EXP:
	case CUDA_MEM_NONE:
		/*in all cases, we need to initialize memory on GPU[0]*/
		cudaSetDevice(0);
		allocate=0;
		CUDA_ALLOC_REPORT(_K.cuda_in,_K.n_inputs,DOUBLE,allocate);
		for(idx=0;idx<_K.n_hiddens;idx++){
			CUDA_ALLOC_REPORT(_K.hiddens[idx].cuda_w,
				_K.hiddens[idx].n_inputs*_K.hiddens[idx].n_neurons,
				DOUBLE,allocate);
			CUDA_ALLOC_REPORT(_K.hiddens[idx].cuda_v,
				_K.hiddens[idx].n_neurons,DOUBLE,allocate);
		}
		CUDA_ALLOC_REPORT(_K.output.cuda_w,
				_K.output.n_inputs*_K.output.n_neurons,DOUBLE,allocate);
		CUDA_ALLOC_REPORT(_K.output.cuda_v,_K.output.n_neurons,DOUBLE,allocate);
		/*allocate a temporary working array buffer with a maximum dimension*/
		_K.max_index=_K.n_inputs;
		if(_K.n_outputs>_K.max_index) _K.max_index=_K.n_outputs;
		for(idx=0;idx<_K.n_hiddens;idx++)
			if(_K.hiddens[idx].n_neurons>_K.max_index)
				_K.max_index=_K.hiddens[idx].n_neurons;
		CUDA_ALLOC_REPORT(_K.tmp_gpu,_K.max_index,DOUBLE,allocate);
		break;
	case CUDA_MEM_CMM:
		cudaSetDevice(0);/*make sure all allocation happen on gpu[0]*/
		allocate=0;
		CUDA_ALLOC_MM_REPORT(_K.cuda_in,_K.n_inputs,DOUBLE,allocate);
		for(idx=0;idx<_K.n_hiddens;idx++){
			CUDA_ALLOC_MM_REPORT(_K.hiddens[idx].cuda_w,
				_K.hiddens[idx].n_inputs*_K.hiddens[idx].n_neurons,
				DOUBLE,allocate);
			CUDA_ALLOC_MM_REPORT(_K.hiddens[idx].cuda_v,
				_K.hiddens[idx].n_neurons,DOUBLE,allocate);
		}
		CUDA_ALLOC_MM_REPORT(_K.output.cuda_w,
				_K.output.n_inputs*_K.output.n_neurons,DOUBLE,allocate);
		CUDA_ALLOC_MM_REPORT(_K.output.cuda_v,_K.output.n_neurons,DOUBLE,allocate);
		/*allocate a temporary working array buffer with a maximum dimension*/
		_K.max_index=_K.n_inputs;
		if(_K.n_outputs>_K.max_index) _K.max_index=_K.n_outputs;
		for(idx=0;idx<_K.n_hiddens;idx++)
			if(_K.hiddens[idx].n_neurons>_K.max_index)
				_K.max_index=_K.hiddens[idx].n_neurons;
		CUDA_ALLOC_MM_REPORT(_K.tmp_gpu,_K.max_index,DOUBLE,allocate);
		break;
	default:
		return;
	}
	_OUT(stdout,"ANN total CUDA allocation: %lu (bytes)\n",allocate);
}
/*--------------------------*/
/*+++ free CUDA-momentum +++*/
/*--------------------------*/
void scuda_ann_free_momentum(_kernel *kernel){
	int idx;
	cudaSetDevice(0);/*make sure all de-allocation happen on gpu[0]*/
	if(_K.cuda_dw==NULL) return;
	for(idx=0;idx<_K.n_hiddens;idx++)
		CUDA_FREE(_K.cuda_dw[idx]);
	CUDA_FREE(_K.cuda_dw[_K.n_hiddens]);
	FREE(_K.cuda_dw);
}
/*------------------------------*/
/*+++ allocate CUDA-momentum +++*/
/*------------------------------*/
void scuda_ann_allocate_momentum(_kernel *kernel,cudastreams *cudas){
	int allocate;
	int idx;
	
	switch(cudas->mem_model){
	case CUDA_MEM_P2P:
	case CUDA_MEM_EXP:
	case CUDA_MEM_NONE:
		cudaSetDevice(0);/*make sure all allocation happen on gpu[0]*/
		allocate=0;
		ALLOC_REPORT(_K.cuda_dw,_K.n_hiddens+1,DOUBLE *,allocate);/*HOST*/
		_OUT(stdout,"[CPU] CUDA MOMENTUM ALLOC: %lu (bytes)\n",allocate);
		allocate=0;
		CUDA_ALLOC_REPORT(_K.cuda_dw[_K.n_hiddens],
			_K.output.n_inputs*_K.output.n_neurons,DOUBLE,allocate);
		for(idx=0;idx<_K.n_hiddens;idx++)
			CUDA_ALLOC_REPORT(_K.cuda_dw[idx],
				_K.hiddens[idx].n_inputs*_K.hiddens[idx].n_neurons,
				DOUBLE,allocate);
		_OUT(stdout,"[GPU] CUDA MOMENTUM ALLOC: %lu (bytes)\n",allocate);
		break;
	case CUDA_MEM_CMM:
		cudaSetDevice(0);/*make sure all allocation happen on gpu[0]*/
		allocate=0;
		ALLOC_REPORT(_K.cuda_dw,_K.n_hiddens+1,DOUBLE *,allocate);/*HOST*/
		_OUT(stdout,"[CPU] CUDA MOMENTUM ALLOC: %lu (bytes)\n",allocate);
		allocate=0;
		CUDA_ALLOC_MM_REPORT(_K.cuda_dw[_K.n_hiddens],
			_K.output.n_inputs*_K.output.n_neurons,DOUBLE,allocate);
		for(idx=0;idx<_K.n_hiddens;idx++)
			CUDA_ALLOC_MM_REPORT(_K.cuda_dw[idx],
				_K.hiddens[idx].n_inputs*_K.hiddens[idx].n_neurons,
				DOUBLE,allocate);
		_OUT(stdout,"[GPU] CUDA MOMENTUM ALLOC: %lu (bytes)\n",allocate);
		break;
	default:
		return;
	}
}
/*----------------------------------------*/
/*+++ transfer weights from CPU to GPU +++*/
/*----------------------------------------*/
void scuda_ann_weights_C2G(_kernel *kernel,cudastreams *cudas){
	int idx;
	int M,N;
	switch(cudas->mem_model){
	case CUDA_MEM_P2P:
	case CUDA_MEM_EXP:
	case CUDA_MEM_NONE:
		cudaSetDevice(0);/*make sure all transfer happen to gpu[0]*/
/*^^^ output*/
		N=_K.output.n_neurons;
		M=_K.output.n_inputs;
		CUDA_C2G_CP(_K.output.weights,_K.output.cuda_w,M*N,double);
		CHK_ERR(memcpy_C2G);
/*^^^ hiddens*/
		for(idx=0;idx<_K.n_hiddens;idx++){
			N=_K.hiddens[idx].n_neurons;
			M=_K.hiddens[idx].n_inputs;
			CUDA_C2G_CP(_K.hiddens[idx].weights,_K.hiddens[idx].cuda_w,M*N,double);
			CHK_ERR(memcpy_C2G);
		}
		cudaDeviceSynchronize();/*only GPU[0]?*/
		break;
	case CUDA_MEM_CMM:
		/*cuda CMM can be access directly on GPU*/
		break;
	default:
		return;
	}
}
/*----------------------------------------*/
/*+++ transfer weights from GPU to CPU +++*/
/*----------------------------------------*/
void scuda_ann_weights_G2C(_kernel *kernel,cudastreams *cudas){
	int idx;
	int M,N;
	switch(cudas->mem_model){
	case CUDA_MEM_P2P:
	case CUDA_MEM_EXP:
	case CUDA_MEM_NONE:
		cudaSetDevice(0);/*make sure all transfer happen from gpu[0]*/
		cudaDeviceSynchronize();/*TODO: is it needed? only GPU[0]?*/
/*^^^ output*/
		N=_K.output.n_neurons;
		M=_K.output.n_inputs;
		CUDA_G2C_CP(_K.output.weights,_K.output.cuda_w,M*N,double);
		CHK_ERR(memcpy_C2G);
/*^^^ hiddens*/
		for(idx=0;idx<_K.n_hiddens;idx++){
			N=_K.hiddens[idx].n_neurons;
			M=_K.hiddens[idx].n_inputs;
			CUDA_G2C_CP(_K.hiddens[idx].weights,_K.hiddens[idx].cuda_w,M*N,double);
		}
	case CUDA_MEM_CMM:
		/*cuda CMM can be access directly on CPU*/
		break;
	default:
		return;
	}
}
/*-----------------------------*/
/*+++ forward kernel update +++*/
/*-----------------------------*/
void scuda_ann_forward(_kernel *kernel,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem,gpu;
	int total_s;
#ifdef _CUBLAS
	double _alpha=1.0;
	double _beta =0.0;
#endif
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
	if(cudas->mem_model==CUDA_MEM_CMM){
		/*prefetch all kernel?*/
	}
/*+++ I - input +++*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef   _CUBLAS
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);/*select FIXME: do we need to select all the time?*/
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],
			CUBLAS_OP_T,M,red,&_alpha,_K.hiddens[0].cuda_w+jdx*M*red,M,
			_K.cuda_in,1,&_beta,_K.hiddens[0].cuda_v+jdx*red,1);
		CHK_ERR(fw_gemv);
		sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.hiddens[0].cuda_v+jdx*red);
		CHK_ERR(fw_sigmoid);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);/*select FIXME: do we need to select all the time?*/
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle[gpu],
		CUBLAS_OP_T,M,red+rem,&_alpha,_K.hiddens[0].cuda_w+jdx*M*red,M,
		_K.cuda_in,1,&_beta,_K.hiddens[0].cuda_v+jdx*red,1);
	CHK_ERR(fw_gemv);
	sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_K.hiddens[0].cuda_v+jdx*red);
	CHK_ERR(fw_sigmoid);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);/*select FIXME: do we need to select all the time?*/
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.hiddens[0].cuda_w+jdx*M*red,_K.cuda_in,
				_K.hiddens[0].cuda_v+jdx*red);
		CHK_ERR(fw_mv_acc);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);/*select FIXME: do we need to select all the time?*/
	fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_K.hiddens[0].cuda_w+jdx*M*red,_K.cuda_in,
			_K.hiddens[0].cuda_v+jdx*red);
	CHK_ERR(fw_mv_acc);
#endif /*_CUBLAS*/
	/*sync all streams/threads on all GPUs*/
	if(cudas->mem_model==CUDA_MEM_EXP){
		/*sync _K.hiddens[0].cuda_v between all GPUs*/
	}
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*+++ II - hidden(s) +++*/
	for(idx=1;idx<_K.n_hiddens;idx++){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		red=N/total_s;
		rem=N%total_s;
#ifdef   _CUBLAS
		for(jdx=0;jdx<total_s-1;jdx++){
			gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
			cudaSetDevice(gpu);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],
				CUBLAS_OP_T,M,red,&_alpha,_K.hiddens[idx].cuda_w+jdx*M*red,M,
				_K.hiddens[idx-1].cuda_v,1,&_beta,
				_K.hiddens[idx].cuda_v+jdx*red,1);
			CHK_ERR(fw_gemv);
			sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.hiddens[idx].cuda_v+jdx*red);
			CHK_ERR(fw_sigmoid);
		}
		gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],
			CUBLAS_OP_T,M,red+rem,&_alpha,_K.hiddens[idx].cuda_w+jdx*M*red,M,
			_K.hiddens[idx-1].cuda_v,1,&_beta,
			_K.hiddens[idx].cuda_v+jdx*red,1);
		CHK_ERR(cublas_1);
		sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,_K.hiddens[idx].cuda_v+jdx*red);
		CHK_ERR(kernel_1);
#else  /*_CUBLAS*/
		for(jdx=0;jdx<total_s-1;jdx++){
			gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
			cudaSetDevice(gpu);
			fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,_K.hiddens[idx].cuda_w+jdx*M*red,
				_K.hiddens[idx-1].cuda_v,_K.hiddens[idx].cuda_v+jdx*red);
			CHK_ERR(fw_mv_acc);
		}
		gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
		cudaSetDevice(gpu);
		fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,_K.hiddens[idx].cuda_w+jdx*M*red,
			_K.hiddens[idx-1].cuda_v,_K.hiddens[idx].cuda_v+jdx*red);
		CHK_ERR(fw_mv_acc);
#endif /*_CUBLAS*/
		/*sync all streams/threads on all GPUs*/
		if(cudas->mem_model==CUDA_MEM_EXP){
			/* sync _K.hiddens[idx].cuda_v between all GPUs*/
		}
		for(gpu=0;gpu<cudas->n_gpu;gpu++){
			cudaSetDevice(gpu);
			cudaDeviceSynchronize();
		}
	}
/*+++ III - output +++*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef   _CUBLAS
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],
			CUBLAS_OP_T,M,red,&_alpha,_K.output.cuda_w+jdx*M*red,M,
			_K.hiddens[_K.n_hiddens-1].cuda_v,1,
			&_beta,_K.output.cuda_v+jdx*red,1);
		CHK_ERR(fw_gemv);
		sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.output.cuda_v+jdx*red);
		CHK_ERR(fw_sigmoid);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle[gpu],
		CUBLAS_OP_T,M,red+rem,&_alpha,_K.output.cuda_w+jdx*M*red,M,
		_K.hiddens[_K.n_hiddens-1].cuda_v,1,&_beta,_K.output.cuda_v+jdx*red,1);
	CHK_ERR(fw_gemv);
	sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_K.output.cuda_v+jdx*red);
	CHK_ERR(fw_sigmoid);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.output.cuda_w+jdx*M*red,_K.hiddens[_K.n_hiddens-1].cuda_v,
			_K.output.cuda_v+jdx*red);
		CHK_ERR(fw_mv_acc);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_K.output.cuda_w+jdx*M*red,_K.hiddens[_K.n_hiddens-1].cuda_v,
		_K.output.cuda_v+jdx*red);
	CHK_ERR(fw_mv_acc);
#endif /*_CUBLAS*/
	/*sync all streams/threads on all GPUs*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
}
/*-----------------------------------------------*/
/*+++ Calculate Training Error TODO: optimize +++*/
/*-----------------------------------------------*/
double scuda_ann_error(_kernel *kernel,double *train,cudastreams *cudas){
	double dEp=0.;
#ifdef   _CUBLAS
	/*amb can be stream o=(t-v)*(t-v) -- worth it?*/
	amb<<<_KG(_K.n_outputs)>>>(_K.n_outputs,_K.tmp_gpu,train,_K.output.cuda_v);
	CHK_ERR(err_amb);
	/*it is possible to accumulate Ep within stream -- worth it?*/
	cudaSetDevice(0);/*only on master*/
	cublasSetStream(cudas->cuda_handle[0],NULL);
	cublasDasum(cudas->cuda_handle[0],_K.n_outputs,_K.tmp_gpu,1,&dEp);
	CHK_ERR(err_asum);
#else  /*_CUBLAS*/
	/*shared memory reduction: no streams*/
	cudaSetDevice(0);/*only on master*/
	amb_acc<<<_KG(_K.n_outputs),sizeof(double)*2*(_TPB)>>>
		(_K.n_outputs,_K.tmp_gpu,train,_K.output.cuda_v);
	CHK_ERR(err_amb_acc);
	CUDA_G2C_CP(&dEp,&(_K.tmp_gpu[0]),1,double);
	CHK_ERR(err_g2c_cp);
#endif /*_CUBLAS*/
	dEp*=0.5;
	return dEp;
}
/*------------------------*/
/*+++ Calculate deltas +++*/
/*------------------------*/
void scuda_ann_delta(_kernel *kernel,double *train,double **delta_ptr,
					 cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem,gpu;
	int total_s;
#ifdef _CUBLAS
	double _alpha=1.0;
	double _beta =0.0;
#endif /*_CUBLAS*/
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
/*^^^ output*/
	N=_K.n_outputs;
	red=N/total_s;
	rem=N%total_s;
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		dsigmoid_mul_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,train+jdx*red,_K.output.cuda_v+jdx*red,
			delta_ptr[_K.n_hiddens]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_dif);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	dsigmoid_mul_diff<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,train+jdx*red,_K.output.cuda_v+jdx*red,
		delta_ptr[_K.n_hiddens]+jdx*red);
	CHK_ERR(train_dsigmoid_mul_dif);
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*^^^ output to hidden*/
	/*distribution over M due to transposed operations*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=M/total_s;
	rem=M%total_s;
#ifdef   _CUBLAS
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
		&_alpha,_K.output.cuda_w+jdx*red,M,delta_ptr[_K.n_hiddens],1,
		&_beta,delta_ptr[_K.n_hiddens-1]+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.hiddens[_K.n_hiddens-1].cuda_v+jdx*red,
			delta_ptr[_K.n_hiddens-1]+jdx*red);
		CHK_ERR(train_dsigmoid);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
	&_alpha,_K.output.cuda_w+jdx*red,M,delta_ptr[_K.n_hiddens],1,
	&_beta,delta_ptr[_K.n_hiddens-1]+jdx*red,1);
	CHK_ERR(train_gemv);
	dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_K.hiddens[_K.n_hiddens-1].cuda_v+jdx*red,
		delta_ptr[_K.n_hiddens-1]+jdx*red);
	CHK_ERR(train_dsigmoid);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
		(red,M,N,_K.output.cuda_w+jdx*red,delta_ptr[_K.n_hiddens],
		_K.hiddens[_K.n_hiddens-1].cuda_v+jdx*red,
		delta_ptr[_K.n_hiddens-1]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,M,N,_K.output.cuda_w+jdx*red,delta_ptr[_K.n_hiddens],
		_K.hiddens[_K.n_hiddens-1].cuda_v+jdx*red,
		delta_ptr[_K.n_hiddens-1]+jdx*red);
	CHK_ERR(train_dsigmoid_mul_delta_T);
#endif /*_CUBLAS*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*^^^ hidden to hidden (if any)*/
	if(_K.n_hiddens>1){
		for(idx=(_K.n_hiddens-2);idx>0;idx--){
			N=_K.hiddens[idx+1].n_neurons;
			M=_K.hiddens[idx+1].n_inputs;
			red=M/total_s;
			rem=M%total_s;
#ifdef   _CUBLAS
			for(jdx=0;jdx<total_s-1;jdx++){
				gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
				cudaSetDevice(gpu);
				cublasSetStream(cudas->cuda_handle[gpu],
								cudas->cuda_streams[jdx]);
				cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
				&_alpha,_K.hiddens[idx+1].cuda_w+jdx*red,M,delta_ptr[idx+1],1,
				&_beta,delta_ptr[idx]+jdx*red,1);
				CHK_ERR(train_gemv);
				dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,_K.hiddens[idx].cuda_v+jdx*red,delta_ptr[idx]+jdx*red);
				CHK_ERR(train_dsigmoid);
			}
			gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
			cudaSetDevice(gpu);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
			&_alpha,_K.hiddens[idx+1].cuda_w+jdx*red,M,delta_ptr[idx+1],1,
			&_beta,delta_ptr[idx]+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
				(red+rem,_K.hiddens[idx].cuda_v+jdx*red,delta_ptr[idx]+jdx*red);
			CHK_ERR(train_dsigmoid);
#else  /*_CUBLAS*/
			for(jdx=0;jdx<total_s-1;jdx++){
				gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
				cudaSetDevice(gpu);
				dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,M,N,_K.hiddens[idx+1].cuda_w+jdx*red,delta_ptr[idx+1],
					_K.hiddens[idx].cuda_v+jdx*red,delta_ptr[idx]+jdx*red);
				CHK_ERR(train_dsigmoid_mul_delta_T);
			}
			gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
			cudaSetDevice(gpu);
			dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
				(red+rem,M,N,_K.hiddens[idx+1].cuda_w+jdx*red,delta_ptr[idx+1],
				_K.hiddens[idx].cuda_v+jdx*red,delta_ptr[idx]+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
#endif /*_CUBLAS*/
			for(gpu=0;gpu<cudas->n_gpu;gpu++){
				cudaSetDevice(gpu);
				cudaDeviceSynchronize();
			}
		}
		/*add zero*/
		N=_K.hiddens[1].n_neurons;
		M=_K.hiddens[1].n_inputs;
		red=M/total_s;
		rem=M%total_s;
#ifdef   _CUBLAS
		for(jdx=0;jdx<total_s-1;jdx++){
			gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
			cudaSetDevice(gpu);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
			&_alpha,_K.hiddens[1].cuda_w+jdx*red,M,delta_ptr[1],1,
			&_beta,delta_ptr[0]+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.hiddens[0].cuda_v+jdx*red,delta_ptr[0]+jdx*red);
			CHK_ERR(train_dsigmoid);
		}
		gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
		&_alpha,_K.hiddens[1].cuda_w+jdx*red,M,delta_ptr[1],1,
		&_beta,delta_ptr[0]+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,_K.hiddens[0].cuda_v+jdx*red,delta_ptr[0]+jdx*red);
		CHK_ERR(train_dsigmoid);
#else  /*_CUBLAS*/
		for(jdx=0;jdx<total_s-1;jdx++){
			gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
			cudaSetDevice(gpu);
			dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,M,N,_K.hiddens[1].cuda_w+jdx*red,
			delta_ptr[1],_K.hiddens[0].cuda_v+jdx*red,delta_ptr[0]+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
		}
		gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
		cudaSetDevice(gpu);
		dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,M,N,_K.hiddens[1].cuda_w+jdx*red,
		delta_ptr[1],_K.hiddens[0].cuda_v+jdx*red,delta_ptr[0]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
#endif /*_CUBLAS*/
		for(gpu=0;gpu<cudas->n_gpu;gpu++){
			cudaSetDevice(gpu);
			cudaDeviceSynchronize();
		}
	}
}
#define LEARN_RATE 0.01
/*------------------------*/
/*+++ back-propagation +++*/
/*------------------------*/
double scuda_ann_train(_kernel *kernel,double *train,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem,gpu;
	int total_s;
	double **delta_ptr;
	double Ep =0.;
	double Epr=0.;
#ifdef _CUBLAS
	double _alpha=1.0;
#endif /*_CUBLAS*/
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
	/*allocate delta_ptr*/
	cudaSetDevice(0);/*make sure all allocation happen on gpu[0]*/
	ALLOC(delta_ptr,_K.n_hiddens+1,DOUBLE *);/*HOST*/
	for(idx=0;idx<_K.n_hiddens;idx++)
		CUDA_ALLOC(delta_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
	CUDA_ALLOC(delta_ptr[_K.n_hiddens],_K.n_outputs,DOUBLE);/*DEVICE*/
/*+++ I - FORWARD +++*/
/*>>> in all cases, the FORWARD move should have already be done <<<*/
	Ep=scuda_ann_error(kernel,train,cudas);
//	printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
	scuda_ann_delta(kernel,train,delta_ptr,cudas);
/*+++ III - back propagation +++*/
/*^^^ output*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef   _CUBLAS
	_alpha=LEARN_RATE;
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
			_K.hiddens[_K.n_hiddens-1].cuda_v,1,delta_ptr[_K.n_hiddens]+jdx*red,
			1,_K.output.cuda_w+jdx*M*red,M);
		CHK_ERR(train_ger);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
		_K.hiddens[_K.n_hiddens-1].cuda_v,1,delta_ptr[_K.n_hiddens]+jdx*red,
		1,_K.output.cuda_w+jdx*M*red,M);
	CHK_ERR(train_ger);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,
			_K.hiddens[_K.n_hiddens-1].cuda_v,_K.output.cuda_w+jdx*M*red);
		CHK_ERR(train_ger_acc);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,
		_K.hiddens[_K.n_hiddens-1].cuda_v,_K.output.cuda_w+jdx*M*red);
	CHK_ERR(train_ger_acc);
#endif /*_CUBLAS*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*^^^ hiddens*/
	for(idx=(_K.n_hiddens-1);idx>0;idx--){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		red=N/cudas->cuda_n_streams;
		rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
		for(jdx=0;jdx<total_s-1;jdx++){
			gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
			cudaSetDevice(gpu);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
			_K.hiddens[idx-1].cuda_v,1,delta_ptr[idx]+jdx*red,1,
			_K.hiddens[idx].cuda_w+jdx*M*red,M);
			CHK_ERR(train_ger);
		}
		gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
		_K.hiddens[idx-1].cuda_v,1,delta_ptr[idx]+jdx*red,1,
		_K.hiddens[idx].cuda_w+jdx*M*red,M);
		CHK_ERR(train_ger);
#else  /*_CUBLAS*/
		for(jdx=0;jdx<total_s-1;jdx++){
			gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
			cudaSetDevice(gpu);
			ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,delta_ptr[idx]+jdx*red,
				_K.hiddens[idx-1].cuda_v,_K.hiddens[idx].cuda_w+jdx*M*red);
			CHK_ERR(train_ger_acc);
		}
		gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
		cudaSetDevice(gpu);
		ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,LEARN_RATE,delta_ptr[idx]+jdx*red,
			_K.hiddens[idx-1].cuda_v,_K.hiddens[idx].cuda_w+jdx*M*red);
		CHK_ERR(train_ger_acc);
#endif /*_CUBLAS*/
		for(gpu=0;gpu<cudas->n_gpu;gpu++){
			cudaSetDevice(gpu);
			cudaDeviceSynchronize();
		}
	}
	/*add zero*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef   _CUBLAS
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,_K.cuda_in,1,
			delta_ptr[0]+jdx*red,1,_K.hiddens[0].cuda_w+jdx*M*red,M);
		CHK_ERR(train_ger);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,_K.cuda_in,1,
		delta_ptr[0]+jdx*red,1,_K.hiddens[0].cuda_w+jdx*M*red,M);
	CHK_ERR(train_ger);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,delta_ptr[0]+jdx*red,_K.cuda_in,
			_K.hiddens[0].cuda_w+jdx*M*red);
		CHK_ERR(train_ger_acc);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,delta_ptr[0]+jdx*red,_K.cuda_in,
		_K.hiddens[0].cuda_w+jdx*M*red);
	CHK_ERR(train_ger_acc);
#endif /*_CUBLAS*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*+++ IV - update error +++*/
	/*update kernel*/
	scuda_ann_forward(kernel,cudas);
	Epr=scuda_ann_error(kernel,train,cudas);
//	fprintf(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ V - cleanup +++*/
	cudaSetDevice(0);/*make sure all de-allocation happen on gpu[0]*/
	for(idx=0;idx<(_K.n_hiddens+1);idx++){
		CUDA_FREE(delta_ptr[idx]);
		delta_ptr[idx]=NULL;
	}
	FREE(delta_ptr);
	CHK_ERR(free_1);
	return Ep-Epr;
}
/*------------------------------*/
/*+++ zeroes momentum arrays +++*/
/*------------------------------*/
void scuda_ann_raz_momentum(_kernel *kernel,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem,gpu;
	int total_s;
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
/*^^^ output*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/total_s;
	rem=N%total_s;
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cudaMemsetAsync(_K.cuda_dw[_K.n_hiddens]+jdx*M*red,0.,
			red*M*sizeof(double),cudas->cuda_streams[jdx]);
		CHK_ERR(moment_memset);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cudaMemsetAsync(_K.cuda_dw[_K.n_hiddens]+jdx*M*red,0.,
		(red+rem)*M*sizeof(double),cudas->cuda_streams[jdx]);
	CHK_ERR(moment_memset);
/*^^^ hiddens*/
	for(idx=(_K.n_hiddens-1);idx>0;idx--){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		red=N/total_s;
		rem=N%total_s;
		for(jdx=0;jdx<total_s-1;jdx++){
			gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
			cudaSetDevice(gpu);
			cudaMemsetAsync(_K.cuda_dw[idx]+jdx*M*red,0.,
				red*M*sizeof(double),cudas->cuda_streams[jdx]);
			CHK_ERR(moment_memset);
		}
		gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
		cudaSetDevice(gpu);
		cudaMemsetAsync(_K.cuda_dw[idx]+jdx*M*red,0.,
			(red+rem)*M*sizeof(double),cudas->cuda_streams[jdx]);
		CHK_ERR(moment_memset);
	}
	/*add zero*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	red=N/total_s;
	rem=N%total_s;
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cudaMemsetAsync(_K.cuda_dw[0]+jdx*M*red,0.,
			red*M*sizeof(double),cudas->cuda_streams[jdx]);
		CHK_ERR(moment_memset);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cudaMemsetAsync(_K.cuda_dw[0]+jdx*M*red,0.,
		(red+rem)*M*sizeof(double),cudas->cuda_streams[jdx]);
	CHK_ERR(moment_memset);
	/*all done, sync required*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
}
/*--------------------------------------*/
/*+++ back-propagation with momentum +++*/
/*--------------------------------------*/
double scuda_ann_train_momentum(_kernel *kernel,double *train,double moment,
								cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem,gpu;
        int total_s;
	double **delta_ptr;
	double Ep =0.;
	double Epr=0.;
	/**/
#ifdef _CUBLAS
	double _alpha=1.0;
	double _un=1.0;
	int kdx;
#endif /*_CUBLAS*/
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
	/*allocate delta_ptr*/
	cudaSetDevice(0);/*make sure all allocation happen on gpu[0]*/
	ALLOC(delta_ptr,_K.n_hiddens+1,DOUBLE *);/*HOST*/
	for(idx=0;idx<_K.n_hiddens;idx++)
		CUDA_ALLOC(delta_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
	CUDA_ALLOC(delta_ptr[_K.n_hiddens],_K.n_outputs,DOUBLE);/*DEVICE*/
/*+++ I - FORWARD +++*/
/*>>> in all cases, the FORWARD move should have already be done <<<*/
	Ep=scuda_ann_error(kernel,train,cudas);
///	printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
	scuda_ann_delta(kernel,train,delta_ptr,cudas);
/*+++ III - back propagation +++*/
/*^^^ output*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef   _CUBLAS
	_alpha=LEARN_RATE;
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
			_K.hiddens[_K.n_hiddens-1].cuda_v,1,delta_ptr[_K.n_hiddens]+jdx*red,
			1,_K.cuda_dw[_K.n_hiddens]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
			_K.cuda_dw[_K.n_hiddens]+jdx*M*red,1,_K.output.cuda_w+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
			_K.cuda_dw[_K.n_hiddens]+jdx*M*red,1);
		CHK_ERR(moment_scal);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
		_K.hiddens[_K.n_hiddens-1].cuda_v,1,delta_ptr[_K.n_hiddens]+jdx*red,
		1,_K.cuda_dw[_K.n_hiddens]+jdx*M*red,M);
	CHK_ERR(moment_ger);
	cublasDaxpy(cudas->cuda_handle[gpu],(red+rem)*M,&_un,
		_K.cuda_dw[_K.n_hiddens]+jdx*M*red,1,_K.output.cuda_w+jdx*M*red,1);
	CHK_ERR(moment_axpy);
	cublasDscal(cudas->cuda_handle[gpu],(red+rem)*M,&moment,
		_K.cuda_dw[_K.n_hiddens]+jdx*M*red,1);
	CHK_ERR(moment_scal);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,moment,delta_ptr[_K.n_hiddens]+jdx*red,
			_K.hiddens[_K.n_hiddens-1].cuda_v,
			_K.cuda_dw[_K.n_hiddens]+jdx*M*red,_K.output.cuda_w+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,moment,delta_ptr[_K.n_hiddens]+jdx*red,
		_K.hiddens[_K.n_hiddens-1].cuda_v,
		_K.cuda_dw[_K.n_hiddens]+jdx*M*red,_K.output.cuda_w+jdx*M*red);
	CHK_ERR(moment_ger_dw_acc);
#endif /*_CUBLAS*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*^^^ hiddens*/
	for(idx=(_K.n_hiddens-1);idx>0;idx--){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		red=N/total_s;
		rem=N%total_s;
#ifdef   _CUBLAS
		for(jdx=0;jdx<total_s-1;jdx++){
			gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
			cudaSetDevice(gpu);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_K.hiddens[idx-1].cuda_v,1,delta_ptr[idx]+jdx*red,1,
				_K.cuda_dw[idx]+jdx*M*red,M);
			CHK_ERR(moment_ger);
			cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
				_K.cuda_dw[idx]+jdx*M*red,1,_K.hiddens[idx].cuda_w+jdx*M*red,1);
			CHK_ERR(moment_axpy);
			cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
				_K.cuda_dw[idx]+jdx*M*red,1);
			CHK_ERR(moment_scal);
		}
		gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
			_K.hiddens[idx-1].cuda_v,1,delta_ptr[idx]+jdx*red,1,
			_K.cuda_dw[idx]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[gpu],(red+rem)*M,&_un,
			_K.cuda_dw[idx]+jdx*M*red,1,_K.hiddens[idx].cuda_w+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[gpu],(red+rem)*M,&moment,
			_K.cuda_dw[idx]+jdx*M*red,1);
		CHK_ERR(moment_scal);
#else  /*_CUBLAS*/
		for(jdx=0;jdx<total_s-1;jdx++){
			gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
			cudaSetDevice(gpu);
			ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,moment,delta_ptr[idx]+jdx*red,
				_K.hiddens[idx-1].cuda_v,
				_K.cuda_dw[idx]+jdx*M*red,_K.hiddens[idx].cuda_w+jdx*M*red);
			CHK_ERR(moment_ger_dw_acc);
		}
		gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
		cudaSetDevice(gpu);
		ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,LEARN_RATE,moment,delta_ptr[idx]+jdx*red,
			_K.hiddens[idx-1].cuda_v,
			_K.cuda_dw[idx]+jdx*M*red,_K.hiddens[idx].cuda_w+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
#endif /*_CUBLAS*/
		for(gpu=0;gpu<cudas->n_gpu;gpu++){
			cudaSetDevice(gpu);
			cudaDeviceSynchronize();
		}
	}
	/*add zero*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef   _CUBLAS
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,_K.cuda_in,1,
			delta_ptr[0]+jdx*red,1,_K.cuda_dw[0]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[gpu],
			red*M,&_un,_K.cuda_dw[0]+jdx*M*red,1,
				_K.hiddens[0].cuda_w+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
			_K.cuda_dw[0]+jdx*M*red,1);
		CHK_ERR(moment_scal);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,_K.cuda_in,1,
		delta_ptr[0]+jdx*red,1,_K.cuda_dw[0]+jdx*M*red,M);
	CHK_ERR(moment_ger);
	cublasDaxpy(cudas->cuda_handle[gpu],
		(red+rem)*M,&_un,_K.cuda_dw[0]+jdx*M*red,1,
				_K.hiddens[0].cuda_w+jdx*M*red,1);
	CHK_ERR(moment_axpy);
	cublasDscal(cudas->cuda_handle[gpu],(red+rem)*M,&moment,
		_K.cuda_dw[0]+jdx*M*red,1);
	CHK_ERR(moment_scal);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,moment,delta_ptr[0]+jdx*red,_K.cuda_in,
			_K.cuda_dw[0]+jdx*M*red,_K.hiddens[0].cuda_w+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,moment,delta_ptr[0]+jdx*red,_K.cuda_in,
		_K.cuda_dw[0]+jdx*M*red,_K.hiddens[0].cuda_w+jdx*M*red);
	CHK_ERR(moment_ger_dw_acc);
#endif /*_CUBLAS*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*+++ IV - update error +++*/
	/*update kernel*/
	scuda_ann_forward(kernel,cudas);
	Epr=scuda_ann_error(kernel,train,cudas);
//	fprintf(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ V - cleanup +++*/
	cudaSetDevice(0);/*make sure all de-allocation happen on gpu[0]*/
	for(idx=0;idx<(_K.n_hiddens+1);idx++){
		CUDA_FREE(delta_ptr[idx]);
		delta_ptr[idx]=NULL;
	}
	FREE(delta_ptr);
	CHK_ERR(free_1);
	return Ep-Epr;
}

}/*extern "C"*/
