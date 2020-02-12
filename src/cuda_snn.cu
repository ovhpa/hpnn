/*
+++ libhpnn - High Performance Neural Network library - file: cuda_snn.cu +++
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
#include <libhpnn/cuda_snn.h>

/*^^^ useful to launch kernels*/
#define _WP  32
#define _TPW 32
#define _TPB (_TPW*_WP)
#define _KG(n) ((n+_TPB-1)/(_TPB)),_TPB
/*---------------*/
/*+++ KERNELS +++*/
/*---------------*/
__global__
void fw_smax(int n, double dv, double *out){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		out[i] = exp( out[i] - 1.0 ) / dv;
}
__global__
void fw_scal(int n, double dv, double *out){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		out[i] = out[i] / dv;
}
__global__
void amb_smax(int n, double *res, double *train, double *out){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i += stride)
		res[i] = -1.0 * train[i] * log( out[i] + TINY );
}
__global__
void fw_s_acc(int m,int n, double *mat,double *vec,double *res){
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        double sum=0.;
        if(tid<n){
                /*a full line*/
                for(int i=0; i<m; i++) sum += vec[i]*mat[(tid*m)+i];
                res[tid]=sum;
        }
}
__global__
void amb_smax_acc(int n, double *res, double *train, double *out){
	extern __shared__ double sh_data[];
	int tid=threadIdx.x;
	int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
	double mySum;
	if(i<n){
		mySum=-1.0*train[i]*log(out[i]+TINY);
	}else{
		mySum=0.;
	}
	if(i+blockDim.x < n) 
		mySum += -1.0*train[i+blockDim.x]*log(out[i+blockDim.x]+TINY);
	sh_data[tid]=mySum;
	__syncthreads();
	/*reduction in shared memory*/
	for(int s=blockDim.x/2;s>0;s>>=1){
		if(tid<s) sh_data[tid] += sh_data[tid+s];
		__syncthreads();
	}
	/*result*/
	if(tid==0) res[blockIdx.x]=sh_data[0];
}
__global__
void dv_acc(int n,double *res,double *out){
	extern __shared__ double sh_data[];
	int tid=threadIdx.x;
	int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
	double mySum = (i < n) ? exp(out[i]-1.0) : 0;
	if(i+blockDim.x < n) mySum += exp(out[i+blockDim.x]-1.0);
	sh_data[tid]=mySum;
	__syncthreads();
	/*reduction in shared memory*/
	for(int s=blockDim.x/2;s>0;s>>=1){
		if(tid<s) sh_data[tid] += sh_data[tid+s];
		__syncthreads();
	}
	/*result*/
	if(tid==0) res[blockIdx.x]=sh_data[0];
}
__global__
void dsmax_diff(int n, double *t, double *o, double *y){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = ( t[i] - o[i] );
}
/*calculate exp(x-1) _and_ accumulate dv*/
__global__
void softmax_acc(int n,double *res,double *out){
	extern __shared__ double sh_data[];
	int tid=threadIdx.x;
	int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
	double mySum;
	if(i<n){
		out[i] = exp(out[i]-1.0);
		mySum = out[i];
	}else{
		mySum = 0.;
	}
	if(i+blockDim.x < n) {
		out[i+blockDim.x] = exp(out[i+blockDim.x]-1.0);
		mySum += out[i+blockDim.x];
	}
	sh_data[tid]=mySum;
	__syncthreads();
	/*reduction in shared memory*/
	for(int s=blockDim.x/2;s>0;s>>=1){
		if(tid<s) sh_data[tid] += sh_data[tid+s];
		__syncthreads();
	}
	/*result*/
	if(tid==0) res[blockIdx.x]=sh_data[0];
}
/*-----------------*/
/* The C interface */
/*-----------------*/
extern "C"{
#define _K (*kernel)
#define _Kx (*kx)
/*-----------------------------*/
/*+++ forward kernel update +++*/
/*-----------------------------*/
void scuda_snn_forward(kernel_ann *kernel,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem,gpu;
	int total_s;
#ifdef _CUBLAS
	double _alpha=1.0;
	double _beta =0.0;
#endif
	double dv;
	kernel_ann *kx;
	int kdx;
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
/*+++ I - input +++*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef   _CUBLAS
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
				_K.hiddens[0].weights+jdx*M*red,M,
				_K.in,1,&_beta,_K.hiddens[0].vec+jdx*red,1);
			CHK_ERR(fw_gemv);
			sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.hiddens[0].vec+jdx*red);
			CHK_ERR(fw_sigmoid);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
			_K.hiddens[0].weights+jdx*M*red,M,
			_K.in,1,&_beta,_K.hiddens[0].vec+jdx*red,1);
		CHK_ERR(fw_gemv);
		sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.hiddens[0].vec+jdx*red);
		CHK_ERR(fw_sigmoid);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle[gpu],
		CUBLAS_OP_T,M,red+rem,&_alpha,_K.hiddens[0].weights+jdx*M*red,M,
		_K.in,1,&_beta,_K.hiddens[0].vec+jdx*red,1);
	CHK_ERR(fw_gemv);
	sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_K.hiddens[0].vec+jdx*red);
	CHK_ERR(fw_sigmoid);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[0],CUBLAS_OP_T,M,red,&_alpha,
			_K.hiddens[0].weights+jdx*M*red,M,
			_K.in,1,&_beta,_K.hiddens[0].vec+jdx*red,1);
		CHK_ERR(fw_gemv);
		sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.hiddens[0].vec+jdx*red);
		CHK_ERR(fw_sigmoid);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
				_Kx.hiddens[0].weights+jdx*M*red,M,
				_Kx.in,1,&_beta,_Kx.hiddens[0].vec+jdx*red,1);
			CHK_ERR(fw_gemv);
			sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_Kx.hiddens[0].vec+jdx*red);
			CHK_ERR(fw_sigmoid);
			/*transfer to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[0].vec+jdx*red,
				_Kx.hiddens[0].vec+jdx*red,red,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
			CHK_ERR(fw_vec_cpy);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
			_Kx.hiddens[0].weights+jdx*M*red,M,
			_Kx.in,1,&_beta,_Kx.hiddens[0].vec+jdx*red,1);
		CHK_ERR(fw_gemv);
		sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_Kx.hiddens[0].vec+jdx*red);
		CHK_ERR(fw_sigmoid);
		/*transfer to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[0].vec+jdx*red,
			_Kx.hiddens[0].vec+jdx*red,red,cudaMemcpyDeviceToDevice,
			cudas->cuda_streams[jdx]);
		CHK_ERR(fw_vec_cpy);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red+rem,&_alpha,
		_Kx.hiddens[0].weights+jdx*M*red,M,
		_Kx.in,1,&_beta,_Kx.hiddens[0].vec+jdx*red,1);
	CHK_ERR(fw_gemv);
	sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_Kx.hiddens[0].vec+jdx*red);
	CHK_ERR(fw_sigmoid);
	/*transfer to GPU[0]*/
	cudaMemcpyAsync(_K.hiddens[0].vec+jdx*red,
		_Kx.hiddens[0].vec+jdx*red,red+rem,cudaMemcpyDeviceToDevice,
		cudas->cuda_streams[jdx]);
	CHK_ERR(fw_vec_cpy);
	/*put back vec from GPU[0] to all GPUs*/
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		for(gpu=1;gpu<cudas->n_gpu;gpu++){
			kx=_K.kerns[gpu];
			cudaMemcpyAsync(_Kx.hiddens[0].vec+jdx*red,
							_K.hiddens[0].vec+jdx*red,
							red,cudaMemcpyDeviceToDevice,
							cudas->cuda_streams[jdx]);
		}
	}
	/*broadcast the last piece*/
	for(gpu=1;gpu<cudas->n_gpu;gpu++){
		kx=_K.kerns[gpu];
		cudaMemcpyAsync(_Kx.hiddens[0].vec+jdx*red,
						_K.hiddens[0].vec+jdx*red,
						red+rem,cudaMemcpyDeviceToDevice,
						cudas->cuda_streams[jdx]);
	}
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,_K.hiddens[0].weights+jdx*M*red,_K.in,
				_K.hiddens[0].vec+jdx*red);
			CHK_ERR(fw_mv_acc);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.hiddens[0].weights+jdx*M*red,_K.in,
			_K.hiddens[0].vec+jdx*red);
		CHK_ERR(fw_mv_acc);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_K.hiddens[0].weights+jdx*M*red,_K.in,
			_K.hiddens[0].vec+jdx*red);
	CHK_ERR(fw_mv_acc);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.hiddens[0].weights+jdx*M*red,_K.in,
			_K.hiddens[0].vec+jdx*red);
		CHK_ERR(fw_mv_acc);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,_Kx.hiddens[0].weights+jdx*M*red,_Kx.in,
				_Kx.hiddens[0].vec+jdx*red);
			CHK_ERR(fw_mv_acc);
			/*transfer to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[0].vec+jdx*red,
				_Kx.hiddens[0].vec+jdx*red,red,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
			CHK_ERR(fw_vec_cpy);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_Kx.hiddens[0].weights+jdx*M*red,_Kx.in,
			_Kx.hiddens[0].vec+jdx*red);
		CHK_ERR(fw_mv_acc);
		/*transfer to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[0].vec+jdx*red,
			_Kx.hiddens[0].vec+jdx*red,red,cudaMemcpyDeviceToDevice,
			cudas->cuda_streams[jdx]);
		CHK_ERR(fw_vec_cpy);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_Kx.hiddens[0].weights+jdx*M*red,_Kx.in,
			_Kx.hiddens[0].vec+jdx*red);
	CHK_ERR(fw_mv_acc);
	/*transfer to GPU[0]*/
	cudaMemcpyAsync(_K.hiddens[0].vec+jdx*red,
		_Kx.hiddens[0].vec+jdx*red,red+rem,cudaMemcpyDeviceToDevice,
		cudas->cuda_streams[jdx]);
	CHK_ERR(fw_vec_cpy);
	/*put back vec from GPU[0] to all GPUs*/
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		for(gpu=1;gpu<cudas->n_gpu;gpu++){
			kx=_K.kerns[gpu];
			cudaMemcpyAsync(_Kx.hiddens[0].vec+jdx*red,
							_K.hiddens[0].vec+jdx*red,
							red,cudaMemcpyDeviceToDevice,
							cudas->cuda_streams[jdx]);
		}
	}
	/*broadcast the last piece*/
	for(gpu=1;gpu<cudas->n_gpu;gpu++){
		kx=_K.kerns[gpu];
		cudaMemcpyAsync(_Kx.hiddens[0].vec+jdx*red,
						_K.hiddens[0].vec+jdx*red,
						red+rem,cudaMemcpyDeviceToDevice,
						cudas->cuda_streams[jdx]);
	}
}
#endif /*_CUBLAS*/
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
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
		for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				cublasSetStream(cudas->cuda_handle[gpu],
								cudas->cuda_streams[jdx]);
				cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
					_K.hiddens[idx].weights+jdx*M*red,M,
					_K.hiddens[idx-1].vec,1,&_beta,
					_K.hiddens[idx].vec+jdx*red,1);
				CHK_ERR(cublas_1);
				sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,_K.hiddens[idx].vec+jdx*red);
				CHK_ERR(fw_sigmoid);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],
							cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
				_K.hiddens[idx].weights+jdx*M*red,M,
				_K.hiddens[idx-1].vec,1,&_beta,
				_K.hiddens[idx].vec+jdx*red,1);
			CHK_ERR(cublas_1);
			sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.hiddens[idx].vec+jdx*red);
			CHK_ERR(fw_sigmoid);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],
			CUBLAS_OP_T,M,red+rem,&_alpha,
			_K.hiddens[idx].weights+jdx*M*red,M,
			_K.hiddens[idx-1].vec,1,&_beta,
			_K.hiddens[idx].vec+jdx*red,1);
		CHK_ERR(cublas_1);
		sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,_K.hiddens[idx].vec+jdx*red);
		CHK_ERR(fw_sigmoid);
}else{
/*>>> first GPU[0]*/
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[0],CUBLAS_OP_T,M,red,&_alpha,
				_K.hiddens[idx].weights+jdx*M*red,M,
				_K.hiddens[idx-1].vec,1,&_beta,
				_K.hiddens[idx].vec+jdx*red,1);
			CHK_ERR(cublas_1);
			sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.hiddens[idx].vec+jdx*red);
			CHK_ERR(fw_sigmoid);
		}
/*>>> next GPUs but the last one*/
		for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			kx=_K.kerns[gpu];
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				cublasSetStream(cudas->cuda_handle[gpu],
								cudas->cuda_streams[jdx]);
				cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
					_Kx.hiddens[idx].weights+jdx*M*red,M,
					_Kx.hiddens[idx-1].vec,1,&_beta,
					_Kx.hiddens[idx].vec+jdx*red,1);
				CHK_ERR(cublas_1);
				sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,_Kx.hiddens[idx].vec+jdx*red);
				CHK_ERR(fw_sigmoid);
				/*transfer to GPU[0]*/
				cudaMemcpyAsync(_K.hiddens[idx].vec+jdx*red,
					_Kx.hiddens[idx].vec+jdx*red,red,cudaMemcpyDeviceToDevice,
					cudas->cuda_streams[jdx]);
				CHK_ERR(fw_vec_cpy);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],
							cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
				_Kx.hiddens[idx].weights+jdx*M*red,M,
				_Kx.hiddens[idx-1].vec,1,&_beta,
				_Kx.hiddens[idx].vec+jdx*red,1);
				CHK_ERR(cublas_1);
			sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_Kx.hiddens[idx].vec+jdx*red);
			CHK_ERR(fw_sigmoid);
			/*transfer to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[idx].vec+jdx*red,
				_Kx.hiddens[idx].vec+jdx*red,red,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
			CHK_ERR(fw_vec_cpy);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red+rem,&_alpha,
			_Kx.hiddens[idx].weights+jdx*M*red,M,
			_Kx.hiddens[idx-1].vec,1,&_beta,
			_Kx.hiddens[idx].vec+jdx*red,1);
		CHK_ERR(cublas_1);
		sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,_Kx.hiddens[idx].vec+jdx*red);
		CHK_ERR(kernel_1);
		/*transfer to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[idx].vec+jdx*red,
			_Kx.hiddens[idx].vec+jdx*red,red+rem,cudaMemcpyDeviceToDevice,
			cudas->cuda_streams[jdx]);
		CHK_ERR(fw_vec_cpy);
		/*put back vec from GPU[0] to all GPUs*/
		red=N/cudas->cuda_n_streams;
		rem=N%cudas->cuda_n_streams;
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			for(gpu=1;gpu<cudas->n_gpu;gpu++){
				kx=_K.kerns[gpu];
				cudaMemcpyAsync(_Kx.hiddens[idx].vec+jdx*red,
					_K.hiddens[idx].vec+jdx*red,red,cudaMemcpyDeviceToDevice,
					cudas->cuda_streams[jdx]);
			}
		}
		/*broadcast the last piece*/
		for(gpu=1;gpu<cudas->n_gpu;gpu++){
			kx=_K.kerns[gpu];
			cudaMemcpyAsync(_Kx.hiddens[idx].vec+jdx*red,
				_K.hiddens[idx].vec+jdx*red,red+rem,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
		}
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
		for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(M,red,_K.hiddens[idx].weights+jdx*M*red,
					_K.hiddens[idx-1].vec,_K.hiddens[idx].vec+jdx*red);
				CHK_ERR(fw_mv_acc);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,_K.hiddens[idx].weights+jdx*M*red,
				_K.hiddens[idx-1].vec,_K.hiddens[idx].vec+jdx*red);
			CHK_ERR(fw_mv_acc);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,_K.hiddens[idx].weights+jdx*M*red,
			_K.hiddens[idx-1].vec,_K.hiddens[idx].vec+jdx*red);
		CHK_ERR(fw_mv_acc);
}else{
/*>>> first GPU[0]*/
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,_K.hiddens[idx].weights+jdx*M*red,
				_K.hiddens[idx-1].vec,_K.hiddens[idx].vec+jdx*red);
			CHK_ERR(fw_mv_acc);
		}
/*>>> next GPUs but the last one*/
		for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			kx=_K.kerns[gpu];
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(M,red,_Kx.hiddens[idx].weights+jdx*M*red,
					_Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].vec+jdx*red);
				CHK_ERR(fw_mv_acc);
				/*transfer to GPU[0]*/
				cudaMemcpyAsync(_K.hiddens[idx].vec+jdx*red,
					_Kx.hiddens[idx].vec+jdx*red,red,cudaMemcpyDeviceToDevice,
					cudas->cuda_streams[jdx]);
				CHK_ERR(fw_vec_cpy);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,_Kx.hiddens[idx].weights+jdx*M*red,
				_Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].vec+jdx*red);
			CHK_ERR(fw_mv_acc);
			/*transfer to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[idx].vec+jdx*red,
				_Kx.hiddens[idx].vec+jdx*red,red,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
			CHK_ERR(fw_vec_cpy);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,_Kx.hiddens[idx].weights+jdx*M*red,
			_Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].vec+jdx*red);
		CHK_ERR(fw_mv_acc);
		/*transfer to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[idx].vec+jdx*red,
			_Kx.hiddens[idx].vec+jdx*red,red+rem,cudaMemcpyDeviceToDevice,
			cudas->cuda_streams[jdx]);
		CHK_ERR(fw_vec_cpy);
		/*put back vec from GPU[0] to all GPUs*/
		red=N/cudas->cuda_n_streams;
		rem=N%cudas->cuda_n_streams;
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			for(gpu=1;gpu<cudas->n_gpu;gpu++){
				kx=_K.kerns[gpu];
				cudaMemcpyAsync(_Kx.hiddens[idx].vec+jdx*red,
					_K.hiddens[idx].vec+jdx*red,red,cudaMemcpyDeviceToDevice,
					cudas->cuda_streams[jdx]);
			}
		}
		/*broadcast the last piece*/
		for(gpu=1;gpu<cudas->n_gpu;gpu++){
			kx=_K.kerns[gpu];
			cudaMemcpyAsync(_Kx.hiddens[idx].vec+jdx*red,
				_K.hiddens[idx].vec+jdx*red,red+rem,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
		}
}
#endif /*_CUBLAS*/
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
	dv=TINY;
#ifdef   _CUBLAS
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
				_K.output.weights+jdx*M*red,M,
				_K.hiddens[_K.n_hiddens-1].vec,1,&_beta,
				_K.output.vec+jdx*red,1);
			CHK_ERR(fw_gemv);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
			_K.output.weights+jdx*M*red,M,
			_K.hiddens[_K.n_hiddens-1].vec,1,&_beta,
			_K.output.vec+jdx*red,1);
		CHK_ERR(fw_gemv);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red+rem,&_alpha,
		_K.output.weights+jdx*M*red,M,
		_K.hiddens[_K.n_hiddens-1].vec,1,&_beta,
		_K.output.vec+jdx*red,1);
	CHK_ERR(fw_gemv);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[0],CUBLAS_OP_T,M,red,&_alpha,
			_K.output.weights+jdx*M*red,M,
			_K.hiddens[_K.n_hiddens-1].vec,1,&_beta,
			_K.output.vec+jdx*red,1);
		CHK_ERR(fw_gemv);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
				_Kx.output.weights+jdx*M*red,M,
				_Kx.hiddens[_Kx.n_hiddens-1].vec,1,&_beta,
				_Kx.output.vec+jdx*red,1);
			/*transfer to GPU[0]*/
			cudaMemcpyAsync(_K.output.vec+jdx*red,_Kx.output.vec+jdx*red,
				red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(fw_vec_cpy);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,&_alpha,
			_Kx.output.weights+jdx*M*red,M,
			_Kx.hiddens[_Kx.n_hiddens-1].vec,1,&_beta,
			_Kx.output.vec+jdx*red,1);
		/*transfer to GPU[0]*/
		cudaMemcpyAsync(_K.output.vec+jdx*red,_Kx.output.vec+jdx*red,
			red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(fw_vec_cpy);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle[gpu],
		CUBLAS_OP_T,M,red+rem,&_alpha,_K.output.weights+jdx*M*red,M,
		_K.hiddens[_K.n_hiddens-1].vec,1,&_beta,_K.output.vec+jdx*red,1);
	CHK_ERR(fw_gemv);
	/*transfer to GPU[0]*/
	cudaMemcpyAsync(_K.output.vec+jdx*red,_Kx.output.vec+jdx*red,
		red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(fw_vec_cpy);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			fw_s_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,_K.output.weights+jdx*M*red,
				_K.hiddens[_K.n_hiddens-1].vec,_K.output.vec+jdx*red);
			CHK_ERR(fw_s_acc);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		fw_s_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.output.weights+jdx*M*red,
			_K.hiddens[_K.n_hiddens-1].vec,_K.output.vec+jdx*red);
		CHK_ERR(fw_s_acc);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	fw_s_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_K.output.weights+jdx*M*red,
		_K.hiddens[_K.n_hiddens-1].vec,_K.output.vec+jdx*red);
	CHK_ERR(fw_s_acc);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		fw_s_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.output.weights+jdx*M*red,
			_K.hiddens[_K.n_hiddens-1].vec,_K.output.vec+jdx*red);
		CHK_ERR(fw_s_acc);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			fw_s_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,_Kx.output.weights+jdx*M*red,
				_Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.vec+jdx*red);
			CHK_ERR(fw_s_acc);
			/*transfer to GPU[0]*/
			cudaMemcpyAsync(_K.output.vec+jdx*red,_Kx.output.vec+jdx*red,
				red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(fw_vec_cpy);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		fw_s_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_Kx.output.weights+jdx*M*red,
			_Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.vec+jdx*red);
		CHK_ERR(fw_s_acc);
		/*transfer to GPU[0]*/
		cudaMemcpyAsync(_K.output.vec+jdx*red,_Kx.output.vec+jdx*red,
			red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(fw_vec_cpy);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	fw_s_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_Kx.output.weights+jdx*M*red,
		_Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.vec+jdx*red);
	CHK_ERR(fw_s_acc);
	/*transfer to GPU[0] (we are not on GPU[0])*/
	cudaMemcpyAsync(_K.output.vec+jdx*red,_Kx.output.vec+jdx*red,
		red+rem,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(fw_vec_cpy);
}
#endif /*_CUBLAS*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*+++ IV - softmax TODO: optimize on multi-GPUs +++*/
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
	cudaSetDevice(0);/*only on master GPU*/
	softmax_acc<<<_KG(N),sizeof(double)*2*(_TPB)>>>
		(N,_K.tmp_gpu,_K.output.vec);
	CHK_ERR(fw_softmax_acc);
	/*SOFTMAX: calculate dv*/
	CUDA_G2C_CP(&dv,&(_K.tmp_gpu[0]),1,double);
	dv+=TINY;
	/*SOFTMAX: calculate output*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		fw_scal<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,dv,_K.output.vec+jdx*red);
			CHK_ERR(fw_scal);
	}
	fw_scal<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,dv,_K.output.vec+jdx*red);
	/*put back vec from GPU[0] to all GPUs*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		for(gpu=1;gpu<cudas->n_gpu;gpu++){
			kx=_K.kerns[gpu];
			cudaMemcpyAsync(_Kx.output.vec+jdx*red,_K.output.vec+jdx*red,
				red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		}
	}
	/*broadcast the last piece*/
	for(gpu=1;gpu<cudas->n_gpu;gpu++){
		kx=_K.kerns[gpu];
		cudaMemcpyAsync(_Kx.output.vec+jdx*red,_K.output.vec+jdx*red,
			red+rem,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	}
	cudaDeviceSynchronize();/*sync GPU[0]*/
}
/*--------------------------------*/
/*+++ Calculate Training Error +++*/
/*--------------------------------*/
double scuda_snn_error(kernel_ann *kernel,double *train,cudastreams *cudas){
	int     jdx;
	int   N,red;
	int rem,gpu;
	int total_s;
	double dEp=0.;
	kernel_ann *kx;
	double **ptr;/*this is used as a GPU-local storage for CUDA_MEM_EXP*/
	int kdx;
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
	ALLOC(ptr,cudas->n_gpu,DOUBLE *);/*HOST*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
	for(gpu=0;gpu<cudas->n_gpu;gpu++) ptr[gpu]=NULL;
}else{
	ptr[0]=NULL;/*wasted for clarity*/
	for(gpu=1;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		CUDA_ALLOC(ptr[gpu],_Kx.max_index,DOUBLE);
	}
}
/**/
	N=_K.n_outputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef _CUBLAS
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			amb_smax<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.tmp_gpu+jdx*red,train+jdx*red,_K.output.vec+jdx*red);
			CHK_ERR(err_amb_smax);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		amb_smax<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.tmp_gpu+jdx*red,train+jdx*red,_K.output.vec+jdx*red);
		CHK_ERR(err_amb_smax);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	amb_smax<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_K.tmp_gpu+jdx*red,train+jdx*red,_K.output.vec+jdx*red);
	CHK_ERR(err_amb_smax);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
		amb_smax<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.tmp_gpu+jdx*red,train+jdx*red,_K.output.vec+jdx*red);
		CHK_ERR(err_amb_smax);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get part of train from GPU[0]*/
		cudaMemcpy(ptr[gpu],train+gpu*(cudas->cuda_n_streams)*red,
			cudas->cuda_n_streams*red,cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate on tmp_gpu*/
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			amb_smax<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_Kx.tmp_gpu+jdx*red,
				ptr[gpu]+kdx*red,_Kx.output.vec+jdx*red);
			CHK_ERR(err_amb_smax);
			/*3- send result to tmp_gpu on GPU[0]*/
			cudaMemcpyAsync(_K.tmp_gpu+jdx*red,_Kx.tmp_gpu+jdx*red,red,
				cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(error_transfer);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get part of train from GPU[0]*/
	cudaMemcpy(ptr[gpu],train+gpu*(cudas->cuda_n_streams)*red,
		cudas->cuda_n_streams*red+rem,cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate on tmp_gpu*/
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		amb_smax<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_Kx.tmp_gpu+jdx*red,
			ptr[gpu]+kdx*red,_Kx.output.vec+jdx*red);
		CHK_ERR(err_amb_smax);
		/*3- send result to tmp_gpu on GPU[0]*/
		cudaMemcpyAsync(_K.tmp_gpu+jdx*red,_Kx.tmp_gpu+jdx*red,red,
			cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(error_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	amb_smax<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_Kx.tmp_gpu+jdx*red,
		ptr[gpu]+kdx*red,_Kx.output.vec+jdx*red);
	CHK_ERR(err_amb_smax);
	/*3- send result to tmp_gpu on GPU[0]*/
	cudaMemcpyAsync(_K.tmp_gpu+jdx*red,_Kx.tmp_gpu+jdx*red,red+rem,
		cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(error_transfer);
}
	/*get dEp*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
	/*Dasum only on gpu[0] TODO: optimize on multi-GPU*/
	cudaSetDevice(0);
	cublasSetStream(cudas->cuda_handle[0],NULL);
	cublasDasum(cudas->cuda_handle[0],N,_K.tmp_gpu,1,&dEp);
	CHK_ERR(err_asum);
#else /*_CUBLAS*/
	/*TODO: optimize on multi-GPU*/
	cudaSetDevice(0);/*only on master GPU*/
	amb_smax_acc<<<_KG(_K.n_outputs),sizeof(double)*2*(_TPB)>>>
		(_K.n_outputs,_K.tmp_gpu,train,_K.output.vec);
	CHK_ERR(err_amb_smax_acc);
	CUDA_G2C_CP(&dEp,&(_K.tmp_gpu[0]),1,double);
	CHK_ERR(err_g2c_cp);
#endif /*_CUBLAS*/
	cudaDeviceSynchronize();/*only gpu[0]*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
	FREE(ptr);
}else{
	for(gpu=1;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		CUDA_FREE(ptr[gpu]);
	}
	FREE(ptr);
}
	dEp/=((double)N);
	return dEp;
}
/*------------------------*/
/*+++ Calculate deltas +++*/
/*------------------------*/
void scuda_snn_delta(kernel_ann *kernel,double *train,double **delta_ptr,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem,gpu;
	int total_s;
#ifdef _CUBLAS
	double _alpha=1.0;
	double _beta =0.0;
#endif /*_CUBLAS*/
	kernel_ann *kx;
	double **ptr;/*this is used as a GPU-local storage for CUDA_MEM_EXP*/
	int kdx;
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
	ALLOC(ptr,cudas->n_gpu,DOUBLE *);/*HOST*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
	for(gpu=0;gpu<cudas->n_gpu;gpu++) ptr[gpu]=NULL;
}else{
	ptr[0]=NULL;/*wasted for clarity*/
	for(gpu=1;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		CUDA_ALLOC(ptr[gpu],_Kx.max_index,DOUBLE);
	}
}
/*^^^ output*/
	N=_K.n_outputs;
	red=N/total_s;
	rem=N%total_s;
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			dsmax_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,train+jdx*red,_K.output.vec+jdx*red,
				delta_ptr[_K.n_hiddens]+jdx*red);
			CHK_ERR(train_dsmax_dif);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		dsmax_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,train+jdx*red,_K.output.vec+jdx*red,
			delta_ptr[_K.n_hiddens]+jdx*red);
		CHK_ERR(train_dsmax_dif);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	dsmax_diff<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,train+jdx*red,_K.output.vec+jdx*red,
		delta_ptr[_K.n_hiddens]+jdx*red);
	CHK_ERR(train_dsmax_dif);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		dsmax_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,train+jdx*red,_K.output.vec+jdx*red,
			delta_ptr[_K.n_hiddens]+jdx*red);
		CHK_ERR(train_dsmax_dif);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get part of train from GPU[0]*/
		cudaMemcpy(ptr[gpu],train+gpu*(cudas->cuda_n_streams),
				   cudas->cuda_n_streams*red,cudaMemcpyDeviceToDevice);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			dsmax_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,ptr[gpu]+kdx*red,_Kx.output.vec+jdx*red,
				_Kx.tmp_gpu+jdx*red);
			CHK_ERR(train_dsmax_dif);
			/*send back data to delta_ptr*/
			cudaMemcpyAsync(delta_ptr[_Kx.n_hiddens]+jdx*red,
				_Kx.tmp_gpu+jdx*red,red,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get part of train from GPU[0]*/
	cudaMemcpy(ptr[gpu],train+gpu*(cudas->cuda_n_streams),
			   cudas->cuda_n_streams*red+rem,cudaMemcpyDeviceToDevice);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		dsmax_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,ptr[gpu]+kdx*red,_Kx.output.vec+jdx*red,_Kx.tmp_gpu+jdx*red);
		CHK_ERR(train_dsmax_dif);
		/*send back data to delta_ptr*/
		cudaMemcpyAsync(delta_ptr[_Kx.n_hiddens]+jdx*red,
			_Kx.tmp_gpu+jdx*red,red,cudaMemcpyDeviceToDevice,
			cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	dsmax_diff<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,ptr[gpu]+kdx*red,_K.output.vec+jdx*red,_Kx.tmp_gpu+jdx*red);
	CHK_ERR(train_dsmax_dif);
	/*send back data to delta_ptr*/
	cudaMemcpyAsync(delta_ptr[_Kx.n_hiddens]+jdx*red,_Kx.tmp_gpu+jdx*red,
					red+rem,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*^^^ output to hidden NOTE: from here calculation is same as for NN_TYPE_ANN*/
	/*distribution over M due to transposed operations*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=M/total_s;
	rem=M%total_s;
#ifdef   _CUBLAS
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
			&_alpha,_K.output.weights+jdx*red,M,delta_ptr[_K.n_hiddens],1,
			&_beta,delta_ptr[_K.n_hiddens-1]+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.hiddens[_K.n_hiddens-1].vec+jdx*red,
				delta_ptr[_K.n_hiddens-1]+jdx*red);
			CHK_ERR(train_dsigmoid);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
		&_alpha,_K.output.weights+jdx*red,M,delta_ptr[_K.n_hiddens],1,
		&_beta,delta_ptr[_K.n_hiddens-1]+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.hiddens[_K.n_hiddens-1].vec+jdx*red,
			delta_ptr[_K.n_hiddens-1]+jdx*red);
		CHK_ERR(train_dsigmoid);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
	&_alpha,_K.output.weights+jdx*red,M,delta_ptr[_K.n_hiddens],1,
	&_beta,delta_ptr[_K.n_hiddens-1]+jdx*red,1);
	CHK_ERR(train_gemv);
	dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_K.hiddens[_K.n_hiddens-1].vec+jdx*red,
		delta_ptr[_K.n_hiddens-1]+jdx*red);
	CHK_ERR(train_dsigmoid);
}else{
	/*delta[_K.n_hiddens-1] is the destination, on GPU[0]*/
	/*delta[_K.n_hiddens] is the source, also on GPU[0]..*/
	/*tmp_gpu stores the local result and ptr[gpu] stores*/
	/*a local copy of delta[_K.n_hiddens], on each GPU...*/
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[0],CUBLAS_OP_N,red,N,
		&_alpha,_K.output.weights+jdx*red,M,delta_ptr[_K.n_hiddens],1,
		&_beta,delta_ptr[_K.n_hiddens-1]+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.hiddens[_K.n_hiddens-1].vec+jdx*red,
			delta_ptr[_K.n_hiddens-1]+jdx*red);
		CHK_ERR(train_dsigmoid);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[_K.n_hiddens] from GPU[0]*/
		cudaMemcpy(ptr[gpu],delta_ptr[_Kx.n_hiddens],
				   M,cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate tmp_gpu = delta[_K.n_hiddens-1]*/
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
			&_alpha,_Kx.output.weights+jdx*red,M,ptr[gpu],1,
			&_beta,_Kx.tmp_gpu+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,
				_K.hiddens[_K.n_hiddens-1].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
			CHK_ERR(train_dsigmoid);
			/*3- send result to delta[_K.n_hiddens-1] on GPU[0]*/
			cudaMemcpyAsync(delta_ptr[_Kx.n_hiddens-1]+jdx*red,
				_Kx.tmp_gpu+jdx*red,red,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get full delta[_K.n_hiddens] from GPU[0]*/
	cudaMemcpy(ptr[gpu],delta_ptr[_Kx.n_hiddens],M,cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate tmp_gpu = delta[_K.n_hiddens-1]*/
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
		&_alpha,_Kx.output.weights+jdx*red,M,ptr[gpu],1,
		&_beta,_Kx.tmp_gpu+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,
			_K.hiddens[_K.n_hiddens-1].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
		CHK_ERR(train_dsigmoid);
		/*3- send result to delta[_K.n_hiddens-1] on GPU[0]*/
		cudaMemcpyAsync(delta_ptr[_Kx.n_hiddens-1]+jdx*red,
			_Kx.tmp_gpu+jdx*red,red,cudaMemcpyDeviceToDevice,
			cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
	&_alpha,_Kx.output.weights+jdx*red,M,ptr[gpu],1,
	&_beta,_Kx.tmp_gpu+jdx*red,1);
	CHK_ERR(train_gemv);
	dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_Kx.hiddens[_Kx.n_hiddens-1].vec+jdx*red,
		_Kx.tmp_gpu+jdx*red);
	CHK_ERR(train_dsigmoid);
	/*send result to delta[_K.n_hiddens-1] on GPU[0]*/
	cudaMemcpyAsync(delta_ptr[_Kx.n_hiddens-1]+jdx*red,_Kx.tmp_gpu+jdx*red,
		red+rem,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,M,N,_K.output.weights+jdx*red,delta_ptr[_K.n_hiddens],
				_K.hiddens[_K.n_hiddens-1].vec+jdx*red,
				delta_ptr[_K.n_hiddens-1]+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,M,N,_K.output.weights+jdx*red,delta_ptr[_K.n_hiddens],
			_K.hiddens[_K.n_hiddens-1].vec+jdx*red,
			delta_ptr[_K.n_hiddens-1]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,M,N,_K.output.weights+jdx*red,delta_ptr[_K.n_hiddens],
		_K.hiddens[_K.n_hiddens-1].vec+jdx*red,
		delta_ptr[_K.n_hiddens-1]+jdx*red);
	CHK_ERR(train_dsigmoid_mul_delta_T);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
		(red,M,N,_K.output.weights+jdx*red,delta_ptr[_K.n_hiddens],
		_K.hiddens[_K.n_hiddens-1].vec+jdx*red,
		delta_ptr[_K.n_hiddens-1]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[_K.n_hiddens] from GPU[0]*/
		cudaMemcpy(ptr[gpu],delta_ptr[_Kx.n_hiddens],
				   M,cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate tmp_gpu = delta[_K.n_hiddens-1]*/
			dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,M,N,_Kx.output.weights+jdx*red,ptr[gpu],
			_Kx.hiddens[_Kx.n_hiddens-1].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
			/*3- send result to delta[_K.n_hiddens-1] on GPU[0]*/
			cudaMemcpyAsync(delta_ptr[_Kx.n_hiddens-1]+jdx*red,
				_Kx.tmp_gpu+jdx*red,red,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get full delta[_K.n_hiddens] from GPU[0]*/
	cudaMemcpy(ptr[gpu],delta_ptr[_Kx.n_hiddens],M,cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate tmp_gpu = delta[_K.n_hiddens-1]*/
		dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
		(red,M,N,_Kx.output.weights+jdx*red,ptr[gpu],
		_Kx.hiddens[_Kx.n_hiddens-1].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
		/*3- send result to delta[_K.n_hiddens-1] on GPU[0]*/
		cudaMemcpyAsync(delta_ptr[_Kx.n_hiddens-1]+jdx*red,
			_Kx.tmp_gpu+jdx*red,red,cudaMemcpyDeviceToDevice,
			cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,M,N,_Kx.output.weights+jdx*red,ptr[gpu],
		_Kx.hiddens[_Kx.n_hiddens-1].vec+jdx*red,
		_Kx.tmp_gpu+jdx*red);
	CHK_ERR(train_dsigmoid_mul_delta_T);
	/*send result to delta[_K.n_hiddens-1] on GPU[0]*/
	cudaMemcpyAsync(delta_ptr[_Kx.n_hiddens-1]+jdx*red,_Kx.tmp_gpu+jdx*red,
		red+rem,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
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
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
		for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				cublasSetStream(cudas->cuda_handle[gpu],
								cudas->cuda_streams[jdx]);
				cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
				&_alpha,_K.hiddens[idx+1].weights+jdx*red,M,delta_ptr[idx+1],1,
				&_beta,delta_ptr[idx]+jdx*red,1);
				CHK_ERR(train_gemv);
				dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,_K.hiddens[idx].vec+jdx*red,delta_ptr[idx]+jdx*red);
				CHK_ERR(train_dsigmoid);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
			&_alpha,_K.hiddens[idx+1].weights+jdx*red,M,delta_ptr[idx+1],1,
			&_beta,delta_ptr[idx]+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.hiddens[idx].vec+jdx*red,delta_ptr[idx]+jdx*red);
			CHK_ERR(train_dsigmoid);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
		&_alpha,_K.hiddens[idx+1].weights+jdx*red,M,delta_ptr[idx+1],1,
		&_beta,delta_ptr[idx]+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,_K.hiddens[idx].vec+jdx*red,delta_ptr[idx]+jdx*red);
		CHK_ERR(train_dsigmoid);
}else{
/*>>> first GPU[0]*/
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[0],CUBLAS_OP_N,red,N,
			&_alpha,_K.hiddens[idx+1].weights+jdx*red,M,delta_ptr[idx+1],1,
			&_beta,delta_ptr[idx]+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.hiddens[idx].vec+jdx*red,delta_ptr[idx]+jdx*red);
			CHK_ERR(train_dsigmoid);
		}
/*>>> next GPUs but the last one*/
		for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			kx=_K.kerns[gpu];
			/*1- get full delta[idx+1] from GPU[0]*/
			cudaMemcpy(ptr[gpu],delta_ptr[idx+1],M,cudaMemcpyDeviceToDevice);
			/*we don't need to sync (I think)*/
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				/*2- calculate tmp_gpu = delta[idx]*/
				cublasSetStream(cudas->cuda_handle[gpu],
								cudas->cuda_streams[jdx]);
				cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
				&_alpha,_Kx.hiddens[idx+1].weights+jdx*red,M,ptr[gpu],1,
				&_beta,_Kx.tmp_gpu+jdx*red,1);
				CHK_ERR(train_gemv);
				dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,_Kx.hiddens[idx].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
				CHK_ERR(train_dsigmoid);
				/*3- send result to delta[idx] on GPU[0]*/
				cudaMemcpyAsync(delta_ptr[idx]+jdx*red,
					_Kx.tmp_gpu+jdx*red,red,cudaMemcpyDeviceToDevice,
					cudas->cuda_streams[jdx]);
				CHK_ERR(delta_transfer);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[idx+1] from GPU[0]*/
		cudaMemcpy(ptr[gpu],delta_ptr[idx+1],M,cudaMemcpyDeviceToDevice);
		/*no sync needed (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate tmp_gpu = delta[idx]*/
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
			&_alpha,_Kx.hiddens[idx+1].weights+jdx*red,M,ptr[gpu],1,
			&_beta,_Kx.tmp_gpu+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_Kx.hiddens[idx].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
			CHK_ERR(train_dsigmoid);
			/*3- send result to delta[idx] on GPU[0]*/
			cudaMemcpyAsync(delta_ptr[idx]+jdx*red,
				_Kx.tmp_gpu+jdx*red,red,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
		&_alpha,_Kx.hiddens[idx+1].weights+jdx*red,M,ptr[gpu],1,
		&_beta,_Kx.tmp_gpu+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,_Kx.hiddens[idx].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
		CHK_ERR(train_dsigmoid);
		/*send result to delta[idx] on GPU[0]*/
		cudaMemcpyAsync(delta_ptr[idx]+jdx*red,_Kx.tmp_gpu+jdx*red,
			red+rem,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
		for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,M,N,_K.hiddens[idx+1].weights+jdx*red,
					delta_ptr[idx+1],_K.hiddens[idx].vec+jdx*red,
					delta_ptr[idx]+jdx*red);
				CHK_ERR(train_dsigmoid_mul_delta_T);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,M,N,_K.hiddens[idx+1].weights+jdx*red,delta_ptr[idx+1],
				_K.hiddens[idx].vec+jdx*red,delta_ptr[idx]+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,M,N,_K.hiddens[idx+1].weights+jdx*red,delta_ptr[idx+1],
			_K.hiddens[idx].vec+jdx*red,delta_ptr[idx]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
}else{
/*>>> first GPU[0]*/
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,M,N,_K.hiddens[idx+1].weights+jdx*red,delta_ptr[idx+1],
				_K.hiddens[idx].vec+jdx*red,delta_ptr[idx]+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
		}
/*>>> next GPUs but the last one*/
		for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			kx=_K.kerns[gpu];
			/*1- get full delta[idx+1] from GPU[0]*/
			cudaMemcpy(ptr[gpu],delta_ptr[idx+1],M,cudaMemcpyDeviceToDevice);
			/*we don't need to sync (I think)*/
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				/*2- calculate tmp_gpu = delta[idx]*/
				dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,M,N,_Kx.hiddens[idx+1].weights+jdx*red,ptr[gpu],
					 _Kx.hiddens[idx].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
				CHK_ERR(train_dsigmoid_mul_delta_T);
				/*3- send result to delta[idx] on GPU[0]*/
				cudaMemcpyAsync(delta_ptr[idx]+jdx*red,
					_Kx.tmp_gpu+jdx*red,red,cudaMemcpyDeviceToDevice,
					cudas->cuda_streams[jdx]);
				CHK_ERR(delta_transfer);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[idx+1] from GPU[0]*/
		cudaMemcpy(ptr[gpu],delta_ptr[idx+1],M,cudaMemcpyDeviceToDevice);
		/*no sync needed (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate tmp_gpu = delta[idx]*/
			dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,M,N,_Kx.hiddens[idx+1].weights+jdx*red,ptr[gpu],
				 _Kx.hiddens[idx].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
			/*3- send result to delta[idx] on GPU[0]*/
			cudaMemcpyAsync(delta_ptr[idx]+jdx*red,
				_Kx.tmp_gpu+jdx*red,red,cudaMemcpyDeviceToDevice,
				cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,M,N,_Kx.hiddens[idx+1].weights+jdx*red,ptr[gpu],
			_Kx.hiddens[idx].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
		/*send result to delta[idx] on GPU[0]*/
		cudaMemcpyAsync(delta_ptr[idx]+jdx*red,_Kx.tmp_gpu+jdx*red,
			red+rem,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
}
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
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
		for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				cublasSetStream(cudas->cuda_handle[gpu],
								cudas->cuda_streams[jdx]);
				cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
				&_alpha,_K.hiddens[1].weights+jdx*red,M,delta_ptr[1],1,
				&_beta,delta_ptr[0]+jdx*red,1);
				CHK_ERR(train_gemv);
				dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,_K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
				CHK_ERR(train_dsigmoid);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
			&_alpha,_K.hiddens[1].weights+jdx*red,M,delta_ptr[1],1,
			&_beta,delta_ptr[0]+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
			CHK_ERR(train_dsigmoid);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
		&_alpha,_K.hiddens[1].weights+jdx*red,M,delta_ptr[1],1,
		&_beta,delta_ptr[0]+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,_K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
		CHK_ERR(train_dsigmoid);
}else{
/*>>> first GPU[0]*/
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[0],CUBLAS_OP_N,red,N,
			&_alpha,_K.hiddens[1].weights+jdx*red,M,delta_ptr[1],1,
			&_beta,delta_ptr[0]+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
			CHK_ERR(train_dsigmoid);
		}
/*>>> next GPUs but the last one*/
		for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			kx=_K.kerns[gpu];
			/*1- get full delta[1] from GPU[0]*/
			cudaMemcpy(ptr[gpu],delta_ptr[1],M,cudaMemcpyDeviceToDevice);
			/*we don't need to sync (I think)*/
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				/*2- calculate tmp_gpu = delta[idx]*/
				cublasSetStream(cudas->cuda_handle[gpu],
								cudas->cuda_streams[jdx]);
				cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
				&_alpha,_Kx.hiddens[1].weights+jdx*red,M,ptr[gpu],1,
				&_beta,_Kx.tmp_gpu+jdx*red,1);
				CHK_ERR(train_gemv);
				dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,_Kx.hiddens[0].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
				CHK_ERR(train_dsigmoid);
				/*3- send result to delta[0] on GPU[0]*/
				cudaMemcpyAsync(delta_ptr[0]+jdx*red,_Kx.tmp_gpu+jdx*red,
					red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
				CHK_ERR(delta_transfer);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[1] from GPU[0]*/
		cudaMemcpy(ptr[gpu],delta_ptr[1],M,cudaMemcpyDeviceToDevice);
		/*no sync needed (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate tmp_gpu = delta[idx]*/
			cublasSetStream(cudas->cuda_handle[gpu],
							cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red,N,
			&_alpha,_Kx.hiddens[1].weights+jdx*red,M,ptr[gpu],1,
			&_beta,_Kx.tmp_gpu+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,_Kx.hiddens[0].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
			CHK_ERR(train_dsigmoid);
			/*3- send result to delta[0] on GPU[0]*/
			cudaMemcpyAsync(delta_ptr[0]+jdx*red,_Kx.tmp_gpu+jdx*red,
				red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
		&_alpha,_Kx.hiddens[1].weights+jdx*red,M,ptr[gpu],1,
		&_beta,_Kx.tmp_gpu+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,_Kx.hiddens[0].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
		CHK_ERR(train_dsigmoid);
		/*send result to delta[0] on GPU[0]*/
		cudaMemcpyAsync(delta_ptr[0]+jdx*red,_Kx.tmp_gpu+jdx*red,
			red+rem,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
		for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,M,N,_K.hiddens[1].weights+jdx*red,delta_ptr[1],
					_K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
				CHK_ERR(train_dsigmoid_mul_delta_T);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,M,N,_K.hiddens[1].weights+jdx*red,
			delta_ptr[1],_K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,M,N,_K.hiddens[1].weights+jdx*red,delta_ptr[1],
			_K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
}else{
/*>>> first GPU[0]*/
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,M,N,_K.hiddens[1].weights+jdx*red,
			delta_ptr[1],_K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
		}
/*>>> next GPUs but the last one*/
		for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			kx=_K.kerns[gpu];
			/*1- get full delta[1] from GPU[0]*/
			cudaMemcpy(ptr[gpu],delta_ptr[1],M,cudaMemcpyDeviceToDevice);
			/*we don't need to sync (I think)*/
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				/*2- calculate tmp_gpu = delta[idx]*/
				dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(red,M,N,_Kx.hiddens[1].weights+jdx*red,ptr[gpu],
					_Kx.hiddens[0].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
				CHK_ERR(train_dsigmoid_mul_delta_T);
				/*3- send result to delta[0] on GPU[0]*/
				cudaMemcpyAsync(delta_ptr[0]+jdx*red,_Kx.tmp_gpu+jdx*red,
					red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
				CHK_ERR(delta_transfer);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[1] from GPU[0]*/
		cudaMemcpy(ptr[gpu],delta_ptr[1],M,cudaMemcpyDeviceToDevice);
		/*no sync needed (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate tmp_gpu = delta[idx]*/
			dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(red,M,N,_Kx.hiddens[1].weights+jdx*red,ptr[gpu],
				_Kx.hiddens[0].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
			/*3- send result to delta[0] on GPU[0]*/
			cudaMemcpyAsync(delta_ptr[0]+jdx*red,_Kx.tmp_gpu+jdx*red,
				red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(red+rem,M,N,_Kx.hiddens[1].weights+jdx*red,ptr[gpu],
			_Kx.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
		/*send result to delta[0] on GPU[0]*/
		cudaMemcpyAsync(delta_ptr[0]+jdx*red,_Kx.tmp_gpu+jdx*red,
			red+rem,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
}
#endif /*_CUBLAS*/
		for(gpu=0;gpu<cudas->n_gpu;gpu++){
			cudaSetDevice(gpu);
			cudaDeviceSynchronize();
		}
	}
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
	FREE(ptr);
}else{
	for(gpu=1;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		CUDA_FREE(ptr[gpu]);
	}
	FREE(ptr);
}
}
#define LEARN_RATE 0.01
/*------------------------*/
/*+++ back-propagation +++*/
/*------------------------*/
double scuda_snn_train(kernel_ann *kernel,double *train,cudastreams *cudas){
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
	kernel_ann *kx;
	int kdx;
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
	/*allocate delta_ptr*/
	cudaSetDevice(0);/*make sure all allocation happen on gpu[0]*/
	ALLOC(delta_ptr,_K.n_hiddens+1,DOUBLE *);/*HOST*/
	for(idx=0;idx<_K.n_hiddens;idx++)
		CUDA_ALLOC(delta_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
	CUDA_ALLOC(delta_ptr[_K.n_hiddens],_K.n_outputs,DOUBLE);/*DEVICE*/
/*+++ I - FORWARD +++*/
/*>>> in all cases, the FORWARD move should have already be done <<<*/
	Ep=scuda_snn_error(kernel,train,cudas);
//	printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
	scuda_snn_delta(kernel,train,delta_ptr,cudas);
/*+++ III - back propagation +++*/
/*^^^ output NOTE: from here calculation is same as for NN_TYPE_ANN*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef   _CUBLAS
	_alpha=LEARN_RATE;
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_K.hiddens[_K.n_hiddens-1].vec,1,
				delta_ptr[_K.n_hiddens]+jdx*red,1,
				_K.output.weights+jdx*M*red,M);
			CHK_ERR(train_ger);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
			_K.hiddens[_K.n_hiddens-1].vec,1,
			delta_ptr[_K.n_hiddens]+jdx*red,1,
			_K.output.weights+jdx*M*red,M);
		CHK_ERR(train_ger);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
		_K.hiddens[_K.n_hiddens-1].vec,1,delta_ptr[_K.n_hiddens]+jdx*red,
		1,_K.output.weights+jdx*M*red,M);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[0],M,red,&_alpha,
			_K.hiddens[_K.n_hiddens-1].vec,1,
			delta_ptr[_K.n_hiddens]+jdx*red,1,
			_K.output.weights+jdx*M*red,M);
		CHK_ERR(train_ger);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[_K.n_hiddens] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[_Kx.n_hiddens],M,
				   cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec is GPU-local)*/
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_Kx.hiddens[_Kx.n_hiddens-1].vec,1,
				_Kx.tmp_gpu+jdx*red,1,
				_Kx.output.weights+jdx*M*red,M);
			CHK_ERR(train_ger);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.output.weights+jdx*M*red,
				_Kx.output.weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get full delta[_K.n_hiddens] from GPU[0]*/
	cudaMemcpy(_Kx.tmp_gpu,delta_ptr[_K.n_hiddens],M,
			   cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate weights (vec is GPU-local)*/
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
			_Kx.hiddens[_Kx.n_hiddens-1].vec,1,
			_Kx.tmp_gpu+jdx*red,1,
			_Kx.output.weights+jdx*M*red,M);
		CHK_ERR(train_ger);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.output.weights+jdx*M*red,
			_Kx.output.weights+jdx*M*red,
			M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
		_Kx.hiddens[_Kx.n_hiddens-1].vec,1,
		_Kx.tmp_gpu+jdx*red,1,
		_Kx.output.weights+jdx*M*red,M);
	CHK_ERR(train_ger);
	/*3- transfer back weights to GPU[0]*/
	cudaMemcpyAsync(_K.output.weights+jdx*M*red,
		_Kx.output.weights+jdx*M*red,
		M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,
				_K.hiddens[_K.n_hiddens-1].vec,_K.output.weights+jdx*M*red);
			CHK_ERR(train_ger_acc);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,
			_K.hiddens[_K.n_hiddens-1].vec,_K.output.weights+jdx*M*red);
		CHK_ERR(train_ger_acc);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,
		_K.hiddens[_K.n_hiddens-1].vec,_K.output.weights+jdx*M*red);
	CHK_ERR(train_ger_acc);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,
			_K.hiddens[_K.n_hiddens-1].vec,_K.output.weights+jdx*M*red);
		CHK_ERR(train_ger_acc);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[_K.n_hiddens] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[_Kx.n_hiddens],M,
				   cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec is GPU-local)*/
			ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
				_Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.weights+jdx*M*red);
			CHK_ERR(train_ger_acc);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.output.weights+jdx*M*red,
				_Kx.output.weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get full delta[_K.n_hiddens] from GPU[0]*/
	cudaMemcpy(_Kx.tmp_gpu,delta_ptr[_Kx.n_hiddens],M,
			   cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate weights (vec is GPU-local)*/
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
			_Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.weights+jdx*M*red);
		CHK_ERR(train_ger_acc);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.output.weights+jdx*M*red,
			_Kx.output.weights+jdx*M*red,
			M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
		_Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.weights+jdx*M*red);
	CHK_ERR(train_ger_acc);
	/*3- transfer back weights to GPU[0]*/
	cudaMemcpyAsync(_K.output.weights+jdx*M*red,
		_Kx.output.weights+jdx*M*red,
		M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
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
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
		for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				cublasSetStream(cudas->cuda_handle[gpu],
								cudas->cuda_streams[jdx]);
				cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
					_K.hiddens[idx-1].vec,1,delta_ptr[idx]+jdx*red,1,
					_K.hiddens[idx].weights+jdx*M*red,M);
				CHK_ERR(train_ger);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_K.hiddens[idx-1].vec,1,delta_ptr[idx]+jdx*red,1,
				_K.hiddens[idx].weights+jdx*M*red,M);
			CHK_ERR(train_ger);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
			_K.hiddens[idx-1].vec,1,delta_ptr[idx]+jdx*red,1,
			_K.hiddens[idx].weights+jdx*M*red,M);
		CHK_ERR(train_ger);
}else{
/*>>> first GPU[0]*/
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[0],M,red,&_alpha,
			_K.hiddens[idx-1].vec,1,delta_ptr[idx]+jdx*red,1,
			_K.hiddens[idx].weights+jdx*M*red,M);
			CHK_ERR(train_ger);
		}
/*>>> next GPUs but the last one*/
		for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			kx=_K.kerns[gpu];
			/*1- get full delta[idx] from GPU[0]*/
			cudaMemcpy(_Kx.tmp_gpu,delta_ptr[idx],M,cudaMemcpyDeviceToDevice);
			/*we don't need to sync (I think)*/
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				/*2- calculate weights (vec is GPU-local)*/
				cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
				cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
					_Kx.hiddens[idx-1].vec,1,_Kx.tmp_gpu+jdx*red,1,
					_Kx.hiddens[idx].weights+jdx*M*red,M);
				CHK_ERR(train_ger);
				/*3- transfer back weights to GPU[0]*/
				cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
					_Kx.hiddens[idx].weights+jdx*M*red,
					M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
				CHK_ERR(delta_transfer);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[idx] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[idx],M,cudaMemcpyDeviceToDevice);
		/*no sync needed (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec is GPU-local)*/
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_Kx.hiddens[idx-1].vec,1,_Kx.tmp_gpu+jdx*red,1,
				_Kx.hiddens[idx].weights+jdx*M*red,M);
			CHK_ERR(train_ger);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
				_Kx.hiddens[idx].weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
			_Kx.hiddens[idx-1].vec,1,_Kx.tmp_gpu+jdx*red,1,
			_Kx.hiddens[idx].weights+jdx*M*red,M);
		CHK_ERR(train_ger);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
			_Kx.hiddens[idx].weights+jdx*M*red,
			M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
		for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(M,red,LEARN_RATE,delta_ptr[idx]+jdx*red,
					_K.hiddens[idx-1].vec,_K.hiddens[idx].weights+jdx*M*red);
				CHK_ERR(train_ger_acc);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,delta_ptr[idx]+jdx*red,
				_K.hiddens[idx-1].vec,_K.hiddens[idx].weights+jdx*M*red);
			CHK_ERR(train_ger_acc);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,LEARN_RATE,delta_ptr[idx]+jdx*red,
			_K.hiddens[idx-1].vec,_K.hiddens[idx].weights+jdx*M*red);
		CHK_ERR(train_ger_acc);
}else{
/*>>> first GPU[0]*/
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,delta_ptr[idx]+jdx*red,
				_K.hiddens[idx-1].vec,_K.hiddens[idx].weights+jdx*M*red);
			CHK_ERR(train_ger_acc);
		}
/*>>> next GPUs but the last one*/
		for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			kx=_K.kerns[gpu];
			/*1- get full delta[idx] from GPU[0]*/
			cudaMemcpy(_Kx.tmp_gpu,delta_ptr[idx],M,cudaMemcpyDeviceToDevice);
			/*we don't need to sync (I think)*/
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				/*2- calculate weights (vec is GPU-local)*/
				ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
					_Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].weights+jdx*M*red);
				CHK_ERR(train_ger_acc);
				/*3- transfer back weights to GPU[0]*/
				cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
					_Kx.hiddens[idx].weights+jdx*M*red,
					M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
				CHK_ERR(delta_transfer);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[idx] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[idx],M,cudaMemcpyDeviceToDevice);
		/*no sync needed (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec is GPU-local)*/
			ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
				_Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].weights+jdx*M*red);
			CHK_ERR(train_ger_acc);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
				_Kx.hiddens[idx].weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
			_Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].weights+jdx*M*red);
		CHK_ERR(train_ger_acc);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
			_Kx.hiddens[idx].weights+jdx*M*red,
			M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
}
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
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,_K.in,1,
				delta_ptr[0]+jdx*red,1,_K.hiddens[0].weights+jdx*M*red,M);
			CHK_ERR(train_ger);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,_K.in,1,
			delta_ptr[0]+jdx*red,1,_K.hiddens[0].weights+jdx*M*red,M);
		CHK_ERR(train_ger);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,_K.in,1,
		delta_ptr[0]+jdx*red,1,_K.hiddens[0].weights+jdx*M*red,M);
	CHK_ERR(train_ger);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[0],M,red,&_alpha,_K.in,1,
			delta_ptr[0]+jdx*red,1,_K.hiddens[0].weights+jdx*M*red,M);
		CHK_ERR(train_ger);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[0] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[0],M,cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec is GPU-local)*/
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,_Kx.in,1,
				_Kx.tmp_gpu+jdx*red,1,_Kx.hiddens[0].weights+jdx*M*red,M);
			CHK_ERR(train_ger);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
				_Kx.hiddens[0].weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get full delta[idx] from GPU[0]*/
	cudaMemcpy(_Kx.tmp_gpu,delta_ptr[0],M,cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate weights (vec is GPU-local)*/
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,_Kx.in,1,
			_Kx.tmp_gpu+jdx*red,1,_Kx.hiddens[0].weights+jdx*M*red,M);
		CHK_ERR(train_ger);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
			_Kx.hiddens[0].weights+jdx*M*red,
			M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,_Kx.in,1,
		_Kx.tmp_gpu+jdx*red,1,_Kx.hiddens[0].weights+jdx*M*red,M);
	CHK_ERR(train_ger);
	/*3- transfer back weights to GPU[0]*/
	cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
		_Kx.hiddens[0].weights+jdx*M*red,
		M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,delta_ptr[0]+jdx*red,_K.in,
				_K.hiddens[0].weights+jdx*M*red);
			CHK_ERR(train_ger_acc);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,delta_ptr[0]+jdx*red,_K.in,
			_K.hiddens[0].weights+jdx*M*red);
		CHK_ERR(train_ger_acc);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,delta_ptr[0]+jdx*red,_K.in,
		_K.hiddens[0].weights+jdx*M*red);
	CHK_ERR(train_ger_acc);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,delta_ptr[0]+jdx*red,_K.in,
			_K.hiddens[0].weights+jdx*M*red);
		CHK_ERR(train_ger_acc);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[0] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[0],M,cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec is GPU-local)*/
			ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,_Kx.in,
				_Kx.hiddens[0].weights+jdx*M*red);
			CHK_ERR(train_ger_acc);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
				_Kx.hiddens[0].weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get full delta[idx] from GPU[0]*/
	cudaMemcpy(_Kx.tmp_gpu,delta_ptr[0],M,cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate weights (vec is GPU-local)*/
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,_Kx.in,
			_Kx.hiddens[0].weights+jdx*M*red);
		CHK_ERR(train_ger_acc);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
			_Kx.hiddens[0].weights+jdx*M*red,
			M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,_Kx.tmp_gpu+jdx*red,_Kx.in,
		_Kx.hiddens[0].weights+jdx*M*red);
	CHK_ERR(train_ger_acc);
	/*3- transfer back weights to GPU[0]*/
	cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
		_Kx.hiddens[0].weights+jdx*M*red,
		M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
#endif /*_CUBLAS*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*+++ IV - update error NOTE: NN_TYPE_SNN specific +++*/
	/*update kernel*/
	scuda_snn_forward(kernel,cudas);
	Epr=scuda_snn_error(kernel,train,cudas);
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
/*--------------------------------------*/
/*+++ back-propagation with momentum +++*/
/*--------------------------------------*/
double scuda_snn_train_momentum(kernel_ann *kernel,double *train,double moment,
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
#endif /*_CUBLAS*/
	kernel_ann *kx;
	int kdx;
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
	/*allocate delta_ptr*/
	cudaSetDevice(0);/*make sure all allocation happen on gpu[0]*/
	ALLOC(delta_ptr,_K.n_hiddens+1,DOUBLE *);/*HOST*/
	for(idx=0;idx<_K.n_hiddens;idx++)
		CUDA_ALLOC(delta_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
	CUDA_ALLOC(delta_ptr[_K.n_hiddens],_K.n_outputs,DOUBLE);/*DEVICE*/
/*+++ I - FORWARD +++*/
/*>>> in all cases, the FORWARD move should have already be done <<<*/
	Ep=scuda_snn_error(kernel,train,cudas);
///	printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
	scuda_snn_delta(kernel,train,delta_ptr,cudas);
/*+++ III - back propagation +++*/
/*^^^ output NOTE: from here calculation is same as for NN_TYPE_ANN*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef   _CUBLAS
	_alpha=LEARN_RATE;
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_K.hiddens[_K.n_hiddens-1].vec,1,
				delta_ptr[_K.n_hiddens]+jdx*red,1,
			  _K.dw[_K.n_hiddens]+jdx*M*red,M);
			CHK_ERR(moment_ger);
			cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
				_K.dw[_K.n_hiddens]+jdx*M*red,1,
				_K.output.weights+jdx*M*red,1);
			CHK_ERR(moment_axpy);
			cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
				_K.dw[_K.n_hiddens]+jdx*M*red,1);
			CHK_ERR(moment_scal);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
			_K.hiddens[_K.n_hiddens-1].vec,1,
			delta_ptr[_K.n_hiddens]+jdx*red,1,
		  _K.dw[_K.n_hiddens]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
			_K.dw[_K.n_hiddens]+jdx*M*red,1,
			_K.output.weights+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
			_K.dw[_K.n_hiddens]+jdx*M*red,1);
		CHK_ERR(moment_scal);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
		_K.hiddens[_K.n_hiddens-1].vec,1,
		delta_ptr[_K.n_hiddens]+jdx*red,1,
		_K.dw[_K.n_hiddens]+jdx*M*red,M);
	CHK_ERR(moment_ger);
	cublasDaxpy(cudas->cuda_handle[gpu],(red+rem)*M,&_un,
		_K.dw[_K.n_hiddens]+jdx*M*red,1,
		_K.output.weights+jdx*M*red,1);
	CHK_ERR(moment_axpy);
	cublasDscal(cudas->cuda_handle[gpu],(red+rem)*M,&moment,
		_K.dw[_K.n_hiddens]+jdx*M*red,1);
	CHK_ERR(moment_scal);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[0],M,red,&_alpha,
			_K.hiddens[_K.n_hiddens-1].vec,1,
			delta_ptr[_K.n_hiddens]+jdx*red,1,
			_K.dw[_K.n_hiddens]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[0],red*M,&_un,
			_K.dw[_K.n_hiddens]+jdx*M*red,1,
			_K.output.weights+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[0],red*M,&moment,
			_K.dw[_K.n_hiddens]+jdx*M*red,1);
		CHK_ERR(moment_scal);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[_K.n_hiddens] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[_Kx.n_hiddens],M,
				   cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec, dw are GPU-local)*/
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_Kx.hiddens[_Kx.n_hiddens-1].vec,1,
				_Kx.tmp_gpu+jdx*red,1,
				_Kx.dw[_K.n_hiddens]+jdx*M*red,M);
			CHK_ERR(moment_ger);
			cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
				_Kx.dw[_Kx.n_hiddens]+jdx*M*red,1,
				_Kx.output.weights+jdx*M*red,1);
			CHK_ERR(moment_axpy);
			cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
				_Kx.dw[_Kx.n_hiddens]+jdx*M*red,1);
			CHK_ERR(moment_scal);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.output.weights+jdx*M*red,
				_Kx.output.weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		/*PS: The same portion of momentum is always applied to the same GPU*/
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get full delta[_K.n_hiddens] from GPU[0]*/
	cudaMemcpy(_Kx.tmp_gpu,delta_ptr[_K.n_hiddens],M,
			   cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate weights (vec is GPU-local)*/
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
			_Kx.hiddens[_K.n_hiddens-1].vec,1,
			_Kx.tmp_gpu+jdx*red,1,
			_Kx.dw[_Kx.n_hiddens]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
			_Kx.dw[_Kx.n_hiddens]+jdx*M*red,1,
			_Kx.output.weights+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
			_Kx.dw[_K.n_hiddens]+jdx*M*red,1);
		CHK_ERR(moment_scal);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.output.weights+jdx*M*red,
			_Kx.output.weights+jdx*M*red,
			M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
		_Kx.hiddens[_Kx.n_hiddens-1].vec,1,
		_Kx.tmp_gpu+jdx*red,1,
		_Kx.dw[_Kx.n_hiddens]+jdx*M*red,M);
	CHK_ERR(moment_ger);
	cublasDaxpy(cudas->cuda_handle[gpu],(red+rem)*M,&_un,
		_Kx.dw[_Kx.n_hiddens]+jdx*M*red,1,
		_Kx.output.weights+jdx*M*red,1);
	CHK_ERR(moment_axpy);
	cublasDscal(cudas->cuda_handle[gpu],(red+rem)*M,&moment,
		_Kx.dw[_K.n_hiddens]+jdx*M*red,1);
	CHK_ERR(moment_scal);
	/*3- transfer back weights to GPU[0]*/
	cudaMemcpyAsync(_K.output.weights+jdx*M*red,
		_Kx.output.weights+jdx*M*red,
		M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,moment,delta_ptr[_K.n_hiddens]+jdx*red,
				_K.hiddens[_K.n_hiddens-1].vec,
				_K.dw[_K.n_hiddens]+jdx*M*red,_K.output.weights+jdx*M*red);
			CHK_ERR(moment_ger_dw_acc);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,moment,delta_ptr[_K.n_hiddens]+jdx*red,
			_K.hiddens[_K.n_hiddens-1].vec,
			_K.dw[_K.n_hiddens]+jdx*M*red,_K.output.weights+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,moment,delta_ptr[_K.n_hiddens]+jdx*red,
		_K.hiddens[_K.n_hiddens-1].vec,
		_K.dw[_K.n_hiddens]+jdx*M*red,_K.output.weights+jdx*M*red);
	CHK_ERR(moment_ger_dw_acc);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,moment,delta_ptr[_K.n_hiddens]+jdx*red,
			_K.hiddens[_K.n_hiddens-1].vec,
			_K.dw[_K.n_hiddens]+jdx*M*red,_K.output.weights+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[_K.n_hiddens] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[_Kx.n_hiddens],M,
				   cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec, dw are GPU-local)*/
			ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
				_Kx.hiddens[_Kx.n_hiddens-1].vec,
				_Kx.dw[_Kx.n_hiddens]+jdx*M*red,_Kx.output.weights+jdx*M*red);
			CHK_ERR(moment_ger_dw_acc);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.output.weights+jdx*M*red,
				_Kx.output.weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		/*PS: The same portion of momentum is always applied to the same GPU*/
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get full delta[_K.n_hiddens] from GPU[0]*/
	cudaMemcpy(_Kx.tmp_gpu,delta_ptr[_K.n_hiddens],M,
			   cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate weights (vec is GPU-local)*/
		ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
			_Kx.hiddens[_Kx.n_hiddens-1].vec,
			_Kx.dw[_Kx.n_hiddens]+jdx*M*red,_Kx.output.weights+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.output.weights+jdx*M*red,
			_Kx.output.weights+jdx*M*red,
			M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
		_Kx.hiddens[_Kx.n_hiddens-1].vec,
		_Kx.dw[_Kx.n_hiddens]+jdx*M*red,_Kx.output.weights+jdx*M*red);
	CHK_ERR(moment_ger_dw_acc);
	/*3- transfer back weights to GPU[0]*/
	cudaMemcpyAsync(_K.output.weights+jdx*M*red,
		_Kx.output.weights+jdx*M*red,
		M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
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
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
		for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				cublasSetStream(cudas->cuda_handle[gpu],
								cudas->cuda_streams[jdx]);
				cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
					_K.hiddens[idx-1].vec,1,
					delta_ptr[idx]+jdx*red,1,
					_K.dw[idx]+jdx*M*red,M);
				CHK_ERR(moment_ger);
				cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
					_K.dw[idx]+jdx*M*red,1,
					_K.hiddens[idx].weights+jdx*M*red,1);
				CHK_ERR(moment_axpy);
				cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
					_K.dw[idx]+jdx*M*red,1);
				CHK_ERR(moment_scal);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],
							cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_K.hiddens[idx-1].vec,1,
				delta_ptr[idx]+jdx*red,1,
				_K.dw[idx]+jdx*M*red,M);
			CHK_ERR(moment_ger);
			cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
				_K.dw[idx]+jdx*M*red,1,
				_K.hiddens[idx].weights+jdx*M*red,1);
			CHK_ERR(moment_axpy);
				cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
				_K.dw[idx]+jdx*M*red,1);
			CHK_ERR(moment_scal);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
			_K.hiddens[idx-1].vec,1,
			delta_ptr[idx]+jdx*red,1,
			_K.dw[idx]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[gpu],(red+rem)*M,&_un,
			_K.dw[idx]+jdx*M*red,1,
			_K.hiddens[idx].weights+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[gpu],(red+rem)*M,&moment,
			_K.dw[idx]+jdx*M*red,1);
		CHK_ERR(moment_scal);
}else{
/*>>> first GPU[0]*/
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[0],M,red,&_alpha,
				_K.hiddens[idx-1].vec,1,
				delta_ptr[idx]+jdx*red,1,
				_K.dw[idx]+jdx*M*red,M);
			CHK_ERR(moment_ger);
			cublasDaxpy(cudas->cuda_handle[0],red*M,&_un,
				_K.dw[idx]+jdx*M*red,1,
				_K.hiddens[idx].weights+jdx*M*red,1);
			CHK_ERR(moment_axpy);
			cublasDscal(cudas->cuda_handle[0],red*M,&moment,
				_K.dw[idx]+jdx*M*red,1);
			CHK_ERR(moment_scal);
		}
/*>>> next GPUs but the last one*/
		for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			kx=_K.kerns[gpu];
			/*1- get full delta[idx] from GPU[0]*/
			cudaMemcpy(_Kx.tmp_gpu,delta_ptr[idx],M,cudaMemcpyDeviceToDevice);
			/*we don't need to sync (I think)*/
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				/*2- calculate weights (vec, dw are GPU-local)*/
				cublasSetStream(cudas->cuda_handle[gpu],
								cudas->cuda_streams[jdx]);
				cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
					_Kx.hiddens[idx-1].vec,1,
					_Kx.tmp_gpu+jdx*red,1,
					_Kx.dw[idx]+jdx*M*red,M);
				CHK_ERR(moment_ger);
				cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
					_Kx.dw[idx]+jdx*M*red,1,
					_Kx.hiddens[idx].weights+jdx*M*red,1);
				CHK_ERR(moment_axpy);
				cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
					_Kx.dw[idx]+jdx*M*red,1);
				CHK_ERR(moment_scal);
				/*3- transfer back weights to GPU[0]*/
				cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
					_Kx.hiddens[idx].weights+jdx*M*red,
					M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
				CHK_ERR(delta_transfer);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[idx] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[idx],M,cudaMemcpyDeviceToDevice);
		/*no sync needed (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec is GPU-local)*/
			cublasSetStream(cudas->cuda_handle[gpu],
							cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_Kx.hiddens[idx-1].vec,1,
				_Kx.tmp_gpu+jdx*red,1,
				_Kx.dw[idx]+jdx*M*red,M);
			CHK_ERR(moment_ger);
			cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
				_Kx.dw[idx]+jdx*M*red,1,
				_Kx.hiddens[idx].weights+jdx*M*red,1);
			CHK_ERR(moment_axpy);
			cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
				_Kx.dw[idx]+jdx*M*red,1);
			CHK_ERR(moment_scal);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
				_Kx.hiddens[idx].weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
			_Kx.hiddens[idx-1].vec,1,
			_Kx.tmp_gpu+jdx*red,1,
			_Kx.dw[idx]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[gpu],(red+rem)*M,&_un,
			_Kx.dw[idx]+jdx*M*red,1,
			_Kx.hiddens[idx].weights+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[gpu],(red+rem)*M,&moment,
			_Kx.dw[idx]+jdx*M*red,1);
		CHK_ERR(moment_scal);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
			_Kx.hiddens[idx].weights+jdx*M*red,
			M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
		for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(M,red,LEARN_RATE,moment,delta_ptr[idx]+jdx*red,
					_K.hiddens[idx-1].vec,_K.dw[idx]+jdx*M*red,
					_K.hiddens[idx].weights+jdx*M*red);
				CHK_ERR(moment_ger_dw_acc);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,moment,delta_ptr[idx]+jdx*red,
				_K.hiddens[idx-1].vec,_K.dw[idx]+jdx*M*red,
				_K.hiddens[idx].weights+jdx*M*red);
			CHK_ERR(moment_ger_dw_acc);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,LEARN_RATE,moment,delta_ptr[idx]+jdx*red,
			_K.hiddens[idx-1].vec,_K.dw[idx]+jdx*M*red,
			_K.hiddens[idx].weights+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
}else{
/*>>> first GPU[0]*/
		cudaSetDevice(0);
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,moment,delta_ptr[idx]+jdx*red,
				_K.hiddens[idx-1].vec,_K.dw[idx]+jdx*M*red,
				_K.hiddens[idx].weights+jdx*M*red);
			CHK_ERR(moment_ger_dw_acc);
		}
/*>>> next GPUs but the last one*/
		for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
			cudaSetDevice(gpu);
			kx=_K.kerns[gpu];
			/*1- get full delta[idx] from GPU[0]*/
			cudaMemcpy(_Kx.tmp_gpu,delta_ptr[idx],M,cudaMemcpyDeviceToDevice);
			/*we don't need to sync (I think)*/
			for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
				jdx=kdx+gpu*(cudas->cuda_n_streams);
				/*2- calculate weights (vec, dw are GPU-local)*/
				ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
					(M,red,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
					_Kx.hiddens[idx-1].vec,_Kx.dw[idx]+jdx*M*red,
					_Kx.hiddens[idx].weights+jdx*M*red);
				CHK_ERR(moment_ger_dw_acc);
				/*3- transfer back weights to GPU[0]*/
				cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
					_Kx.hiddens[idx].weights+jdx*M*red,
					M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
				CHK_ERR(delta_transfer);
			}
		}
/*>>> last GPU*/
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[idx] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[idx],M,cudaMemcpyDeviceToDevice);
		/*no sync needed (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec is GPU-local)*/
			ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
				_Kx.hiddens[idx-1].vec,_Kx.dw[idx]+jdx*M*red,
				_Kx.hiddens[idx].weights+jdx*M*red);
			CHK_ERR(moment_ger_dw_acc);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
				_Kx.hiddens[idx].weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
/*>>> last stream*/
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
			_K.hiddens[idx-1].vec,_K.dw[idx]+jdx*M*red,
			_K.hiddens[idx].weights+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[idx].weights+jdx*M*red,
			_Kx.hiddens[idx].weights+jdx*M*red,
			M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
}
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
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_K.in,1,
				delta_ptr[0]+jdx*red,1,
				_K.dw[0]+jdx*M*red,M);
			CHK_ERR(moment_ger);
			cublasDaxpy(cudas->cuda_handle[gpu],
				red*M,&_un,_K.dw[0]+jdx*M*red,1,
				_K.hiddens[0].weights+jdx*M*red,1);
			CHK_ERR(moment_axpy);
			cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
				_K.dw[0]+jdx*M*red,1);
			CHK_ERR(moment_scal);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
			_K.in,1,
			delta_ptr[0]+jdx*red,1,
			_K.dw[0]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[gpu],
			red*M,&_un,_K.dw[0]+jdx*M*red,1,
			_K.hiddens[0].weights+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
			_K.dw[0]+jdx*M*red,1);
		CHK_ERR(moment_scal);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
		_K.in,1,
		delta_ptr[0]+jdx*red,1,
		_K.dw[0]+jdx*M*red,M);
	CHK_ERR(moment_ger);
	cublasDaxpy(cudas->cuda_handle[gpu],(red+rem)*M,&_un,
		_K.dw[0]+jdx*M*red,1,
		_K.hiddens[0].weights+jdx*M*red,1);
	CHK_ERR(moment_axpy);
	cublasDscal(cudas->cuda_handle[gpu],(red+rem)*M,&moment,
		_K.dw[0]+jdx*M*red,1);
	CHK_ERR(moment_scal);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[0],M,red,&_alpha,
			_K.in,1,
			delta_ptr[0]+jdx*red,1,
			_K.dw[0]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[0],red*M,&_un,
			_K.dw[0]+jdx*M*red,1,
			_K.hiddens[0].weights+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[0],red*M,&moment,
			_K.dw[0]+jdx*M*red,1);
		CHK_ERR(moment_scal);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[0] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[0],M,cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec, dw are GPU-local)*/
			cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
				_Kx.in,1,
				_Kx.tmp_gpu+jdx*red,1,
				_Kx.dw[0]+jdx*M*red,M);
			CHK_ERR(moment_ger);
			cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
				_Kx.dw[0]+jdx*M*red,1,
				_Kx.hiddens[0].weights+jdx*M*red,1);
			CHK_ERR(moment_axpy);
			cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
				_Kx.dw[0]+jdx*M*red,1);
			CHK_ERR(moment_scal);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
				_Kx.hiddens[0].weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get full delta[idx] from GPU[0]*/
	cudaMemcpy(_Kx.tmp_gpu,delta_ptr[idx],M,cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate weights (vec is GPU-local)*/
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
			_Kx.in,1,
			_Kx.tmp_gpu+jdx*red,1,
			_Kx.dw[0]+jdx*M*red,M);
		CHK_ERR(moment_ger);
		cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,
			_Kx.dw[0]+jdx*M*red,1,
			_Kx.hiddens[0].weights+jdx*M*red,1);
		CHK_ERR(moment_axpy);
		cublasDscal(cudas->cuda_handle[gpu],red*M,&moment,
			_Kx.dw[0]+jdx*M*red,1);
		CHK_ERR(moment_scal);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
			_Kx.hiddens[0].weights+jdx*M*red,
			M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
		_Kx.in,1,
		_Kx.tmp_gpu+jdx*red,1,
		_Kx.dw[0]+jdx*M*red,M);
	CHK_ERR(moment_ger);
	cublasDaxpy(cudas->cuda_handle[gpu],(red+rem)*M,&_un,
		_Kx.dw[0]+jdx*M*red,1,
		_Kx.hiddens[0].weights+jdx*M*red,1);
	CHK_ERR(moment_axpy);
	cublasDscal(cudas->cuda_handle[gpu],(red+rem)*M,&moment,
		_Kx.dw[0]+jdx*M*red,1);
	CHK_ERR(moment_scal);
	/*3- transfer back weights to GPU[0]*/
	cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
		_Kx.hiddens[0].weights+jdx*M*red,
		M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
	for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,moment,delta_ptr[0]+jdx*red,
				_K.in,_K.dw[0]+jdx*M*red,_K.hiddens[0].weights+jdx*M*red);
			CHK_ERR(moment_ger_dw_acc);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,moment,delta_ptr[0]+jdx*red,
			_K.in,_K.dw[0]+jdx*M*red,_K.hiddens[0].weights+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,moment,delta_ptr[0]+jdx*red,
		_K.in,_K.dw[0]+jdx*M*red,_K.hiddens[0].weights+jdx*M*red);
	CHK_ERR(moment_ger_dw_acc);
}else{
/*>>> first GPU[0]*/
	cudaSetDevice(0);
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,moment,delta_ptr[0]+jdx*red,
			_K.in,_K.dw[0]+jdx*M*red,_K.hiddens[0].weights+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
	}
/*>>> next GPUs but the last one*/
	for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
		cudaSetDevice(gpu);
		kx=_K.kerns[gpu];
		/*1- get full delta[0] from GPU[0]*/
		cudaMemcpy(_Kx.tmp_gpu,delta_ptr[0],M,cudaMemcpyDeviceToDevice);
		/*we don't need to sync (I think)*/
		for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
			jdx=kdx+gpu*(cudas->cuda_n_streams);
			/*2- calculate weights (vec, dw are GPU-local)*/
			ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
				_Kx.in,_Kx.dw[0]+jdx*M*red,_Kx.hiddens[0].weights+jdx*M*red);
			CHK_ERR(moment_ger_dw_acc);
			/*3- transfer back weights to GPU[0]*/
			cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
				_Kx.hiddens[0].weights+jdx*M*red,
				M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
			CHK_ERR(delta_transfer);
		}
	}
/*>>> last GPU*/
	cudaSetDevice(gpu);
	kx=_K.kerns[gpu];
	/*1- get full delta[idx] from GPU[0]*/
	cudaMemcpy(_Kx.tmp_gpu,delta_ptr[idx],M,cudaMemcpyDeviceToDevice);
	/*no sync needed (I think)*/
	for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
		jdx=kdx+gpu*(cudas->cuda_n_streams);
		/*2- calculate weights (vec is GPU-local)*/
		ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
			_Kx.in,_Kx.dw[0]+jdx*M*red,_Kx.hiddens[0].weights+jdx*M*red);
		CHK_ERR(moment_ger_dw_acc);
		/*3- transfer back weights to GPU[0]*/
		cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
			_Kx.hiddens[0].weights+jdx*M*red,
			M*red,cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
		CHK_ERR(delta_transfer);
	}
/*>>> last stream*/
	jdx=kdx+gpu*(cudas->cuda_n_streams);
	ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
		 _Kx.in,_Kx.dw[0]+jdx*M*red,_Kx.hiddens[0].weights+jdx*M*red);
	CHK_ERR(moment_ger_dw_acc);
	/*3- transfer back weights to GPU[0]*/
	cudaMemcpyAsync(_K.hiddens[0].weights+jdx*M*red,
		_Kx.hiddens[0].weights+jdx*M*red,
		M*(red+rem),cudaMemcpyDeviceToDevice,cudas->cuda_streams[jdx]);
	CHK_ERR(delta_transfer);
}
#endif /*_CUBLAS*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
/*+++ IV - update error NOTE: NN_TYPE_SNN specific +++*/
	/*update kernel*/
	scuda_snn_forward(kernel,cudas);
	Epr=scuda_snn_error(kernel,train,cudas);
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
