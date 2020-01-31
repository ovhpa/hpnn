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
	if(i+blockDim.x < n) mySum += -1.0*train[i+blockDim.x]*log(out[i+blockDim.x]+TINY);
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
/*-----------------------------*/
/*+++ forward kernel update +++*/
/*-----------------------------*/
void scuda_snn_forward(_kernel *kernel,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem,gpu;
        int total_s;
#ifdef _CUBLAS
        double _alpha=1.0;
        double _beta =0.0;
#endif
	double dv;
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
/*+++ I - input +++*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef   _CUBLAS
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
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
	cudaSetDevice(gpu);
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
		cudaSetDevice(gpu);
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.hiddens[0].cuda_w+jdx*M*red,_K.cuda_in,
				_K.hiddens[0].cuda_v+jdx*red);
		CHK_ERR(fw_mv_acc);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_K.hiddens[0].cuda_w+jdx*M*red,_K.cuda_in,
			_K.hiddens[0].cuda_v+jdx*red);
	CHK_ERR(fw_mv_acc);
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
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle[gpu],
			CUBLAS_OP_T,M,red,&_alpha,_K.output.cuda_w+jdx*M*red,M,
			_K.hiddens[_K.n_hiddens-1].cuda_v,1,
			&_beta,_K.output.cuda_v+jdx*red,1);
		CHK_ERR(fw_gemv);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle[gpu],
		CUBLAS_OP_T,M,red+rem,&_alpha,_K.output.cuda_w+jdx*M*red,M,
		_K.hiddens[_K.n_hiddens-1].cuda_v,1,&_beta,_K.output.cuda_v+jdx*red,1);
	CHK_ERR(fw_gemv);
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
	/*TODO: optimize this one on multi-gpu?*/
	cudaSetDevice(0);/*only on master GPU*/
	softmax_acc<<<_KG(N),sizeof(double)*2*(_TPB)>>>(N,_K.tmp_gpu,_K.output.cuda_v);
	CHK_ERR(fw_softmax_acc);
	/*SOFTMAX: calculate dv*/
	CUDA_G2C_CP(&dv,&(_K.tmp_gpu[0]),1,double);
	dv=1.0/(dv+TINY);
	/*SOFTMAX: calculate output*/
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		cublasDscal(cudas->cuda_handle[gpu],red,&dv,_K.output.cuda_v+jdx*red,1);
		CHK_ERR(fw_scal);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	cublasDscal(cudas->cuda_handle[gpu],red+rem,&dv,_K.output.cuda_v+jdx*red,1);
	CHK_ERR(fw_scal);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		fw_s_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.output.cuda_w+jdx*M*red,_K.hiddens[_K.n_hiddens-1].cuda_v,
			_K.output.cuda_v+jdx*red);
		CHK_ERR(fw_s_acc);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	fw_s_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_K.output.cuda_w+jdx*M*red,_K.hiddens[_K.n_hiddens-1].cuda_v,
		_K.output.cuda_v+jdx*red);
	CHK_ERR(fw_s_acc);
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
	/*TODO: optimize this one on multi-gpu?*/
	cudaSetDevice(0);/*only on master GPU*/
	softmax_acc<<<_KG(N),sizeof(double)*2*(_TPB)>>>(N,_K.tmp_gpu,_K.output.cuda_v);
	CHK_ERR(fw_softmax_acc);
	/*SOFTMAX: calculate dv*/
	CUDA_G2C_CP(&dv,&(_K.tmp_gpu[0]),1,double);
	dv+=TINY;
	/*SOFTMAX: calculate output*/
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		fw_scal<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,dv,_K.output.cuda_v+jdx*red);
		CHK_ERR(fw_scal);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	fw_scal<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,dv,_K.output.cuda_v+jdx*red);
	CHK_ERR(fw_scal);
#endif /*_CUBLAS*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
}
/*--------------------------------*/
/*+++ Calculate Training Error +++*/
/*--------------------------------*/
double scuda_snn_error(_kernel *kernel,double *train,cudastreams *cudas){
	int     jdx;
	int   N,red;
	int rem,gpu;
	int total_s;
	double dEp=0.;
	total_s=cudas->cuda_n_streams*cudas->n_gpu;
	N=_K.n_outputs;
	red=N/total_s;
	rem=N%total_s;
#ifdef _CUBLAS
	for(jdx=0;jdx<total_s-1;jdx++){
		gpu=jdx/cudas->cuda_n_streams;/*gpu number*/
		cudaSetDevice(gpu);
		cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
		amb_smax<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.tmp_gpu+jdx*red,train+jdx*red,_K.output.cuda_v+jdx*red);
		CHK_ERR(err_amb_smax);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
	amb_smax<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_K.tmp_gpu+jdx*red,train+jdx*red,_K.output.cuda_v+jdx*red);
	CHK_ERR(err_amb_smax);
	/*get dEp*/
	for(gpu=0;gpu<cudas->n_gpu;gpu++){
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}
	/*Dasum only on gpu[0]*/
	cudaSetDevice(0);
	cublasSetStream(cudas->cuda_handle[0],NULL);
	cublasDasum(cudas->cuda_handle[gpu],N,_K.tmp_gpu,1,&dEp);
	CHK_ERR(err_asum);
#else /*_CUBLAS*/
	/*TODO: optimize this one on multi-gpu?*/
	cudaSetDevice(0);/*only on master GPU*/
	amb_smax_acc<<<_KG(_K.n_outputs),sizeof(double)*2*(_TPB)>>>
		(_K.n_outputs,_K.tmp_gpu,train,_K.output.cuda_v);
	CHK_ERR(err_amb_smax_acc);
	CUDA_G2C_CP(&dEp,&(_K.tmp_gpu[0]),1,double);
	CHK_ERR(err_g2c_cp);
#endif /*_CUBLAS*/
	cudaDeviceSynchronize();/*only gpu[0]*/
	dEp/=((double)N);
	return dEp;
}
/*------------------------*/
/*+++ Calculate deltas +++*/
/*------------------------*/
void scuda_snn_delta(_kernel *kernel,double *train,double **delta_ptr,cudastreams *cudas){
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
		dsmax_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,train+jdx*red,_K.output.cuda_v+jdx*red,
			delta_ptr[_K.n_hiddens]+jdx*red);
		CHK_ERR(train_dsmax_dif);
	}
	gpu=total_s/cudas->cuda_n_streams;/*last gpu and stream*/
	cudaSetDevice(gpu);
	dsmax_diff<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,train+jdx*red,_K.output.cuda_v+jdx*red,
		delta_ptr[_K.n_hiddens]+jdx*red);
	CHK_ERR(train_dsmax_dif);
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
				cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
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
double scuda_snn_train(_kernel *kernel,double *train,cudastreams *cudas){
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
	Ep=scuda_snn_error(kernel,train,cudas);
//	printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
	scuda_snn_delta(kernel,train,delta_ptr,cudas);
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
		red=N/total_s;
		rem=N%total_s;
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
double scuda_snn_train_momentum(_kernel *kernel,double *train,double moment,cudastreams *cudas){
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
	Ep=scuda_snn_error(kernel,train,cudas);
///	printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
	scuda_snn_delta(kernel,train,delta_ptr,cudas);
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
		cublasDaxpy(cudas->cuda_handle[gpu],red*M,&_un,_K.cuda_dw[0]+jdx*M*red,1,
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
	cublasDaxpy(cudas->cuda_handle[gpu],(red+rem)*M,&_un,_K.cuda_dw[0]+jdx*M*red,1,
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
