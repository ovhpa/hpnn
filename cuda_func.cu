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
	if(i+blockDim.x < n) mySum += (a[i+blockDim.x]-b[i+blockDim.x])*(a[i+blockDim.x]-b[i+blockDim.x]);
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
void dsigmoid_mul_delta_T(int red,int m,int n, double *w,double *d,double *h,double *res){
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
/*-----------------*/
/* The C interface */
/*-----------------*/
extern "C"{
#define _K (*kernel)
/*+++ forward update +++*/
void scuda_ann_forward_cublas(_kernel *kernel,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem;
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
/*+++ I - output +++*/
	N=_K.hiddens[0].n_neurons;
	M=_K.hiddens[0].n_inputs;
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle,
			CUBLAS_OP_T,M,red,&_alpha,_K.hiddens[0].cuda_w+jdx*M*red,M,
			_K.cuda_in,1,&_beta,_K.hiddens[0].cuda_v+jdx*red,1);
		CHK_ERR(fw_gemv);
		sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,_K.hiddens[0].cuda_v+jdx*red);
		CHK_ERR(fw_sigmoid);
	}
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle,
		CUBLAS_OP_T,M,red+rem,&_alpha,_K.hiddens[0].cuda_w+jdx*M*red,M,
		_K.cuda_in,1,&_beta,_K.hiddens[0].cuda_v+jdx*red,1);
	CHK_ERR(fw_gemv);
	sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,_K.hiddens[0].cuda_v+jdx*red);
	CHK_ERR(fw_sigmoid);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.hiddens[0].cuda_w+jdx*M*red,_K.cuda_in,_K.hiddens[0].cuda_v+jdx*red);
		CHK_ERR(fw_mv_acc);
	}
	fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_K.hiddens[0].cuda_w+jdx*M*red,_K.cuda_in,_K.hiddens[0].cuda_v+jdx*red);
	CHK_ERR(fw_mv_acc);
#endif /*_CUBLAS*/
	cudaDeviceSynchronize();/*get all stream at this point*/
/*+++ II - hidden(s) +++*/
	for(idx=1;idx<_K.n_hiddens;idx++){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		red=N/cudas->cuda_n_streams;
		rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle,
				CUBLAS_OP_T,M,red,&_alpha,_K.hiddens[idx].cuda_w+jdx*M*red,M,
				_K.hiddens[idx-1].cuda_v,1,&_beta,_K.hiddens[idx].cuda_v+jdx*red,1);
			CHK_ERR(fw_gemv);
			sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,_K.hiddens[idx].cuda_v+jdx*red);
			CHK_ERR(fw_sigmoid);
		}
		/*launch the last kernel*/
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle,
			CUBLAS_OP_T,M,red+rem,&_alpha,_K.hiddens[idx].cuda_w+jdx*M*red,M,
			_K.hiddens[idx-1].cuda_v,1,&_beta,_K.hiddens[idx].cuda_v+jdx*red,1);
		CHK_ERR(cublas_1);
		sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,_K.hiddens[idx].cuda_v+jdx*red);
		CHK_ERR(kernel_1);
#else  /*_CUBLAS*/
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,_K.hiddens[idx].cuda_w+jdx*M*red,_K.hiddens[idx-1].cuda_v,_K.hiddens[idx].cuda_v+jdx*red);
			CHK_ERR(fw_mv_acc);
		}
		fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,_K.hiddens[idx].cuda_w+jdx*M*red,_K.hiddens[idx-1].cuda_v,_K.hiddens[idx].cuda_v+jdx*red);
		CHK_ERR(fw_mv_acc);
#endif /*_CUBLAS*/
		cudaDeviceSynchronize();
	}
/*+++ III - output +++*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle,
			CUBLAS_OP_T,M,red,&_alpha,_K.output.cuda_w+jdx*M*red,M,
			_K.hiddens[_K.n_hiddens-1].cuda_v,1,&_beta,_K.output.cuda_v+jdx*red,1);
		CHK_ERR(fw_gemv);
		sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,_K.output.cuda_v+jdx*red);
		CHK_ERR(fw_sigmoid);
	}
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle,
		CUBLAS_OP_T,M,red+rem,&_alpha,_K.output.cuda_w+jdx*M*red,M,
		_K.hiddens[_K.n_hiddens-1].cuda_v,1,&_beta,_K.output.cuda_v+jdx*red,1);
	CHK_ERR(fw_gemv);
	sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,_K.output.cuda_v+jdx*red);
	CHK_ERR(fw_sigmoid);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.output.cuda_w+jdx*M*red,_K.hiddens[_K.n_hiddens-1].cuda_v,_K.output.cuda_v+jdx*red);
		CHK_ERR(fw_mv_acc);
	}
	fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,_K.output.cuda_w+jdx*M*red,_K.hiddens[_K.n_hiddens-1].cuda_v,_K.output.cuda_v+jdx*red);
	CHK_ERR(fw_mv_acc);
#endif /*_CUBLAS*/
	cudaDeviceSynchronize();
#ifdef _TIMING
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time,start,stop);
printf("scuda_ann_forward_cublas: streams = %i time = %f\n",cudas->cuda_n_streams,time);
#endif
}
/*update error*/
double scuda_ann_error(_kernel *kernel,double *train,cudastreams *cudas){
	double dEp=0.;
#ifdef   _CUBLAS
	amb<<<_KG(_K.n_outputs)>>>(_K.n_outputs,_K.tmp_gpu,train,_K.output.cuda_v);
	CHK_ERR(err_amb);
	cublasSetStream(cudas->cuda_handle,NULL);
	cublasDasum(cudas->cuda_handle,_K.n_outputs,_K.tmp_gpu,1,&dEp);
	CHK_ERR(err_asum);
#else  /*_CUBLAS*/
	amb_acc<<<_KG(_K.n_outputs),sizeof(double)*2*(_TPB)>>>(_K.n_outputs,_K.tmp_gpu,train,_K.output.cuda_v);
//	amb_acc<<<1,_TPB/2,sizeof(double)*(_TPB)>>>(_K.n_outputs,_K.tmp_gpu,train,_K.output.cuda_v);
	CHK_ERR(err_amb_acc);
	CUDA_G2C_CP(&dEp,&(_K.tmp_gpu[0]),1,double);
	CHK_ERR(err_g2c_cp);
#endif /*_CUBLAS*/
	dEp*=0.5;
	return dEp;
}
#define LEARN_RATE 0.01
double scuda_ann_train_cublas(_kernel *kernel,double *train,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int rem;
	double **delta_ptr;
	double Ep =0.;
	double Epr=0.;
	/**/
	double _alpha=1.0;
	double _beta =0.0;
	/*allocate delta_ptr*/
	ALLOC(delta_ptr,_K.n_hiddens+1,DOUBLE *);/*HOST*/
	for(idx=0;idx<_K.n_hiddens;idx++) CUDA_ALLOC(delta_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
	CUDA_ALLOC(delta_ptr[_K.n_hiddens],_K.n_outputs,DOUBLE);/*DEVICE*/
/*+++ I - FORWARD +++*/
/*>>> in all cases, the FORWARD move should have already be done <<<*/
//	scuda_ann_forward_cublas(kernel,cudas);
	Ep=scuda_ann_error(kernel,train,cudas);
//	printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
/*^^^ output*/
	N=_K.n_outputs;
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		dsigmoid_mul_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,train+jdx*red,_K.output.cuda_v+jdx*red,delta_ptr[_K.n_hiddens]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_dif);
	}
	dsigmoid_mul_diff<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,train+jdx*red,_K.output.cuda_v+jdx*red,delta_ptr[_K.n_hiddens]+jdx*red);
	CHK_ERR(train_dsigmoid_mul_dif);
/*^^^ output to hidden*/
	/*distribution over M due to transposed operations*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=M/cudas->cuda_n_streams;
	rem=M%cudas->cuda_n_streams;
#ifdef   _CUBLAS
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red,N,
		&_alpha,_K.output.cuda_w+jdx*red,M,delta_ptr[_K.n_hiddens],1,
		&_beta,delta_ptr[_K.n_hiddens-1]+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(red,_K.hiddens[_K.n_hiddens-1].cuda_v+jdx*red,delta_ptr[_K.n_hiddens-1]+jdx*red);
		CHK_ERR(train_dsigmoid);
	}
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
	cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red+rem,N,
	&_alpha,_K.output.cuda_w+jdx*red,M,delta_ptr[_K.n_hiddens],1,
	&_beta,delta_ptr[_K.n_hiddens-1]+jdx*red,1);
	CHK_ERR(train_gemv);
	dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
		(red+rem,_K.hiddens[_K.n_hiddens-1].cuda_v+jdx*red,delta_ptr[_K.n_hiddens-1]+jdx*red);
	CHK_ERR(train_dsigmoid);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,M,N,_K.output.cuda_w+jdx*red,
		delta_ptr[_K.n_hiddens],_K.hiddens[_K.n_hiddens-1].cuda_v+jdx*red,delta_ptr[_K.n_hiddens-1]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
	}
	dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,M,N,_K.output.cuda_w+jdx*red,
	delta_ptr[_K.n_hiddens],_K.hiddens[_K.n_hiddens-1].cuda_v+jdx*red,delta_ptr[_K.n_hiddens-1]+jdx*red);
	CHK_ERR(train_dsigmoid_mul_delta_T);
#endif /*_CUBLAS*/
	cudaDeviceSynchronize();
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
				cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red,N,
				&_alpha,_K.hiddens[idx+1].cuda_w+jdx*red,M,delta_ptr[idx+1],1,
				&_beta,delta_ptr[idx]+jdx*red,1);
				CHK_ERR(train_gemv);
				dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,_K.hiddens[idx].cuda_v+jdx*red,delta_ptr[idx]+jdx*red);
				CHK_ERR(train_dsigmoid);
			}
			cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red+rem,N,
			&_alpha,_K.hiddens[idx+1].cuda_w+jdx*red,M,delta_ptr[idx+1],1,
			&_beta,delta_ptr[idx]+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,_K.hiddens[idx].cuda_v+jdx*red,delta_ptr[idx]+jdx*red);
			CHK_ERR(train_dsigmoid);
#else  /*_CUBLAS*/
			for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
				dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,M,N,_K.hiddens[idx+1].cuda_w+jdx*red,
				delta_ptr[idx+1],_K.hiddens[idx].cuda_v+jdx*red,delta_ptr[idx]+jdx*red);
				CHK_ERR(train_dsigmoid_mul_delta_T);
			}
			dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,M,N,_K.hiddens[idx+1].cuda_w+jdx*red,
			delta_ptr[idx+1],_K.hiddens[idx].cuda_v+jdx*red,delta_ptr[idx]+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
#endif /*_CUBLAS*/
			cudaDeviceSynchronize();
		}
		/*add zero*/
		N=_K.hiddens[1].n_neurons;
		M=_K.hiddens[1].n_inputs;
		red=M/cudas->cuda_n_streams;
		rem=M%cudas->cuda_n_streams;
#ifdef   _CUBLAS
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
			cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red,N,
			&_alpha,_K.hiddens[1].cuda_w+jdx*red,M,delta_ptr[1],1,
			&_beta,delta_ptr[0]+jdx*red,1);
			CHK_ERR(train_gemv);
			dsigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,_K.hiddens[0].cuda_v+jdx*red,delta_ptr[0]+jdx*red);
			CHK_ERR(train_dsigmoid);
		}
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDgemv(cudas->cuda_handle,CUBLAS_OP_N,red+rem,N,
		&_alpha,_K.hiddens[1].cuda_w+jdx*red,M,delta_ptr[1],1,
		&_beta,delta_ptr[0]+jdx*red,1);
		CHK_ERR(train_gemv);
		dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,_K.hiddens[0].cuda_v+jdx*red,delta_ptr[0]+jdx*red);
		CHK_ERR(train_dsigmoid);
#else  /*_CUBLAS*/
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>(red,M,N,_K.hiddens[1].cuda_w+jdx*red,
			delta_ptr[1],_K.hiddens[0].cuda_v+jdx*red,delta_ptr[0]+jdx*red);
			CHK_ERR(train_dsigmoid_mul_delta_T);
		}
		dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>(red+rem,M,N,_K.hiddens[1].cuda_w+jdx*red,
		delta_ptr[1],_K.hiddens[0].cuda_v+jdx*red,delta_ptr[0]+jdx*red);
		CHK_ERR(train_dsigmoid_mul_delta_T);
#endif /*_CUBLAS*/
		cudaDeviceSynchronize();
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
		cublasDger(cudas->cuda_handle,M,red,&_alpha,_K.hiddens[_K.n_hiddens-1].cuda_v,1,
		delta_ptr[_K.n_hiddens]+jdx*red,1,_K.output.cuda_w+jdx*M*red,M);
		CHK_ERR(train_ger);
	}
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle,M,red+rem,&_alpha,_K.hiddens[_K.n_hiddens-1].cuda_v,1,
	delta_ptr[_K.n_hiddens]+jdx*red,1,_K.output.cuda_w+jdx*M*red,M);
	CHK_ERR(train_ger);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,_K.hiddens[_K.n_hiddens-1].cuda_v,_K.output.cuda_w+jdx*M*red);
		CHK_ERR(train_ger_acc);
	}
	ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,_K.hiddens[_K.n_hiddens-1].cuda_v,_K.output.cuda_w+jdx*M*red);
	CHK_ERR(train_ger_acc);
#endif /*_CUBLAS*/
	cudaDeviceSynchronize();
/*^^^ hiddens*/
	for(idx=(_K.n_hiddens-1);idx>0;idx--){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		red=N/cudas->cuda_n_streams;
		rem=N%cudas->cuda_n_streams;
#ifdef   _CUBLAS
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
			cublasDger(cudas->cuda_handle,M,red,&_alpha,_K.hiddens[idx-1].cuda_v,1,
			delta_ptr[idx]+jdx*red,1,_K.hiddens[idx].cuda_w+jdx*M*red,M);
			CHK_ERR(train_ger);
		}
		cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
		cublasDger(cudas->cuda_handle,M,red+rem,&_alpha,_K.hiddens[idx-1].cuda_v,1,
		delta_ptr[idx]+jdx*red,1,_K.hiddens[idx].cuda_w+jdx*M*red,M);
		CHK_ERR(train_ger);
#else  /*_CUBLAS*/
		for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
			ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
				(M,red,LEARN_RATE,delta_ptr[idx]+jdx*red,_K.hiddens[idx-1].cuda_v,_K.hiddens[idx].cuda_w+jdx*M*red);
			CHK_ERR(train_ger_acc);
		}
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red+rem,LEARN_RATE,delta_ptr[idx]+jdx*red,_K.hiddens[idx-1].cuda_v,_K.hiddens[idx].cuda_w+jdx*M*red);
		CHK_ERR(train_ger_acc);
#endif /*_CUBLAS*/
		cudaDeviceSynchronize();
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
		CHK_ERR(train_ger);
	}
	cublasSetStream(cudas->cuda_handle,cudas->cuda_streams[jdx]);
	cublasDger(cudas->cuda_handle,M,red+rem,&_alpha,_K.cuda_in,1,delta_ptr[0]+jdx*red,1,_K.hiddens[0].cuda_w+jdx*M*red,M);
	CHK_ERR(train_ger);
#else  /*_CUBLAS*/
	for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
		ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
			(M,red,LEARN_RATE,delta_ptr[0]+jdx*red,_K.cuda_in,_K.hiddens[0].cuda_w+jdx*M*red);
		CHK_ERR(train_ger_acc);
	}
	ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
		(M,red+rem,LEARN_RATE,delta_ptr[0]+jdx*red,_K.cuda_in,_K.hiddens[0].cuda_w+jdx*M*red);
	CHK_ERR(train_ger_acc);
#endif /*_CUBLAS*/
	cudaDeviceSynchronize();
/*+++ IV - update error +++*/
	N=_K.n_outputs;
	/*update kernel*/
	scuda_ann_forward_cublas(kernel,cudas);
	Epr=scuda_ann_error(kernel,train,cudas);
//	fprintf(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ V - cleanup +++*/
	for(idx=0;idx<(_K.n_hiddens+1);idx++){
		CUDA_FREE(delta_ptr[idx]);
		delta_ptr[idx]=NULL;
	}
	FREE(delta_ptr);
	CHK_ERR(free_1);
	return Ep-Epr;
}

}/*extern "C"*/
