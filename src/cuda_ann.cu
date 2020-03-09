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
#ifdef DEBUG
#include <cuda.h>
#endif /*DEBUG*/
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
#define _Kx (*kx)
/*---------------------------------------*/
/*+++ de-allocate CUDA-part of kernel +++*/
/*---------------------------------------*/
void scuda_ann_deallocate(kernel_ann *kernel,cudastreams *cudas){
    int idx;
    CUDA_SET_DEV(*cudas,0);/*make sure all de-allocation happen on gpu[0]*/
    CUDA_FREE(_K.in);
    for(idx=0;idx<_K.n_hiddens;idx++){
        CUDA_FREE(_K.hiddens[idx].weights);
        CUDA_FREE(_K.hiddens[idx].vec);
    }
    CUDA_FREE(_K.output.weights);
    CUDA_FREE(_K.output.vec);
    CUDA_FREE(_K.tmp_gpu);
    if(cudas->mem_model==CUDA_MEM_EXP){
        /*free allocations for all other GPUs*/
        if(cudas->n_gpu>1){
            int gpu;
            kernel_ann *kx;
            for(gpu=1;gpu<cudas->n_gpu;gpu++){
                CUDA_SET_DEV(*cudas,gpu);
                kx=_K.kerns[gpu];
                CUDA_FREE(_Kx.in);
                for(idx=0;idx<_Kx.n_hiddens;idx++){
                    CUDA_FREE(_Kx.hiddens[idx].weights);
                    CUDA_FREE(_Kx.hiddens[idx].vec);
                }
                CUDA_FREE(_Kx.output.weights);
                CUDA_FREE(_Kx.output.vec);
                CUDA_FREE(_Kx.tmp_gpu);
            }
        }
    }
}
/*------------------------------------*/
/*+++ allocate CUDA-part of kernel +++*/
/*------------------------------------*/
int64_t scuda_ann_allocate(kernel_ann *kernel,cudastreams *cudas){
    int64_t allocate=0;
    int idx;
    /*allocate everything according to memory model*/
    switch(cudas->mem_model){
    case CUDA_MEM_EXP:
        /*allocate on other GPUs (no report)*/
        if(cudas->n_gpu>1){
            int gpu;
            kernel_ann *kx;
            for(gpu=1;gpu<cudas->n_gpu;gpu++){
                CUDA_SET_DEV(*cudas,gpu);
                kx=_K.kerns[gpu];
                CUDA_ALLOC(_Kx.in,_Kx.n_inputs,DOUBLE);
                for(idx=0;idx<_Kx.n_hiddens;idx++){
                    CUDA_ALLOC(_Kx.hiddens[idx].weights,
                    _K.hiddens[idx].n_inputs*_Kx.hiddens[idx].n_neurons,DOUBLE);
                    CUDA_ALLOC(_Kx.hiddens[idx].vec,
                        _Kx.hiddens[idx].n_neurons,DOUBLE);
                }
                CUDA_ALLOC(_Kx.output.weights,
                        _Kx.output.n_inputs*_K.output.n_neurons,DOUBLE);
                CUDA_ALLOC(_Kx.output.vec,
                        _Kx.output.n_neurons,DOUBLE);
                /*allocate the temporary GPU array*/
                CUDA_ALLOC(_Kx.tmp_gpu,_Kx.max_index,DOUBLE);
            }
        }
        /*pass through*/
    case CUDA_MEM_P2P:
    case CUDA_MEM_NONE:
        /*in all cases, we need to initialize memory on GPU[0]*/
        CUDA_SET_DEV(*cudas,0);
        CUDA_ALLOC_REPORT(_K.in,_K.n_inputs,DOUBLE,allocate);
        for(idx=0;idx<_K.n_hiddens;idx++){
            CUDA_ALLOC_REPORT(_K.hiddens[idx].weights,
                _K.hiddens[idx].n_inputs*_K.hiddens[idx].n_neurons,
                DOUBLE,allocate);
            CUDA_ALLOC_REPORT(_K.hiddens[idx].vec,
                _K.hiddens[idx].n_neurons,DOUBLE,allocate);
        }
        CUDA_ALLOC_REPORT(_K.output.weights,
                _K.output.n_inputs*_K.output.n_neurons,DOUBLE,allocate);
        CUDA_ALLOC_REPORT(_K.output.vec,_K.output.n_neurons,DOUBLE,allocate);
        /*allocate the temporary GPU array*/
        CUDA_ALLOC_REPORT(_K.tmp_gpu,_K.max_index,DOUBLE,allocate);
        break;
    case CUDA_MEM_CMM:
        CUDA_SET_DEV(*cudas,0);/*make sure all allocation happen on gpu[0]*/
        CUDA_ALLOC_MM_REPORT(_K.in,_K.n_inputs,DOUBLE,allocate);
        for(idx=0;idx<_K.n_hiddens;idx++){
            CUDA_ALLOC_MM_REPORT(_K.hiddens[idx].weights,
                _K.hiddens[idx].n_inputs*_K.hiddens[idx].n_neurons,
                DOUBLE,allocate);
            CUDA_ALLOC_MM_REPORT(_K.hiddens[idx].vec,_K.hiddens[idx].n_neurons,
                DOUBLE,allocate);
        }
        CUDA_ALLOC_MM_REPORT(_K.output.weights,
                _K.output.n_inputs*_K.output.n_neurons,DOUBLE,allocate);
        CUDA_ALLOC_MM_REPORT(_K.output.vec,_K.output.n_neurons,DOUBLE,allocate);
        /*allocate the temporary GPU array*/
        CUDA_ALLOC_MM_REPORT(_K.tmp_gpu,_K.max_index,DOUBLE,allocate);
        break;
    default:
        break;
    }
    return allocate;
}
/*--------------------------*/
/*+++ free CUDA-momentum +++*/
/*--------------------------*/
BOOL scuda_ann_free_momentum(kernel_ann *kernel,cudastreams *cudas){
    int idx;
    if(_K.dw==NULL) return FALSE;
    CUDA_SET_DEV(*cudas,0);/*make sure all de-allocation happen on gpu[0]*/
    for(idx=0;idx<_K.n_hiddens+1;idx++) CUDA_FREE(_K.dw[idx]);
    if(cudas->mem_model==CUDA_MEM_EXP){
        /*free allocations for all other GPUs*/
        if(cudas->n_gpu>1){
            int gpu;
            kernel_ann *kx;
            for(gpu=1;gpu<cudas->n_gpu;gpu++){
                CUDA_SET_DEV(*cudas,gpu);
                kx=_K.kerns[gpu];
                for(idx=0;idx<_Kx.n_hiddens+1;idx++) CUDA_FREE(_Kx.dw[idx]);
            }
        }
    }
    return TRUE;
}
/*------------------------------*/
/*+++ allocate CUDA-momentum +++*/
/*------------------------------*/
int64_t scuda_ann_allocate_momentum(kernel_ann *kernel,cudastreams *cudas){
    int64_t allocate=0;
    int idx;
    switch(cudas->mem_model){
    case CUDA_MEM_EXP:
        /*allocate on other GPUs*/
        if(cudas->n_gpu>1){
            int gpu;
            kernel_ann *kx;
            for(gpu=1;gpu<cudas->n_gpu;gpu++){
                CUDA_SET_DEV(*cudas,gpu);
                kx=_K.kerns[gpu];
                CUDA_ALLOC_REPORT(_Kx.dw[_Kx.n_hiddens],
                    _Kx.output.n_inputs*_Kx.output.n_neurons,DOUBLE,allocate);
                for(idx=0;idx<_Kx.n_hiddens;idx++)
                    CUDA_ALLOC(_Kx.dw[idx],
                        _Kx.hiddens[idx].n_inputs*_Kx.hiddens[idx].n_neurons,
                        DOUBLE);
            }
        }
        /*pass through*/
    case CUDA_MEM_P2P:
    case CUDA_MEM_NONE:
        CUDA_SET_DEV(*cudas,0);/*make sure all allocation happen on gpu[0]*/
        allocate=0;
        CUDA_ALLOC_REPORT(_K.dw[_K.n_hiddens],
            _K.output.n_inputs*_K.output.n_neurons,DOUBLE,allocate);
        for(idx=0;idx<_K.n_hiddens;idx++)
            CUDA_ALLOC_REPORT(_K.dw[idx],
                _K.hiddens[idx].n_inputs*_K.hiddens[idx].n_neurons,
                DOUBLE,allocate);
        break;
    case CUDA_MEM_CMM:
        CUDA_SET_DEV(*cudas,0);/*make sure all allocation happen on gpu[0]*/
        allocate=0;
        CUDA_ALLOC_MM_REPORT(_K.dw[_K.n_hiddens],
            _K.output.n_inputs*_K.output.n_neurons,DOUBLE,allocate);
        for(idx=0;idx<_K.n_hiddens;idx++)
            CUDA_ALLOC_MM_REPORT(_K.dw[idx],
                _K.hiddens[idx].n_inputs*_K.hiddens[idx].n_neurons,
                DOUBLE,allocate);
        break;
    default:
        break;
    }
    return allocate;
}
/*--------------------------------------*/
/*+++ transfer a weight array to GPU +++*/
/*--------------------------------------*/
void scuda_ann_weight_transfer_C2G
(kernel_ann *kernel, int index, DOUBLE *weight, cudastreams *cudas){
    int M, N;
    /*index correspond to the hidden layer index
     * unless index>=n_hiddens then weight comes
     * from the output layer.           -- OVHPA*/
    if(index>=_K.n_hiddens){
        /*target: output*/
        N=_K.output.n_neurons;
        M=_K.output.n_inputs;
    }else{
        /*target: hiddens[idx]*/
        N=_K.hiddens[index].n_neurons;
        M=_K.hiddens[index].n_inputs;
    }
    switch(cudas->mem_model){
    case CUDA_MEM_EXP:
        /*transfer to other GPUs*/
        if(cudas->n_gpu>1){
            int gpu;
            kernel_ann *kx;
            for(gpu=1;gpu<cudas->n_gpu;gpu++){
                CUDA_SET_DEV(*cudas,gpu);
                kx=_K.kerns[gpu];
                if(index>=_K.n_hiddens){
                    /*target: output*/
                    CUDA_C2G_CP(weight,_Kx.output.weights,M*N,double);
                    CHK_ERR(weights_transfer_C2G);
                }else{
                    /*target: hiddens[idx]*/
                    CUDA_C2G_CP(weight,_Kx.hiddens[index].weights,M*N,double);
                    CHK_ERR(weights_transfer_C2G);
                }
            }
        }
        /*pass through*/
    case CUDA_MEM_P2P:
    case CUDA_MEM_NONE:
        CUDA_SET_DEV(*cudas,0);/*make sure all transfer happen to gpu[0]*/
        if(index>=_K.n_hiddens){
            /*target: output*/
            CUDA_C2G_CP(weight,_K.output.weights,M*N,double);
            CHK_ERR(weights_transfer_C2G);
        }else{
            /*target: hiddens[idx]*/
            CUDA_C2G_CP(weight,_K.hiddens[index].weights,M*N,double);
            CHK_ERR(weights_transfer_C2G);
        }
        break;
    case CUDA_MEM_CMM:
        /*weights can be access directly on GPU*/
        break;
    default:
        return;
    }
    CUDA_SYNC();
}
/*--------------------------------------*/
/*+++ transfer a weight array to CPU +++*/
/*--------------------------------------*/
void scuda_ann_weight_transfer_G2C(kernel_ann *kernel,int index,
                                    DOUBLE **weight,cudastreams *cudas){
    int M, N;
    if(index>=_K.n_hiddens){
        N=_K.output.n_neurons;
        M=_K.output.n_inputs;
    }else{
        N=_K.hiddens[index].n_neurons;
        M=_K.hiddens[index].n_inputs;
    }
    switch(cudas->mem_model){
    case CUDA_MEM_EXP:
        /*no need to transfer from other GPUs!*/
    case CUDA_MEM_P2P:
    case CUDA_MEM_NONE:
        CUDA_SET_DEV(*cudas,0);/*make sure all transfer happen from gpu[0]*/
        if(index>=_K.n_hiddens){
            /*target: output*/
            CUDA_G2C_CP(*weight,_K.output.weights,M*N,double);
            CHK_ERR(weights_transfer_C2G);
        }else{
            /*target: hiddens[idx]*/
            CUDA_G2C_CP(*weight,_K.hiddens[index].weights,M*N,double);
            CHK_ERR(weights_transfer_C2G);
        }
        break;
    case CUDA_MEM_CMM:
        /*weights can be access directly on CPU*/
        break;
    default:
        return;
    }
    /*implicit synchronization*/
}
/*-----------------------------*/
/*+++ forward kernel update +++*/
/*-----------------------------*/
void scuda_ann_forward(kernel_ann *kernel,cudastreams *cudas){
    int idx,jdx;
    int M,N,red;
    int rem,gpu;
    int total_s;
#ifdef _CUBLAS
    double _alpha=1.0;
    double _beta =0.0;
#endif
    kernel_ann *kx;
    int kdx;
    total_s=cudas->cuda_n_streams*cudas->n_gpu;
    CUDA_SET_DEV(*cudas,0);/*always start from GPU[0]*/
if(cudas->mem_model==CUDA_MEM_CMM){
/*+++ Prefetch everything +++*/
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        /*prefetch all input for all GPUs*/
        cudaMemPrefetchAsync(_K.in,_K.n_inputs*sizeof(double),gpu,NULL);
        /*prefetch hiddens[idx].weights and hiddens[idx].vec*/
        for(idx=1;idx<_K.n_hiddens;idx++){
            N=_K.hiddens[idx].n_neurons;
            M=_K.hiddens[idx].n_inputs;
            red=N/total_s;
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                cudaMemPrefetchAsync(_K.hiddens[idx].weights+jdx*M*red,
                    M*red*sizeof(double),gpu,cudas->cuda_streams[jdx]);
                cudaMemPrefetchAsync(_K.hiddens[idx].vec+jdx*red,
                    red*sizeof(double),gpu,cudas->cuda_streams[jdx]);
            }
        }
        /*prefetch output.weights and output.vec*/
        N=_K.output.n_neurons;
        M=_K.output.n_inputs;
        red=N/total_s;
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cudaMemPrefetchAsync(_K.output.weights+jdx*M*red,
                M*red*sizeof(double),gpu,cudas->cuda_streams[jdx]);
            cudaMemPrefetchAsync(_K.output.vec+jdx*red,
                red*sizeof(double),gpu,cudas->cuda_streams[jdx]);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    /*prefetch all input for all GPUs*/
    cudaMemPrefetchAsync(_K.in,_K.n_inputs*sizeof(double),gpu,NULL);
    /*prefetch hiddens[idx].weights and hiddens[idx].vec*/
    for(idx=1;idx<_K.n_hiddens;idx++){
        N=_K.hiddens[idx].n_neurons;
        M=_K.hiddens[idx].n_inputs;
        red=N/total_s;
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cudaMemPrefetchAsync(_K.hiddens[idx].weights+jdx*M*red,
                M*red*sizeof(double),gpu,cudas->cuda_streams[jdx]);
            cudaMemPrefetchAsync(_K.hiddens[idx].vec+jdx*red,
                red*sizeof(double),gpu,cudas->cuda_streams[jdx]);
        }
    }
    /*prefetch output.weights and output.vec*/
    N=_K.output.n_neurons;
    M=_K.output.n_inputs;
    red=N/total_s;
    for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cudaMemPrefetchAsync(_K.output.weights+jdx*M*red,
            M*red*sizeof(double),gpu,cudas->cuda_streams[jdx]);
        cudaMemPrefetchAsync(_K.output.vec+jdx*red,
            red*sizeof(double),gpu,cudas->cuda_streams[jdx]);
    }
/*>>> last stream*/
    jdx=total_s-1;
    /*prefetch hiddens[idx].weights and hiddens[idx].vec*/
    for(idx=1;idx<_K.n_hiddens;idx++){
        N=_K.hiddens[idx].n_neurons;
        M=_K.hiddens[idx].n_inputs;
        red=N/total_s;
        rem=N%total_s;
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cudaMemPrefetchAsync(_K.hiddens[idx].weights+jdx*M*red,
                M*(red+rem)*sizeof(double),gpu,cudas->cuda_streams[jdx]);
            cudaMemPrefetchAsync(_K.hiddens[idx].vec+jdx*red,
                (red+rem)*sizeof(double),gpu,cudas->cuda_streams[jdx]);
        }
    }
    /*prefetch output.weights and output.vec*/
    N=_K.output.n_neurons;
    M=_K.output.n_inputs;
    red=N/total_s;
    rem=N%total_s;
    for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cudaMemPrefetchAsync(_K.output.weights+jdx*M*red,
            M*(red+rem)*sizeof(double),gpu,cudas->cuda_streams[jdx]);
        cudaMemPrefetchAsync(_K.output.vec+jdx*red,
        (red+rem)*sizeof(double),gpu,cudas->cuda_streams[jdx]);
    }
    /*sync all streams/threads on all GPUs*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
}
/*+++ I - input +++*/
    N=_K.hiddens[0].n_neurons;
    M=_K.hiddens[0].n_inputs;
    red=N/total_s;
    rem=N%total_s;
#ifdef   _CUBLAS
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
            cublasDgemv(cudas->cuda_handle[gpu],
                CUBLAS_OP_T,M,red,&_alpha,_K.hiddens[0].weights+jdx*M*red,M,
                _K.in,1,&_beta,_K.hiddens[0].vec+jdx*red,1);
            CHK_ERR(fw_gemv);
            sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,_K.hiddens[0].vec+jdx*red);
            CHK_ERR(fw_sigmoid);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDgemv(cudas->cuda_handle[gpu],
            CUBLAS_OP_T,M,red,&_alpha,_K.hiddens[0].weights+jdx*M*red,M,
            _K.in,1,&_beta,_K.hiddens[0].vec+jdx*red,1);
        CHK_ERR(fw_gemv);
        sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,_K.hiddens[0].vec+jdx*red);
        CHK_ERR(fw_sigmoid);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
    cublasDgemv(cudas->cuda_handle[gpu],
        CUBLAS_OP_T,M,red+rem,&_alpha,_K.hiddens[0].weights+jdx*M*red,M,
        _K.in,1,&_beta,_K.hiddens[0].vec+jdx*red,1);
    CHK_ERR(fw_gemv);
    sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (red+rem,_K.hiddens[0].vec+jdx*red);
    CHK_ERR(fw_sigmoid);
/*>>> sync all streams/threads on all GPUs*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
        cublasDgemv(cudas->cuda_handle[0],
            CUBLAS_OP_T,M,red,&_alpha,_K.hiddens[0].weights+jdx*M*red,M,
            _K.in,1,&_beta,_K.hiddens[0].vec+jdx*red,1);
        CHK_ERR(fw_gemv);
        sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,_K.hiddens[0].vec+jdx*red);
        CHK_ERR(fw_sigmoid);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
            cublasDgemv(cudas->cuda_handle[gpu],
                CUBLAS_OP_T,M,red,&_alpha,_Kx.hiddens[0].weights+jdx*M*red,M,
                _Kx.in,1,&_beta,_Kx.hiddens[0].vec+jdx*red,1);
            CHK_ERR(fw_gemv);
            sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,_Kx.hiddens[0].vec+jdx*red);
            CHK_ERR(fw_sigmoid);
            /*transfer to GPU[0]*/
            CUDA_G2G_SCP(_Kx.hiddens[0].vec+jdx*red,
                _K.hiddens[0].vec+jdx*red,red,double,cudas->cuda_streams[jdx]);
            CHK_ERR(fw_vec_cpy);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDgemv(cudas->cuda_handle[gpu],
            CUBLAS_OP_T,M,red,&_alpha,_Kx.hiddens[0].weights+jdx*M*red,M,
            _Kx.in,1,&_beta,_Kx.hiddens[0].vec+jdx*red,1);
        CHK_ERR(fw_gemv);
        sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,_Kx.hiddens[0].vec+jdx*red);
        CHK_ERR(fw_sigmoid);
        /*transfer to GPU[0]*/
        CUDA_G2G_SCP(_Kx.hiddens[0].vec+jdx*red,
            _K.hiddens[0].vec+jdx*red,red,double,cudas->cuda_streams[jdx]);
        CHK_ERR(fw_vec_cpy);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
    cublasDgemv(cudas->cuda_handle[gpu],
        CUBLAS_OP_T,M,red+rem,&_alpha,_Kx.hiddens[0].weights+jdx*M*red,M,
        _Kx.in,1,&_beta,_Kx.hiddens[0].vec+jdx*red,1);
    CHK_ERR(fw_gemv);
    sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (red+rem,_Kx.hiddens[0].vec+jdx*red);
    CHK_ERR(fw_sigmoid);
    /*transfer to GPU[0]*/
    CUDA_G2G_SCP(_Kx.hiddens[0].vec+jdx*red,
        _K.hiddens[0].vec+jdx*red,red+rem,double,cudas->cuda_streams[jdx]);
    CHK_ERR(fw_vec_cpy);
/*>>> put back vec from GPU[0] to all GPUs*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    red=N/cudas->cuda_n_streams;
    rem=N%cudas->cuda_n_streams;
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            kx=_K.kerns[gpu];
            CUDA_G2G_SCP(_K.hiddens[0].vec+jdx*red,
                _Kx.hiddens[0].vec+jdx*red,red,
                double,cudas->cuda_streams[jdx]);
        }
    }
    /*broadcast the last piece*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        kx=_K.kerns[gpu];
        CUDA_G2G_SCP(_K.hiddens[0].vec+jdx*red,
            _Kx.hiddens[0].vec+jdx*red,red+rem,
            double,cudas->cuda_streams[jdx]);
    }
    CUDA_SYNC();
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,_K.hiddens[0].weights+jdx*M*red,_K.in,
                _K.hiddens[0].vec+jdx*red);
            CHK_ERR(fw_mv_acc);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,_K.hiddens[0].weights+jdx*M*red,_K.in,
            _K.hiddens[0].vec+jdx*red);
        CHK_ERR(fw_mv_acc);
    }
/*>>> last stream*/
    jdx=total_s-1;
    fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,_K.hiddens[0].weights+jdx*M*red,_K.in,
            _K.hiddens[0].vec+jdx*red);
    CHK_ERR(fw_mv_acc);
/*>>> sync all streams/threads on all GPUs*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,_K.hiddens[0].weights+jdx*M*red,_K.in,
            _K.hiddens[0].vec+jdx*red);
        CHK_ERR(fw_mv_acc);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,_Kx.hiddens[0].weights+jdx*M*red,_Kx.in,
                _Kx.hiddens[0].vec+jdx*red);
            CHK_ERR(fw_mv_acc);
            /*transfer to GPU[0]*/
            CUDA_G2G_SCP(_Kx.hiddens[0].vec+jdx*red,
                _K.hiddens[0].vec+jdx*red,red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(fw_vec_cpy);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,_Kx.hiddens[0].weights+jdx*M*red,_Kx.in,
            _Kx.hiddens[0].vec+jdx*red);
        CHK_ERR(fw_mv_acc);
        /*transfer to GPU[0]*/
        CUDA_G2G_SCP(_Kx.hiddens[0].vec+jdx*red,
            _K.hiddens[0].vec+jdx*red,red,double,cudas->cuda_streams[jdx]);
        CHK_ERR(fw_vec_cpy);
    }
/*>>> last stream*/
    jdx=total_s-1;
    fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,_Kx.hiddens[0].weights+jdx*M*red,_Kx.in,
            _Kx.hiddens[0].vec+jdx*red);
    CHK_ERR(fw_mv_acc);
    /*transfer to GPU[0]*/
    CUDA_G2G_SCP(_Kx.hiddens[0].vec+jdx*red,
        _K.hiddens[0].vec+jdx*red,red+rem,double,cudas->cuda_streams[jdx]);
    CHK_ERR(fw_vec_cpy);
/*>>> put back vec from GPU[0] to all GPUs*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    red=N/cudas->cuda_n_streams;
    rem=N%cudas->cuda_n_streams;
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            kx=_K.kerns[gpu];
            CUDA_G2G_SCP(_K.hiddens[0].vec+jdx*red,
                _Kx.hiddens[0].vec+jdx*red,red,double,cudas->cuda_streams[jdx]);
        }
    }
    /*broadcast the last piece*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        kx=_K.kerns[gpu];
        CUDA_G2G_SCP(_K.hiddens[0].vec+jdx*red,
            _Kx.hiddens[0].vec+jdx*red,red+rem,double,cudas->cuda_streams[jdx]);
    }
    CUDA_SYNC();
}
#endif /*_CUBLAS*/
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
            CUDA_SET_DEV(*cudas,gpu);
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                cublasSetStream(cudas->cuda_handle[gpu],
                                cudas->cuda_streams[jdx]);
                cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,
                    &_alpha,_K.hiddens[idx].weights+jdx*M*red,
                    M,_K.hiddens[idx-1].vec,1,
                    &_beta,_K.hiddens[idx].vec+jdx*red,1);
                CHK_ERR(fw_gemv);
                sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                    (red,_K.hiddens[idx].vec+jdx*red);
                CHK_ERR(fw_sigmoid);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
            cublasDgemv(cudas->cuda_handle[gpu],
                CUBLAS_OP_T,M,red,&_alpha,_K.hiddens[idx].weights+jdx*M*red,M,
                _K.hiddens[idx-1].vec,1,&_beta,_K.hiddens[idx].vec+jdx*red,1);
            CHK_ERR(fw_gemv);
            sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,_K.hiddens[idx].vec+jdx*red);
            CHK_ERR(fw_sigmoid);
        }
/*>>> last stream*/
        jdx=total_s-1;
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDgemv(cudas->cuda_handle[gpu],
            CUBLAS_OP_T,M,red+rem,&_alpha,_K.hiddens[idx].weights+jdx*M*red,M,
            _K.hiddens[idx-1].vec,1,&_beta,_K.hiddens[idx].vec+jdx*red,1);
        CHK_ERR(cublas_1);
        sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (red+rem,_K.hiddens[idx].vec+jdx*red);
        CHK_ERR(kernel_1);
/*>>> sync all streams/threads on all GPUs*/
        for(gpu=0;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
        }
}else{
/*>>> first GPU[0]*/
        CUDA_SET_DEV(*cudas,0);
        for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
            cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
            cublasDgemv(cudas->cuda_handle[0],
                CUBLAS_OP_T,M,red,&_alpha,_K.hiddens[idx].weights+jdx*M*red,M,
                _K.hiddens[idx-1].vec,1,&_beta,_K.hiddens[idx].vec+jdx*red,1);
            CHK_ERR(fw_gemv);
            sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,_K.hiddens[idx].vec+jdx*red);
            CHK_ERR(fw_sigmoid);
        }
/*>>> next GPUs but the last one*/
        for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                cublasSetStream(cudas->cuda_handle[gpu],
                                cudas->cuda_streams[jdx]);
                cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_T,M,red,
                &_alpha,_Kx.hiddens[idx].weights+jdx*M*red,
                M,_Kx.hiddens[idx-1].vec,1,
                &_beta,_Kx.hiddens[idx].vec+jdx*red,1);
                CHK_ERR(fw_gemv);
                sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                    (red,_Kx.hiddens[idx].vec+jdx*red);
                CHK_ERR(fw_sigmoid);
                /*transfer to GPU[0]*/
                CUDA_G2G_SCP(_Kx.hiddens[idx].vec+jdx*red,
                    _K.hiddens[idx].vec+jdx*red,red,
                    double,cudas->cuda_streams[jdx]);
                CHK_ERR(fw_vec_cpy);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
            cublasDgemv(cudas->cuda_handle[gpu],
                CUBLAS_OP_T,M,red,&_alpha,_Kx.hiddens[idx].weights+jdx*M*red,M,
                _Kx.hiddens[idx-1].vec,1,&_beta,_Kx.hiddens[idx].vec+jdx*red,1);
            CHK_ERR(fw_gemv);
            sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,_Kx.hiddens[idx].vec+jdx*red);
            CHK_ERR(fw_sigmoid);
            /*transfer to GPU[0]*/
            CUDA_G2G_SCP(_Kx.hiddens[idx].vec+jdx*red,
                _K.hiddens[idx].vec+jdx*red,red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(fw_vec_cpy);
        }
/*>>> last stream*/
        jdx=total_s-1;
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDgemv(cudas->cuda_handle[gpu],
            CUBLAS_OP_T,M,red+rem,&_alpha,_Kx.hiddens[idx].weights+jdx*M*red,M,
            _Kx.hiddens[idx-1].vec,1,&_beta,_Kx.hiddens[idx].vec+jdx*red,1);
        CHK_ERR(cublas_1);
        sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (red+rem,_Kx.hiddens[idx].vec+jdx*red);
        CHK_ERR(kernel_1);
        /*transfer to GPU[0]*/
        CUDA_G2G_SCP(_Kx.hiddens[idx].vec+jdx*red,
            _K.hiddens[idx].vec+jdx*red,red+rem,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(fw_vec_cpy);
/*>>> put back vec from GPU[0] to all GPUs*/
        /*before that we may need to sync all copies to GPU[0]*/
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
        }
        red=N/cudas->cuda_n_streams;
        rem=N%cudas->cuda_n_streams;
        CUDA_SET_DEV(*cudas,0);
        for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
            for(gpu=1;gpu<cudas->n_gpu;gpu++){
                kx=_K.kerns[gpu];
                CUDA_G2G_SCP(_K.hiddens[idx].vec+jdx*red,
                    _Kx.hiddens[idx].vec+jdx*red,red,
                    double,cudas->cuda_streams[jdx]);
            }
        }
        /*broadcast the last piece*/
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            kx=_K.kerns[gpu];
            CUDA_G2G_SCP(_K.hiddens[idx].vec+jdx*red,
                _Kx.hiddens[idx].vec+jdx*red,red+rem,
                double,cudas->cuda_streams[jdx]);
        }
        CUDA_SYNC();
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
        for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                    (M,red,_K.hiddens[idx].weights+jdx*M*red,
                    _K.hiddens[idx-1].vec,_K.hiddens[idx].vec+jdx*red);
                CHK_ERR(fw_mv_acc);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,_K.hiddens[idx].weights+jdx*M*red,
                _K.hiddens[idx-1].vec,_K.hiddens[idx].vec+jdx*red);
            CHK_ERR(fw_mv_acc);
        }
/*>>> last stream*/
        jdx=total_s-1;
        fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (M,red+rem,_K.hiddens[idx].weights+jdx*M*red,
            _K.hiddens[idx-1].vec,_K.hiddens[idx].vec+jdx*red);
        CHK_ERR(fw_mv_acc);
/*>>> sync all streams/threads on all GPUs*/
        for(gpu=0;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
        }
}else{
/*>>> first GPU[0]*/
        CUDA_SET_DEV(*cudas,0);
        for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
            fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,_K.hiddens[idx].weights+jdx*M*red,
                _K.hiddens[idx-1].vec,_K.hiddens[idx].vec+jdx*red);
            CHK_ERR(fw_mv_acc);
        }
/*>>> next GPUs but the last one*/
        for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                    (M,red,_Kx.hiddens[idx].weights+jdx*M*red,
                    _Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].vec+jdx*red);
                CHK_ERR(fw_mv_acc);
                /*transfer to GPU[0]*/
                CUDA_G2G_SCP(_Kx.hiddens[idx].vec+jdx*red,
                    _K.hiddens[idx].vec+jdx*red,red,
                    double,cudas->cuda_streams[jdx]);
                CHK_ERR(fw_vec_cpy);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,_Kx.hiddens[idx].weights+jdx*M*red,
                _Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].vec+jdx*red);
            CHK_ERR(fw_mv_acc);
            /*transfer to GPU[0]*/
            CUDA_G2G_SCP(_Kx.hiddens[idx].vec+jdx*red,
                _K.hiddens[idx].vec+jdx*red,red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(fw_vec_cpy);
        }
/*>>> last stream*/
        jdx=total_s-1;
        fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (M,red+rem,_Kx.hiddens[idx].weights+jdx*M*red,
            _Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].vec+jdx*red);
        CHK_ERR(fw_mv_acc);
        /*transfer to GPU[0]*/
        CUDA_G2G_SCP(_Kx.hiddens[idx].vec+jdx*red,
            _K.hiddens[idx].vec+jdx*red,red+rem,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(fw_vec_cpy);
/*>>> put back vec from GPU[0] to all GPUs*/
        /*before that we may need to sync all copies to GPU[0]*/
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
        }
        red=N/cudas->cuda_n_streams;
        rem=N%cudas->cuda_n_streams;
        CUDA_SET_DEV(*cudas,0);
        for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
            for(gpu=1;gpu<cudas->n_gpu;gpu++){
                kx=_K.kerns[gpu];
                CUDA_G2G_SCP(_K.hiddens[idx].vec+jdx*red,
                    _Kx.hiddens[idx].vec+jdx*red,red,
                    double,cudas->cuda_streams[jdx]);
            }
        }
        /*broadcast the last piece*/
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            kx=_K.kerns[gpu];
            CUDA_G2G_SCP(_K.hiddens[idx].vec+jdx*red,
                _Kx.hiddens[idx].vec+jdx*red,red+rem,
                double,cudas->cuda_streams[jdx]);
        }
        CUDA_SYNC();
}
#endif /*_CUBLAS*/
    }
/*+++ III - output +++*/
    N=_K.output.n_neurons;
    M=_K.output.n_inputs;
    red=N/total_s;
    rem=N%total_s;
#ifdef   _CUBLAS
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
            cublasDgemv(cudas->cuda_handle[gpu],
                CUBLAS_OP_T,M,red,&_alpha,_K.output.weights+jdx*M*red,M,
                _K.hiddens[_K.n_hiddens-1].vec,1,
                &_beta,_K.output.vec+jdx*red,1);
            CHK_ERR(fw_gemv);
            sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,_K.output.vec+jdx*red);
            CHK_ERR(fw_sigmoid);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDgemv(cudas->cuda_handle[gpu],
            CUBLAS_OP_T,M,red,&_alpha,_K.output.weights+jdx*M*red,M,
            _K.hiddens[_K.n_hiddens-1].vec,1,
            &_beta,_K.output.vec+jdx*red,1);
        CHK_ERR(fw_gemv);
        sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,_K.output.vec+jdx*red);
        CHK_ERR(fw_sigmoid);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
    cublasDgemv(cudas->cuda_handle[gpu],
        CUBLAS_OP_T,M,red+rem,&_alpha,_K.output.weights+jdx*M*red,M,
        _K.hiddens[_K.n_hiddens-1].vec,1,
        &_beta,_K.output.vec+jdx*red,1);
    CHK_ERR(fw_gemv);
    sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (red+rem,_K.output.vec+jdx*red);
    CHK_ERR(fw_sigmoid);
    /*sync all streams/threads on all GPUs*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
        cublasDgemv(cudas->cuda_handle[0],
            CUBLAS_OP_T,M,red,&_alpha,_K.output.weights+jdx*M*red,M,
            _K.hiddens[_K.n_hiddens-1].vec,1,
            &_beta,_K.output.vec+jdx*red,1);
        CHK_ERR(fw_gemv);
        sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,_K.output.vec+jdx*red);
        CHK_ERR(fw_sigmoid);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
            cublasDgemv(cudas->cuda_handle[gpu],
                CUBLAS_OP_T,M,red,&_alpha,_Kx.output.weights+jdx*M*red,M,
                _Kx.hiddens[_Kx.n_hiddens-1].vec,1,
                &_beta,_Kx.output.vec+jdx*red,1);
            CHK_ERR(fw_gemv);
            sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,_Kx.output.vec+jdx*red);
            CHK_ERR(fw_sigmoid);
            /*transfer to GPU[0]*/
            CUDA_G2G_SCP(_Kx.output.vec+jdx*red,
                _K.output.vec+jdx*red,red,double,cudas->cuda_streams[jdx]);
            CHK_ERR(fw_vec_cpy);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDgemv(cudas->cuda_handle[gpu],
            CUBLAS_OP_T,M,red,&_alpha,_Kx.output.weights+jdx*M*red,M,
            _Kx.hiddens[_Kx.n_hiddens-1].vec,1,
            &_beta,_Kx.output.vec+jdx*red,1);
        CHK_ERR(fw_gemv);
        sigmoid<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,_Kx.output.vec+jdx*red);
        CHK_ERR(fw_sigmoid);
        /*transfer to GPU[0]*/
        CUDA_G2G_SCP(_Kx.output.vec+jdx*red,
            _K.output.vec+jdx*red,red,double,cudas->cuda_streams[jdx]);
        CHK_ERR(fw_vec_cpy);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
    cublasDgemv(cudas->cuda_handle[gpu],
        CUBLAS_OP_T,M,red+rem,&_alpha,_Kx.output.weights+jdx*M*red,M,
        _Kx.hiddens[_Kx.n_hiddens-1].vec,1,
        &_beta,_Kx.output.vec+jdx*red,1);
    CHK_ERR(fw_gemv);
    sigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (red+rem,_Kx.output.vec+jdx*red);
    CHK_ERR(fw_sigmoid);
    /*transfer to GPU[0] (we are not on GPU[0])*/
    CUDA_G2G_SCP(_Kx.output.vec+jdx*red,
        _K.output.vec+jdx*red,red+rem,double,cudas->cuda_streams[jdx]);
    CHK_ERR(fw_vec_cpy);
/*>>> put back vec from GPU[0] to all GPUs*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    red=N/cudas->cuda_n_streams;
    rem=N%cudas->cuda_n_streams;
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            kx=_K.kerns[gpu];
            CUDA_G2G_SCP(_K.output.vec+jdx*red,
                _Kx.output.vec+jdx*red,red,double,cudas->cuda_streams[jdx]);
        }
    }
    /*broadcast the last piece*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        kx=_K.kerns[gpu];
        CUDA_G2G_SCP(_K.output.vec+jdx*red,
            _Kx.output.vec+jdx*red,red+rem,double,cudas->cuda_streams[jdx]);
    }
    CUDA_SYNC();
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,_K.output.weights+jdx*M*red,
             _K.hiddens[_K.n_hiddens-1].vec,_K.output.vec+jdx*red);
            CHK_ERR(fw_mv_acc);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
        (M,red,_K.output.weights+jdx*M*red,
         _K.hiddens[_K.n_hiddens-1].vec,_K.output.vec+jdx*red);
        CHK_ERR(fw_mv_acc);
    }
/*>>> last stream*/
    jdx=total_s-1;
    fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,_K.output.weights+jdx*M*red,
         _K.hiddens[_K.n_hiddens-1].vec,_K.output.vec+jdx*red);
    CHK_ERR(fw_mv_acc);
    /*sync all streams/threads on all GPUs*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
        (M,red,_K.output.weights+jdx*M*red,
         _K.hiddens[_K.n_hiddens-1].vec,_K.output.vec+jdx*red);
        CHK_ERR(fw_mv_acc);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,_Kx.output.weights+jdx*M*red,
                _Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.vec+jdx*red);
            CHK_ERR(fw_mv_acc);
            /*transfer to GPU[0]*/
            CUDA_G2G_SCP(_Kx.output.vec+jdx*red,
                _K.output.vec+jdx*red,red,double,cudas->cuda_streams[jdx]);
            CHK_ERR(fw_vec_cpy);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        fw_mv_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,_Kx.output.weights+jdx*M*red,
            _Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.vec+jdx*red);
        CHK_ERR(fw_mv_acc);
        /*transfer to GPU[0]*/
        CUDA_G2G_SCP(_Kx.output.vec+jdx*red,
            _K.output.vec+jdx*red,red,double,cudas->cuda_streams[jdx]);
        CHK_ERR(fw_vec_cpy);
    }
/*>>> last stream*/
    jdx=total_s-1;
    fw_mv_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,_Kx.output.weights+jdx*M*red,
         _Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.vec+jdx*red);
    CHK_ERR(fw_mv_acc);
    /*transfer to GPU[0] (we are not on GPU[0])*/
    CUDA_G2G_SCP(_Kx.output.vec+jdx*red,
        _K.output.vec+jdx*red,red+rem,double,cudas->cuda_streams[jdx]);
    CHK_ERR(fw_vec_cpy);
/*>>> put back vec from GPU[0] to all GPUs*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    red=N/cudas->cuda_n_streams;
    rem=N%cudas->cuda_n_streams;
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams-1;jdx++){
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            kx=_K.kerns[gpu];
            CUDA_G2G_SCP(_K.output.vec+jdx*red,
                _Kx.output.vec+jdx*red,red,double,cudas->cuda_streams[jdx]);
        }
    }
    /*broadcast the last piece*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        kx=_K.kerns[gpu];
        CUDA_G2G_SCP(_K.output.vec+jdx*red,
            _Kx.output.vec+jdx*red,red+rem,double,cudas->cuda_streams[jdx]);
    }
    CUDA_SYNC();
}
#endif /*_CUBLAS*/
}
/*-----------------------------------------------*/
/*+++ Calculate Training Error TODO: optimize +++*/
/*-----------------------------------------------*/
double scuda_ann_error(kernel_ann *kernel,double *train,cudastreams *cudas){
    double dEp=0.;
if(cudas->mem_model==CUDA_MEM_CMM){
    cudaMemPrefetchAsync(_K.output.vec,_K.n_outputs*sizeof(double),0,NULL);
    cudaMemPrefetchAsync(_K.tmp_gpu,_K.n_outputs*sizeof(double),0,NULL);
    cudaMemPrefetchAsync(train,_K.n_outputs*sizeof(double),0,NULL);
}
#ifdef   _CUBLAS
    CUDA_SET_DEV(*cudas,0);/*only on master*/
    /*amb can be stream o=(t-v)*(t-v) -- worth it?*/
    amb<<<_KG(_K.n_outputs)>>>(_K.n_outputs,_K.tmp_gpu,train,_K.output.vec);
    CHK_ERR(err_amb);
    /*it is possible to accumulate Ep within stream -- worth it?*/
    cublasSetStream(cudas->cuda_handle[0],NULL);
    cublasDasum(cudas->cuda_handle[0],_K.n_outputs,_K.tmp_gpu,1,&dEp);
    CHK_ERR(err_asum);
#else  /*_CUBLAS*/
    /*shared memory reduction: no streams*/
    CUDA_SET_DEV(*cudas,0);/*only on master*/
    amb_acc<<<_KG(_K.n_outputs),sizeof(double)*2*(_TPB)>>>
        (_K.n_outputs,_K.tmp_gpu,train,_K.output.vec);
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
void scuda_ann_delta(kernel_ann *kernel,double *train,double **delta_ptr,
                     cudastreams *cudas){
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
    CUDA_SET_DEV(*cudas,0);/*always start from GPU[0]*/
    ALLOC(ptr,cudas->n_gpu,DOUBLE *);/*HOST*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
    for(gpu=0;gpu<cudas->n_gpu;gpu++) ptr[gpu]=NULL;
}else{
    ptr[0]=NULL;/*wasted for clarity*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
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
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            dsigmoid_mul_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,train+jdx*red,_K.output.vec+jdx*red,
                delta_ptr[_K.n_hiddens]+jdx*red);
            CHK_ERR(delta_dsigmoid_mul_dif);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        dsigmoid_mul_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,train+jdx*red,_K.output.vec+jdx*red,
            delta_ptr[_K.n_hiddens]+jdx*red);
        CHK_ERR(delta_dsigmoid_mul_dif);
    }
/*>>> last stream*/
    jdx=total_s-1;
    dsigmoid_mul_diff<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (red+rem,train+jdx*red,_K.output.vec+jdx*red,
        delta_ptr[_K.n_hiddens]+jdx*red);
    CHK_ERR(delta_dsigmoid_mul_dif);
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        dsigmoid_mul_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,train+jdx*red,_K.output.vec+jdx*red,
            delta_ptr[_K.n_hiddens]+jdx*red);
        CHK_ERR(delta_dsigmoid_mul_dif);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get part of train from GPU[0]*/
        jdx=gpu*(cudas->cuda_n_streams);
        CUDA_G2G_CP(train+jdx*red,ptr[gpu]+jdx*red,red*cudas->cuda_n_streams,double);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            dsigmoid_mul_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,ptr[gpu]+kdx*red,_Kx.output.vec+jdx*red,
                 _Kx.tmp_gpu+jdx*red);
            CHK_ERR(delta_dsigmoid_mul_dif);
            /*send back data to delta_ptr*/
            CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                delta_ptr[_Kx.n_hiddens]+jdx*red,red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get part of train from GPU[0]*/
    jdx=gpu*(cudas->cuda_n_streams);
    CUDA_G2G_CP(train+jdx*red,ptr[gpu]+jdx*red,red*cudas->cuda_n_streams+rem,double);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        dsigmoid_mul_diff<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,ptr[gpu]+kdx*red,_Kx.output.vec+jdx*red,_Kx.tmp_gpu+jdx*red);
        CHK_ERR(delta_dsigmoid_mul_dif);
        /*send back data to delta_ptr*/
        CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
            delta_ptr[_Kx.n_hiddens]+jdx*red,red,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
    dsigmoid_mul_diff<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (red+rem,ptr[gpu]+kdx*red,_Kx.output.vec+jdx*red,
        _Kx.tmp_gpu+jdx*red);
    CHK_ERR(delta_dsigmoid_mul_dif);
    /*send back data to delta_ptr*/
    CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
        delta_ptr[_Kx.n_hiddens]+jdx*red,red+rem,
        double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
}
    /*synchronize GPU(s)*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
/*^^^ output to hidden*/
    /*distribution over M due to transposed operations*/
    N=_K.output.n_neurons;
    M=_K.output.n_inputs;
    red=M/total_s;
    rem=M%total_s;
#ifdef   _CUBLAS
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
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
    CUDA_SET_DEV(*cudas,gpu);
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
    jdx=total_s-1;
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
    /*tmp_gpu is available to store one array on each GPU*/
    /*so we need a temporary copy of delta[_K.n_hiddens]!*/
    /* FIX: delta[idx] size is _K.hiddens[idx].n_neurons */
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
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
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[_K.n_hiddens] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[_Kx.n_hiddens],
            ptr[gpu],N,double);
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
                _Kx.hiddens[_Kx.n_hiddens-1].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
            CHK_ERR(train_dsigmoid);
            /*3- send result to delta[_K.n_hiddens-1] on GPU[0]*/
            CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                delta_ptr[_Kx.n_hiddens-1]+jdx*red,red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get full delta[_K.n_hiddens] from GPU[0]*/
    CUDA_G2G_CP(delta_ptr[_Kx.n_hiddens],
        ptr[gpu],N,double);
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
            _Kx.hiddens[_Kx.n_hiddens-1].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
        CHK_ERR(train_dsigmoid);
        /*3- send result to delta[_K.n_hiddens-1] on GPU[0]*/
        CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
            delta_ptr[_Kx.n_hiddens-1]+jdx*red,red,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
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
    CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
        delta_ptr[_Kx.n_hiddens-1]+jdx*red,red+rem,
        double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
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
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,M,N,_K.output.weights+jdx*red,delta_ptr[_K.n_hiddens],
            _K.hiddens[_K.n_hiddens-1].vec+jdx*red,
            delta_ptr[_K.n_hiddens-1]+jdx*red);
        CHK_ERR(train_dsigmoid_mul_delta_T);
    }
/*>>> last stream*/
    jdx=total_s-1;
    dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (red+rem,M,N,_K.output.weights+jdx*red,delta_ptr[_K.n_hiddens],
        _K.hiddens[_K.n_hiddens-1].vec+jdx*red,
        delta_ptr[_K.n_hiddens-1]+jdx*red);
    CHK_ERR(train_dsigmoid_mul_delta_T);
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
        (red,M,N,_K.output.weights+jdx*red,delta_ptr[_K.n_hiddens],
        _K.hiddens[_K.n_hiddens-1].vec+jdx*red,
        delta_ptr[_K.n_hiddens-1]+jdx*red);
        CHK_ERR(train_dsigmoid_mul_delta_T);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[_K.n_hiddens] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[_Kx.n_hiddens],
            ptr[gpu],N,double);
        /*we don't need to sync (I think)*/
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            /*2- calculate tmp_gpu = delta[_K.n_hiddens-1]*/
            dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (red,M,N,_Kx.output.weights+jdx*red,ptr[gpu],
            _Kx.hiddens[_Kx.n_hiddens-1].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
            CHK_ERR(train_dsigmoid_mul_delta_T);
            /*3- send result to delta[_K.n_hiddens-1] on GPU[0]*/
            CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                delta_ptr[_Kx.n_hiddens-1]+jdx*red,red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get full delta[_K.n_hiddens] from GPU[0]*/
    CUDA_G2G_CP(delta_ptr[_Kx.n_hiddens],
        ptr[gpu],N,double);
    /*no sync needed (I think)*/
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        /*2- calculate tmp_gpu = delta[_K.n_hiddens-1]*/
        dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
        (red,M,N,_Kx.output.weights+jdx*red,ptr[gpu],
        _Kx.hiddens[_Kx.n_hiddens-1].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
        CHK_ERR(train_dsigmoid_mul_delta_T);
        /*3- send result to delta[_K.n_hiddens-1] on GPU[0]*/
        CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
            delta_ptr[_Kx.n_hiddens-1]+jdx*red,red,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
    dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (red+rem,M,N,_Kx.output.weights+jdx*red,ptr[gpu],
        _Kx.hiddens[_Kx.n_hiddens-1].vec+jdx*red,
        _Kx.tmp_gpu+jdx*red);
    CHK_ERR(train_dsigmoid_mul_delta_T);
    /*send result to delta[_K.n_hiddens-1] on GPU[0]*/
    CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
        delta_ptr[_Kx.n_hiddens-1]+jdx*red,red+rem,
        double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
}
#endif /*_CUBLAS*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
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
            CUDA_SET_DEV(*cudas,gpu);
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
        CUDA_SET_DEV(*cudas,gpu);
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
        jdx=total_s-1;
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
        CUDA_SET_DEV(*cudas,0);
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
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            /*1- get full delta[idx+1] from GPU[0]*/
            CUDA_G2G_CP(delta_ptr[idx+1],
                ptr[gpu],N,double);
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
                CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                    delta_ptr[idx]+jdx*red,red,double,cudas->cuda_streams[jdx]);
                CHK_ERR(delta_transfer);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[idx+1] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[idx+1],ptr[gpu],N,double);
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
            CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                delta_ptr[idx]+jdx*red,red,double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
/*>>> last stream*/
        jdx=total_s-1;
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
        &_alpha,_Kx.hiddens[idx+1].weights+jdx*red,M,ptr[gpu],1,
        &_beta,_Kx.tmp_gpu+jdx*red,1);
        CHK_ERR(train_gemv);
        dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (red+rem,_Kx.hiddens[idx].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
        CHK_ERR(train_dsigmoid);
        /*send result to delta[idx] on GPU[0]*/
        CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
            delta_ptr[idx]+jdx*red,red+rem,double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
        for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
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
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,M,N,_K.hiddens[idx+1].weights+jdx*red,delta_ptr[idx+1],
                _K.hiddens[idx].vec+jdx*red,delta_ptr[idx]+jdx*red);
            CHK_ERR(train_dsigmoid_mul_delta_T);
        }
/*>>> last stream*/
        jdx=total_s-1;
        dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (red+rem,M,N,_K.hiddens[idx+1].weights+jdx*red,delta_ptr[idx+1],
            _K.hiddens[idx].vec+jdx*red,delta_ptr[idx]+jdx*red);
        CHK_ERR(train_dsigmoid_mul_delta_T);
}else{
/*>>> first GPU[0]*/
        CUDA_SET_DEV(*cudas,0);
        for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
            dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,M,N,_K.hiddens[idx+1].weights+jdx*red,delta_ptr[idx+1],
                _K.hiddens[idx].vec+jdx*red,delta_ptr[idx]+jdx*red);
            CHK_ERR(train_dsigmoid_mul_delta_T);
        }
/*>>> next GPUs but the last one*/
        for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            /*1- get full delta[idx+1] from GPU[0]*/
            CUDA_G2G_CP(delta_ptr[idx+1],
                ptr[gpu],N,double);
            /*we don't need to sync (I think)*/
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                /*2- calculate tmp_gpu = delta[idx]*/
                dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                    (red,M,N,_Kx.hiddens[idx+1].weights+jdx*red,ptr[gpu],
                     _Kx.hiddens[idx].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
                CHK_ERR(train_dsigmoid_mul_delta_T);
                /*3- send result to delta[idx] on GPU[0]*/
                CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                    delta_ptr[idx]+jdx*red,red,double,cudas->cuda_streams[jdx]);
                CHK_ERR(delta_transfer);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[idx+1] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[idx+1],
            ptr[gpu],N,double);
        /*no sync needed (I think)*/
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            /*2- calculate tmp_gpu = delta[idx]*/
            dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,M,N,_Kx.hiddens[idx+1].weights+jdx*red,ptr[gpu],
                 _Kx.hiddens[idx].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
            CHK_ERR(train_dsigmoid_mul_delta_T);
            /*3- send result to delta[idx] on GPU[0]*/
            CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                delta_ptr[idx]+jdx*red,red,double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
/*>>> last stream*/
        jdx=total_s-1;
        dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (red+rem,M,N,_Kx.hiddens[idx+1].weights+jdx*red,ptr[gpu],
            _Kx.hiddens[idx].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
        CHK_ERR(train_dsigmoid_mul_delta_T);
        /*send result to delta[idx] on GPU[0]*/
        CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
            delta_ptr[idx]+jdx*red,red+rem,double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
}
#endif /*_CUBLAS*/
            for(gpu=0;gpu<cudas->n_gpu;gpu++){
                CUDA_SET_DEV(*cudas,gpu);
                CUDA_SYNC();
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
            CUDA_SET_DEV(*cudas,gpu);
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
        CUDA_SET_DEV(*cudas,gpu);
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
        jdx=total_s-1;
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
        CUDA_SET_DEV(*cudas,0);
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
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            /*1- get full delta[1] from GPU[0]*/
            CUDA_G2G_CP(delta_ptr[1],
                ptr[gpu],N,double);
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
                CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                    delta_ptr[0]+jdx*red,red,double,cudas->cuda_streams[jdx]);
                CHK_ERR(delta_transfer);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[1] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[1],
            ptr[gpu],N,double);
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
            CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                delta_ptr[0]+jdx*red,red,double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
/*>>> last stream*/
        jdx=total_s-1;
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDgemv(cudas->cuda_handle[gpu],CUBLAS_OP_N,red+rem,N,
        &_alpha,_Kx.hiddens[1].weights+jdx*red,M,ptr[gpu],1,
        &_beta,_Kx.tmp_gpu+jdx*red,1);
        CHK_ERR(train_gemv);
        dsigmoid<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (red+rem,_Kx.hiddens[0].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
        CHK_ERR(train_dsigmoid);
        /*send result to delta[0] on GPU[0]*/
        CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
            delta_ptr[0]+jdx*red,red+rem,double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
        for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                    (red,M,N,_K.hiddens[1].weights+jdx*red,delta_ptr[1],
                    _K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
                CHK_ERR(train_dsigmoid_mul_delta_T);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,M,N,_K.hiddens[1].weights+jdx*red,
            delta_ptr[1],_K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
            CHK_ERR(train_dsigmoid_mul_delta_T);
        }
/*>>> last stream*/
        jdx=total_s-1;
        dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (red+rem,M,N,_K.hiddens[1].weights+jdx*red,delta_ptr[1],
            _K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
        CHK_ERR(train_dsigmoid_mul_delta_T);
}else{
/*>>> first GPU[0]*/
        CUDA_SET_DEV(*cudas,0);
        for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
            dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,M,N,_K.hiddens[1].weights+jdx*red,
            delta_ptr[1],_K.hiddens[0].vec+jdx*red,delta_ptr[0]+jdx*red);
            CHK_ERR(train_dsigmoid_mul_delta_T);
        }
/*>>> next GPUs but the last one*/
        for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            /*1- get full delta[1] from GPU[0]*/
            CUDA_G2G_CP(delta_ptr[1],
                ptr[gpu],N,double);
            /*we don't need to sync (I think)*/
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                /*2- calculate tmp_gpu = delta[idx]*/
                dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                    (red,M,N,_Kx.hiddens[1].weights+jdx*red,ptr[gpu],
                    _Kx.hiddens[0].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
                CHK_ERR(train_dsigmoid_mul_delta_T);
                /*3- send result to delta[0] on GPU[0]*/
                CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                    delta_ptr[0]+jdx*red,red,
                    double,cudas->cuda_streams[jdx]);
                CHK_ERR(delta_transfer);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[1] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[1],
            ptr[gpu],N,double);
        /*no sync needed (I think)*/
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            /*2- calculate tmp_gpu = delta[idx]*/
            dsigmoid_mul_delta_T<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (red,M,N,_Kx.hiddens[1].weights+jdx*red,ptr[gpu],
                _Kx.hiddens[0].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
            CHK_ERR(train_dsigmoid_mul_delta_T);
            /*3- send result to delta[0] on GPU[0]*/
            CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
                delta_ptr[0]+jdx*red,red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
/*>>> last stream*/
        jdx=total_s-1;
        dsigmoid_mul_delta_T<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (red+rem,M,N,_Kx.hiddens[1].weights+jdx*red,ptr[gpu],
            _Kx.hiddens[0].vec+jdx*red,_Kx.tmp_gpu+jdx*red);
        CHK_ERR(train_dsigmoid_mul_delta_T);
        /*send result to delta[0] on GPU[0]*/
        CUDA_G2G_SCP(_Kx.tmp_gpu+jdx*red,
            delta_ptr[0]+jdx*red,red+rem,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
}
#endif /*_CUBLAS*/
        for(gpu=0;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
        }
    }
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
    FREE(ptr);
}else{
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_FREE(ptr[gpu]);
    }
    CUDA_SET_DEV(*cudas,0);/*go back to GPU[0] <- just in case*/
    FREE(ptr);
}
}
#define LEARN_RATE 0.01
/*------------------------*/
/*+++ back-propagation +++*/
/*------------------------*/
double scuda_ann_train(kernel_ann *kernel,double *train,cudastreams *cudas){
    int idx,jdx;
    int M,N,red;
    int rem,gpu;
    int total_s;
    double **delta_ptr;/*THIS delta belongs to GPU[0]*/
    double Ep =0.;
    double Epr=0.;
#ifdef _CUBLAS
    double _alpha=1.0;
#endif /*_CUBLAS*/
    kernel_ann *kx;
    int kdx;
    total_s=cudas->cuda_n_streams*cudas->n_gpu;
    CUDA_SET_DEV(*cudas,0);/*always start from GPU[0]*/
    /*allocate delta_ptr*/
    CUDA_SET_DEV(*cudas,0);/*make sure all allocation happen on gpu[0]*/
    ALLOC(delta_ptr,_K.n_hiddens+1,DOUBLE *);/*HOST*/
    for(idx=0;idx<_K.n_hiddens;idx++)
        CUDA_ALLOC(delta_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
    CUDA_ALLOC(delta_ptr[_K.n_hiddens],_K.output.n_neurons,DOUBLE);/*DEVICE*/
/*+++ I - FORWARD +++*/
/*>>> in all cases, the FORWARD move should have already be done <<<*/
    Ep=scuda_ann_error(kernel,train,cudas);
//  printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
    scuda_ann_delta(kernel,train,delta_ptr,cudas);
/*+++ III - back propagation +++*/
/*^^^ output*/
    N=_K.output.n_neurons;
    M=_K.output.n_inputs;
    red=N/total_s;
    rem=N%total_s;
#ifdef   _CUBLAS
    _alpha=LEARN_RATE;/*TODO: set as a parameter*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
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
    CUDA_SET_DEV(*cudas,gpu);
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
    jdx=total_s-1;
    cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
    cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
        _K.hiddens[_K.n_hiddens-1].vec,1,delta_ptr[_K.n_hiddens]+jdx*red,
        1,_K.output.weights+jdx*M*red,M);
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
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
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[_K.n_hiddens] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[_Kx.n_hiddens],
            _Kx.tmp_gpu,N,double);
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
            CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
                _K.output.weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get full delta[_K.n_hiddens] from GPU[0]*/
    CUDA_G2G_CP(delta_ptr[_K.n_hiddens],
        _Kx.tmp_gpu,N,double);
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
        CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
            _K.output.weights+jdx*M*red,M*red,double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
    cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
        _Kx.hiddens[_Kx.n_hiddens-1].vec,1,
        _Kx.tmp_gpu+jdx*red,1,
        _Kx.output.weights+jdx*M*red,M);
    CHK_ERR(train_ger);
    /*3- transfer back weights to GPU[0]*/
    CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
        _K.output.weights+jdx*M*red,M*(red+rem),
                 double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
    /*TODO: stream copy version*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    CUDA_SET_DEV(*cudas,0);
    /*TRANSFER*/
    for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
        kx=_K.kerns[gpu];
        /*left part*/
        CUDA_G2G_CP(_K.output.weights,_Kx.output.weights,
            cudas->cuda_n_streams*gpu*M*red,double);
        /*right part*/
        if(gpu==cudas->n_gpu-1) continue;
        CUDA_G2G_CP(_K.output.weights,_Kx.output.weights,
            M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
    }
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,
                _K.hiddens[_K.n_hiddens-1].vec,_K.output.weights+jdx*M*red);
            CHK_ERR(train_ger_acc);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,
            _K.hiddens[_K.n_hiddens-1].vec,_K.output.weights+jdx*M*red);
        CHK_ERR(train_ger_acc);
    }
/*>>> last stream*/
    jdx=total_s-1;
    ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,
        _K.hiddens[_K.n_hiddens-1].vec,_K.output.weights+jdx*M*red);
    CHK_ERR(train_ger_acc);
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,delta_ptr[_K.n_hiddens]+jdx*red,
            _K.hiddens[_K.n_hiddens-1].vec,_K.output.weights+jdx*M*red);
        CHK_ERR(train_ger_acc);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[_K.n_hiddens] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[_Kx.n_hiddens],
            _Kx.tmp_gpu,N,double);
        /*we don't need to sync (I think)*/
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            /*2- calculate weights (vec is GPU-local)*/
            ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
                _Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.weights+jdx*M*red);
            CHK_ERR(train_ger_acc);
            /*3- transfer back weights to GPU[0]*/
            CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
                _K.output.weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get full delta[_K.n_hiddens] from GPU[0]*/
    CUDA_G2G_CP(delta_ptr[_Kx.n_hiddens],
        _Kx.tmp_gpu,N,double);
    /*no sync needed (I think)*/
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        /*2- calculate weights (vec is GPU-local)*/
        ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
            _Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.weights+jdx*M*red);
        CHK_ERR(train_ger_acc);
        /*3- transfer back weights to GPU[0]*/
        CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
            _K.output.weights+jdx*M*red,M*red,double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
    ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
        _Kx.hiddens[_Kx.n_hiddens-1].vec,_Kx.output.weights+jdx*M*red);
    CHK_ERR(train_ger_acc);
    /*3- transfer back weights to GPU[0]*/
    CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
        _K.output.weights+jdx*M*red,M*red,double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
    /*TODO: stream copy version*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    CUDA_SET_DEV(*cudas,0);
    /*TRANSFER*/
    for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
        kx=_K.kerns[gpu];
        /*left part*/
        CUDA_G2G_CP(_K.output.weights,_Kx.output.weights,
            cudas->cuda_n_streams*gpu*M*red,double);
        /*right part*/
        if(gpu==cudas->n_gpu-1) continue;
        CUDA_G2G_CP(_K.output.weights,_Kx.output.weights,
            M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
    }
}
#endif /*_CUBLAS*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
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
            CUDA_SET_DEV(*cudas,gpu);
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
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
            cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
                _K.hiddens[idx-1].vec,1,delta_ptr[idx]+jdx*red,1,
                _K.hiddens[idx].weights+jdx*M*red,M);
            CHK_ERR(train_ger);
        }
/*>>> last stream*/
        jdx=total_s-1;
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
            _K.hiddens[idx-1].vec,1,delta_ptr[idx]+jdx*red,1,
            _K.hiddens[idx].weights+jdx*M*red,M);
        CHK_ERR(train_ger);
}else{
/*>>> first GPU[0]*/
        CUDA_SET_DEV(*cudas,0);
        for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
            cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
            cublasDger(cudas->cuda_handle[0],M,red,&_alpha,
            _K.hiddens[idx-1].vec,1,delta_ptr[idx]+jdx*red,1,
            _K.hiddens[idx].weights+jdx*M*red,M);
            CHK_ERR(train_ger);
        }
/*>>> next GPUs but the last one*/
        for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            /*1- get full delta[idx] from GPU[0]*/
            CUDA_G2G_CP(delta_ptr[idx],
                _Kx.tmp_gpu,N,double);
            /*we don't need to sync (I think)*/
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                /*2- calculate weights (vec is GPU-local)*/
                cublasSetStream(cudas->cuda_handle[gpu],
                                cudas->cuda_streams[jdx]);
                cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,
                    _Kx.hiddens[idx-1].vec,1,_Kx.tmp_gpu+jdx*red,1,
                    _Kx.hiddens[idx].weights+jdx*M*red,M);
                CHK_ERR(train_ger);
                /*3- transfer back weights to GPU[0]*/
                CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
                    _K.hiddens[idx].weights+jdx*M*red,M*red,
                    double,cudas->cuda_streams[jdx]);
                CHK_ERR(delta_transfer);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[idx] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[idx],
            _Kx.tmp_gpu,N,double);
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
            CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
                _K.hiddens[idx].weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
/*>>> last stream*/
        jdx=total_s-1;
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,
            _Kx.hiddens[idx-1].vec,1,_Kx.tmp_gpu+jdx*red,1,
            _Kx.hiddens[idx].weights+jdx*M*red,M);
        CHK_ERR(train_ger);
        /*3- transfer back weights to GPU[0]*/
        CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
            _K.hiddens[idx].weights+jdx*M*red,M*(red+rem),
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
        /*TODO: stream copy version*/
        /*before that we may need to sync all copies to GPU[0]*/
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
        }
        CUDA_SET_DEV(*cudas,0);
        /*TRANSFER*/
        for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
            kx=_K.kerns[gpu];
            /*left part*/
            CUDA_G2G_CP(_K.hiddens[idx].weights,_Kx.hiddens[idx].weights,
                cudas->cuda_n_streams*gpu*M*red,double);
            /*right part*/
            if(gpu==cudas->n_gpu-1) continue;
            CUDA_G2G_CP(_K.hiddens[idx].weights,_Kx.hiddens[idx].weights,
                M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
        }
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
        for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                    (M,red,LEARN_RATE,delta_ptr[idx]+jdx*red,
                    _K.hiddens[idx-1].vec,_K.hiddens[idx].weights+jdx*M*red);
                CHK_ERR(train_ger_acc);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,delta_ptr[idx]+jdx*red,
                _K.hiddens[idx-1].vec,_K.hiddens[idx].weights+jdx*M*red);
            CHK_ERR(train_ger_acc);
        }
/*>>> last stream*/
        jdx=total_s-1;
        ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (M,red+rem,LEARN_RATE,delta_ptr[idx]+jdx*red,
            _K.hiddens[idx-1].vec,_K.hiddens[idx].weights+jdx*M*red);
        CHK_ERR(train_ger_acc);
}else{
/*>>> first GPU[0]*/
        CUDA_SET_DEV(*cudas,0);
        for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
            ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,delta_ptr[idx]+jdx*red,
                _K.hiddens[idx-1].vec,_K.hiddens[idx].weights+jdx*M*red);
            CHK_ERR(train_ger_acc);
        }
/*>>> next GPUs but the last one*/
        for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            /*1- get full delta[idx] from GPU[0]*/
            CUDA_G2G_CP(delta_ptr[idx],_Kx.tmp_gpu,N,double);
            /*we don't need to sync (I think)*/
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                /*2- calculate weights (vec is GPU-local)*/
                ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                    (M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
                    _Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].weights+jdx*M*red);
                CHK_ERR(train_ger_acc);
                /*3- transfer back weights to GPU[0]*/
                CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
                    _K.hiddens[idx].weights+jdx*M*red,M*red,
                    double,cudas->cuda_streams[jdx]);
                CHK_ERR(delta_transfer);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[idx] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[idx],_Kx.tmp_gpu,N,double);
        /*no sync needed (I think)*/
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            /*2- calculate weights (vec is GPU-local)*/
            ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
                _Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].weights+jdx*M*red);
            CHK_ERR(train_ger_acc);
            /*3- transfer back weights to GPU[0]*/
            CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
                _K.hiddens[idx].weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
/*>>> last stream*/
        jdx=total_s-1;
        ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (M,red+rem,LEARN_RATE,_Kx.tmp_gpu+jdx*red,
            _Kx.hiddens[idx-1].vec,_Kx.hiddens[idx].weights+jdx*M*red);
        CHK_ERR(train_ger_acc);
        /*3- transfer back weights to GPU[0]*/
        CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
            _K.hiddens[idx].weights+jdx*M*red,M*(red+rem),
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
        /*TODO: stream copy version*/
        /*before that we may need to sync all copies to GPU[0]*/
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
        }
        CUDA_SET_DEV(*cudas,0);
        /*TRANSFER*/
        for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
            kx=_K.kerns[gpu];
            /*left part*/
            CUDA_G2G_CP(_K.hiddens[idx].weights,_Kx.hiddens[idx].weights,
                cudas->cuda_n_streams*gpu*M*red,double);
            /*right part*/
            if(gpu==cudas->n_gpu-1) continue;
            CUDA_G2G_CP(_K.hiddens[idx].weights,_Kx.hiddens[idx].weights,
                M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
        }
}
#endif /*_CUBLAS*/
        for(gpu=0;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
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
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
            cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,_K.in,1,
                delta_ptr[0]+jdx*red,1,_K.hiddens[0].weights+jdx*M*red,M);
            CHK_ERR(train_ger);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,_K.in,1,
            delta_ptr[0]+jdx*red,1,_K.hiddens[0].weights+jdx*M*red,M);
        CHK_ERR(train_ger);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
    cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,_K.in,1,
        delta_ptr[0]+jdx*red,1,_K.hiddens[0].weights+jdx*M*red,M);
    CHK_ERR(train_ger);
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        cublasSetStream(cudas->cuda_handle[0],cudas->cuda_streams[jdx]);
        cublasDger(cudas->cuda_handle[0],M,red,&_alpha,_K.in,1,
            delta_ptr[0]+jdx*red,1,_K.hiddens[0].weights+jdx*M*red,M);
        CHK_ERR(train_ger);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[0] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[0],_Kx.tmp_gpu,N,double);
        /*we don't need to sync (I think)*/
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            /*2- calculate weights (vec is GPU-local)*/
            cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
            cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,_Kx.in,1,
                _Kx.tmp_gpu+jdx*red,1,_Kx.hiddens[0].weights+jdx*M*red,M);
            CHK_ERR(train_ger);
            /*3- transfer back weights to GPU[0]*/
            CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
                _K.hiddens[0].weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get full delta[idx] from GPU[0]*/
    CUDA_G2G_CP(delta_ptr[0],_Kx.tmp_gpu,N,double);
    /*no sync needed (I think)*/
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        /*2- calculate weights (vec is GPU-local)*/
        cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
        cublasDger(cudas->cuda_handle[gpu],M,red,&_alpha,_Kx.in,1,
            _Kx.tmp_gpu+jdx*red,1,_Kx.hiddens[0].weights+jdx*M*red,M);
        CHK_ERR(train_ger);
        /*3- transfer back weights to GPU[0]*/
        CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
            _K.hiddens[0].weights+jdx*M*red,M*red,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cublasSetStream(cudas->cuda_handle[gpu],cudas->cuda_streams[jdx]);
    cublasDger(cudas->cuda_handle[gpu],M,red+rem,&_alpha,_Kx.in,1,
        _Kx.tmp_gpu+jdx*red,1,_Kx.hiddens[0].weights+jdx*M*red,M);
    CHK_ERR(train_ger);
    /*3- transfer back weights to GPU[0]*/
    CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
        _K.hiddens[0].weights+jdx*M*red,M*(red+rem),
        double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
    /*TODO: stream copy version*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    CUDA_SET_DEV(*cudas,0);
    /*TRANSFER*/
    for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
        kx=_K.kerns[gpu];
        /*left part*/
        CUDA_G2G_CP(_K.hiddens[0].weights,_Kx.hiddens[0].weights,
            cudas->cuda_n_streams*gpu*M*red,double);
        /*right part*/
        if(gpu==cudas->n_gpu-1) continue;
        CUDA_G2G_CP(_K.hiddens[0].weights,_Kx.hiddens[0].weights,
            M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
    }
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,delta_ptr[0]+jdx*red,_K.in,
                _K.hiddens[0].weights+jdx*M*red);
            CHK_ERR(train_ger_acc);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,delta_ptr[0]+jdx*red,_K.in,
            _K.hiddens[0].weights+jdx*M*red);
        CHK_ERR(train_ger_acc);
    }
/*>>> last stream*/
    jdx=total_s-1;
    ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,LEARN_RATE,delta_ptr[0]+jdx*red,_K.in,
        _K.hiddens[0].weights+jdx*M*red);
    CHK_ERR(train_ger_acc);
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,delta_ptr[0]+jdx*red,_K.in,
            _K.hiddens[0].weights+jdx*M*red);
        CHK_ERR(train_ger_acc);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[0] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[0],_Kx.tmp_gpu,N,double);
        /*we don't need to sync (I think)*/
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            /*2- calculate weights (vec is GPU-local)*/
            ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,_Kx.in,
                _Kx.hiddens[0].weights+jdx*M*red);
            CHK_ERR(train_ger_acc);
            /*3- transfer back weights to GPU[0]*/
            CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
                _K.hiddens[0].weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get full delta[idx] from GPU[0]*/
    CUDA_G2G_CP(delta_ptr[0],_Kx.tmp_gpu,N,double);
    /*no sync needed (I think)*/
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        /*2- calculate weights (vec is GPU-local)*/
        ger_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,_Kx.tmp_gpu+jdx*red,_Kx.in,
            _Kx.hiddens[0].weights+jdx*M*red);
        CHK_ERR(train_ger_acc);
        /*3- transfer back weights to GPU[0]*/
        CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
            _K.hiddens[0].weights+jdx*M*red,M*red,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
    ger_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,LEARN_RATE,_Kx.tmp_gpu+jdx*red,_Kx.in,
        _Kx.hiddens[0].weights+jdx*M*red);
    CHK_ERR(train_ger_acc);
    /*3- transfer back weights to GPU[0]*/
    CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
        _K.hiddens[0].weights+jdx*M*red,M*(red+rem),
        double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
    /*TODO: stream copy version*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    CUDA_SET_DEV(*cudas,0);
    /*TRANSFER*/
    for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
        kx=_K.kerns[gpu];
        /*left part*/
        CUDA_G2G_CP(_K.hiddens[0].weights,_Kx.hiddens[0].weights,
            cudas->cuda_n_streams*gpu*M*red,double);
        /*right part*/
        if(gpu==cudas->n_gpu-1) continue;
        CUDA_G2G_CP(_K.hiddens[0].weights,_Kx.hiddens[0].weights,
            M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
    }
}
#endif /*_CUBLAS*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
/*+++ IV - update error +++*/
    /*update kernel*/
    scuda_ann_forward(kernel,cudas);
    Epr=scuda_ann_error(kernel,train,cudas);
//  fprintf(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ V - cleanup +++*/
    CUDA_SET_DEV(*cudas,0);/*make sure all de-allocation happen on gpu[0]*/
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
void scuda_ann_raz_momentum(kernel_ann *kernel,cudastreams *cudas){
    int idx,jdx;
    int M,N,red;
    int rem,gpu;
    int total_s;
    total_s=cudas->cuda_n_streams*cudas->n_gpu;
    kernel_ann *kx;
    int kdx;
/*^^^ output*/
    N=_K.output.n_neurons;
    M=_K.output.n_inputs;
    red=N/total_s;
    rem=N%total_s;
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
                cudaMemsetAsync(_K.dw[_K.n_hiddens]+jdx*M*red,0.,
                red*M*sizeof(double),cudas->cuda_streams[jdx]);
            CHK_ERR(moment_memset);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cudaMemsetAsync(_K.dw[_K.n_hiddens]+jdx*M*red,0.,
            red*M*sizeof(double),cudas->cuda_streams[jdx]);
        CHK_ERR(moment_memset);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cudaMemsetAsync(_K.dw[_K.n_hiddens]+jdx*M*red,0.,
        (red+rem)*M*sizeof(double),cudas->cuda_streams[jdx]);
    CHK_ERR(moment_memset);
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        cudaMemsetAsync(_K.dw[_K.n_hiddens]+jdx*M*red,0.,
            red*M*sizeof(double),cudas->cuda_streams[jdx]);
        CHK_ERR(moment_memset);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cudaMemsetAsync(_Kx.dw[_Kx.n_hiddens]+jdx*M*red,0.,
                red*M*sizeof(double),cudas->cuda_streams[jdx]);
            CHK_ERR(moment_memset);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cudaMemsetAsync(_Kx.dw[_Kx.n_hiddens]+jdx*M*red,0.,
            red*M*sizeof(double),cudas->cuda_streams[jdx]);
        CHK_ERR(moment_memset);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cudaMemsetAsync(_Kx.dw[_Kx.n_hiddens]+jdx*M*red,0.,
        (red+rem)*M*sizeof(double),cudas->cuda_streams[jdx]);
    CHK_ERR(moment_memset);
}
/*^^^ hiddens*/
    for(idx=(_K.n_hiddens-1);idx>0;idx--){
        N=_K.hiddens[idx].n_neurons;
        M=_K.hiddens[idx].n_inputs;
        red=N/total_s;
        rem=N%total_s;
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
        for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                cudaMemsetAsync(_K.dw[idx]+jdx*M*red,0.,
                    red*M*sizeof(double),cudas->cuda_streams[jdx]);
                CHK_ERR(moment_memset);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cudaMemsetAsync(_K.dw[idx]+jdx*M*red,0.,
                red*M*sizeof(double),cudas->cuda_streams[jdx]);
            CHK_ERR(moment_memset);
        }
/*>>> last stream*/
        jdx=total_s-1;
        cudaMemsetAsync(_K.dw[idx]+jdx*M*red,0.,
            (red+rem)*M*sizeof(double),cudas->cuda_streams[jdx]);
        CHK_ERR(moment_memset);
}else{
/*>>> first GPU[0]*/
        CUDA_SET_DEV(*cudas,0);
        for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
            cudaMemsetAsync(_K.dw[idx]+jdx*M*red,0.,
                red*M*sizeof(double),cudas->cuda_streams[jdx]);
            CHK_ERR(moment_memset);
        }
/*>>> next GPUs but the last one*/
        for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
                jdx=kdx+gpu*(cudas->cuda_n_streams);
                cudaMemsetAsync(_Kx.dw[idx]+jdx*M*red,0.,
                    red*M*sizeof(double),cudas->cuda_streams[jdx]);
                CHK_ERR(moment_memset);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cudaMemsetAsync(_Kx.dw[idx]+jdx*M*red,0.,
                red*M*sizeof(double),cudas->cuda_streams[jdx]);
            CHK_ERR(moment_memset);
        }
/*>>> last stream*/
        jdx=total_s-1;
        cudaMemsetAsync(_Kx.dw[idx]+jdx*M*red,0.,
            (red+rem)*M*sizeof(double),cudas->cuda_streams[jdx]);
        CHK_ERR(moment_memset);
}
    }
    /*add zero*/
    N=_K.hiddens[0].n_neurons;
    M=_K.hiddens[0].n_inputs;
    red=N/total_s;
    rem=N%total_s;
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cudaMemsetAsync(_K.dw[0]+jdx*M*red,0.,
                red*M*sizeof(double),cudas->cuda_streams[jdx]);
            CHK_ERR(moment_memset);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cudaMemsetAsync(_K.dw[0]+jdx*M*red,0.,
            red*M*sizeof(double),cudas->cuda_streams[jdx]);
        CHK_ERR(moment_memset);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cudaMemsetAsync(_K.dw[0]+jdx*M*red,0.,
        (red+rem)*M*sizeof(double),cudas->cuda_streams[jdx]);
    CHK_ERR(moment_memset);
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        cudaMemsetAsync(_K.dw[0]+jdx*M*red,0.,
            red*M*sizeof(double),cudas->cuda_streams[jdx]);
        CHK_ERR(moment_memset);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            cudaMemsetAsync(_Kx.dw[0]+jdx*M*red,0.,
                red*M*sizeof(double),cudas->cuda_streams[jdx]);
            CHK_ERR(moment_memset);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        cudaMemsetAsync(_Kx.dw[0]+jdx*M*red,0.,
            red*M*sizeof(double),cudas->cuda_streams[jdx]);
        CHK_ERR(moment_memset);
    }
/*>>> last stream*/
    jdx=total_s-1;
    cudaMemsetAsync(_Kx.dw[0]+jdx*M*red,0.,
        (red+rem)*M*sizeof(double),cudas->cuda_streams[jdx]);
    CHK_ERR(moment_memset);
}
    /*all done, sync required*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    CUDA_SET_DEV(*cudas,0);/*go back to GPU[0] (just in case)*/
}
/*--------------------------------------*/
/*+++ back-propagation with momentum +++*/
/*--------------------------------------*/
double scuda_ann_train_momentum
    (kernel_ann *kernel,double *train,double moment,cudastreams *cudas){
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
    CUDA_SET_DEV(*cudas,0);/*always start from GPU[0]*/
    /*allocate delta_ptr*/
    CUDA_SET_DEV(*cudas,0);/*make sure all allocation happen on gpu[0]*/
    ALLOC(delta_ptr,_K.n_hiddens+1,DOUBLE *);/*HOST*/
    for(idx=0;idx<_K.n_hiddens;idx++){
        CUDA_ALLOC(delta_ptr[idx],_K.hiddens[idx].n_neurons,DOUBLE);/*DEVICE*/
    }
    CUDA_ALLOC(delta_ptr[_K.n_hiddens],_K.output.n_neurons,DOUBLE);/*DEVICE*/
/*+++ I - FORWARD +++*/
/*>>> in all cases, the FORWARD move should have already be done <<<*/
    Ep=scuda_ann_error(kernel,train,cudas);
/// printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
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
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
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
    CUDA_SET_DEV(*cudas,gpu);
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
    jdx=total_s-1;
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
    CUDA_SYNC();
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
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
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[_K.n_hiddens] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[_Kx.n_hiddens],_Kx.tmp_gpu,N,double);
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
            CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
                _K.output.weights+jdx*M*red,
                M*red,double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        /*PS: The same portion of momentum is always applied to the same GPU*/
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get full delta[_K.n_hiddens] from GPU[0]*/
    CUDA_G2G_CP(delta_ptr[_K.n_hiddens],_Kx.tmp_gpu,N,double);
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
        CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
            _K.output.weights+jdx*M*red,
            M*red,double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
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
    CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
        _K.output.weights+jdx*M*red,M*(red+rem),
        double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
    /*TODO: stream copy version*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    CUDA_SET_DEV(*cudas,0);
    /*TRANSFER*/
    for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
        kx=_K.kerns[gpu];
        /*left part*/
        CUDA_G2G_CP(_K.output.weights,_Kx.output.weights,
            cudas->cuda_n_streams*gpu*M*red,double);
        /*right part*/
        if(gpu==cudas->n_gpu-1) continue;
        CUDA_G2G_CP(_K.output.weights,_Kx.output.weights,
            M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
    }
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
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
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,moment,delta_ptr[_K.n_hiddens]+jdx*red,
            _K.hiddens[_K.n_hiddens-1].vec,
            _K.dw[_K.n_hiddens]+jdx*M*red,_K.output.weights+jdx*M*red);
        CHK_ERR(moment_ger_dw_acc);
    }
/*>>> last stream*/
    jdx=total_s-1;
    ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,LEARN_RATE,moment,delta_ptr[_K.n_hiddens]+jdx*red,
        _K.hiddens[_K.n_hiddens-1].vec,
        _K.dw[_K.n_hiddens]+jdx*M*red,_K.output.weights+jdx*M*red);
    CHK_ERR(moment_ger_dw_acc);
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,moment,delta_ptr[_K.n_hiddens]+jdx*red,
            _K.hiddens[_K.n_hiddens-1].vec,
            _K.dw[_K.n_hiddens]+jdx*M*red,_K.output.weights+jdx*M*red);
        CHK_ERR(moment_ger_dw_acc);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[_K.n_hiddens] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[_Kx.n_hiddens],_Kx.tmp_gpu,N,double);
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
            CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
                _K.output.weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        /*PS: The same portion of momentum is always applied to the same GPU*/
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get full delta[_K.n_hiddens] from GPU[0]*/
    CUDA_G2G_CP(delta_ptr[_K.n_hiddens],_Kx.tmp_gpu,N,double);
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
        CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
            _K.output.weights+jdx*M*red,M*red,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
    ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
        _Kx.hiddens[_Kx.n_hiddens-1].vec,
        _Kx.dw[_Kx.n_hiddens]+jdx*M*red,_Kx.output.weights+jdx*M*red);
    CHK_ERR(moment_ger_dw_acc);
    /*3- transfer back weights to GPU[0]*/
    CUDA_G2G_SCP(_Kx.output.weights+jdx*M*red,
        _K.output.weights+jdx*M*red,M*(red+rem),
        double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
    /*TODO: stream copy version*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    CUDA_SET_DEV(*cudas,0);
    /*TRANSFER*/
    for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
        kx=_K.kerns[gpu];
        /*left part*/
        CUDA_G2G_CP(_K.output.weights,_Kx.output.weights,
            cudas->cuda_n_streams*gpu*M*red,double);
        /*right part*/
        if(gpu==cudas->n_gpu-1) continue;
        CUDA_G2G_CP(_K.output.weights,_Kx.output.weights,
            M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
    }
}
#endif /*_CUBLAS*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
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
            CUDA_SET_DEV(*cudas,gpu);
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
        CUDA_SET_DEV(*cudas,gpu);
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
        jdx=total_s-1;
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
        CUDA_SET_DEV(*cudas,0);
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
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            /*1- get full delta[idx] from GPU[0]*/
            CUDA_G2G_CP(delta_ptr[idx],_Kx.tmp_gpu,N,double);
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
                CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
                    _K.hiddens[idx].weights+jdx*M*red,M*red,
                    double,cudas->cuda_streams[jdx]);
                CHK_ERR(delta_transfer);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[idx] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[idx],_Kx.tmp_gpu,N,double);
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
            CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
                _K.hiddens[idx].weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
/*>>> last stream*/
        jdx=total_s-1;
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
        CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
            _K.hiddens[idx].weights+jdx*M*red,M*(red+rem),
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
        /*TODO: stream copy version*/
        /*before that we may need to sync all copies to GPU[0]*/
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
        }
        CUDA_SET_DEV(*cudas,0);
        /*TRANSFER*/
        for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
            kx=_K.kerns[gpu];
            /*left part*/
            CUDA_G2G_CP(_K.hiddens[idx].weights,_Kx.hiddens[idx].weights,
                cudas->cuda_n_streams*gpu*M*red,double);
            /*right part*/
            if(gpu==cudas->n_gpu-1) continue;
            CUDA_G2G_CP(_K.hiddens[idx].weights,_Kx.hiddens[idx].weights,
                M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
        }
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
        for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
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
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,moment,delta_ptr[idx]+jdx*red,
                _K.hiddens[idx-1].vec,_K.dw[idx]+jdx*M*red,
                _K.hiddens[idx].weights+jdx*M*red);
            CHK_ERR(moment_ger_dw_acc);
        }
/*>>> last stream*/
        jdx=total_s-1;
        ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (M,red+rem,LEARN_RATE,moment,delta_ptr[idx]+jdx*red,
            _K.hiddens[idx-1].vec,_K.dw[idx]+jdx*M*red,
            _K.hiddens[idx].weights+jdx*M*red);
        CHK_ERR(moment_ger_dw_acc);
}else{
/*>>> first GPU[0]*/
        CUDA_SET_DEV(*cudas,0);
        for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
            ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,moment,delta_ptr[idx]+jdx*red,
                _K.hiddens[idx-1].vec,_K.dw[idx]+jdx*M*red,
                _K.hiddens[idx].weights+jdx*M*red);
            CHK_ERR(moment_ger_dw_acc);
        }
/*>>> next GPUs but the last one*/
        for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            kx=_K.kerns[gpu];
            /*1- get full delta[idx] from GPU[0]*/
            CUDA_G2G_CP(delta_ptr[idx],_Kx.tmp_gpu,N,double);
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
                CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
                    _K.hiddens[idx].weights+jdx*M*red,M*red,
                    double,cudas->cuda_streams[jdx]);
                CHK_ERR(delta_transfer);
            }
        }
/*>>> last GPU*/
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[idx] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[idx],_Kx.tmp_gpu,N,double);
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
            CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
                _K.hiddens[idx].weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
/*>>> last stream*/
        jdx=total_s-1;
        ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
            (M,red+rem,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
            _K.hiddens[idx-1].vec,_K.dw[idx]+jdx*M*red,
            _K.hiddens[idx].weights+jdx*M*red);
        CHK_ERR(moment_ger_dw_acc);
        /*3- transfer back weights to GPU[0]*/
        CUDA_G2G_SCP(_Kx.hiddens[idx].weights+jdx*M*red,
            _K.hiddens[idx].weights+jdx*M*red,M*(red+rem),
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
        /*TODO: stream copy version*/
        /*before that we may need to sync all copies to GPU[0]*/
        for(gpu=1;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
        }
        CUDA_SET_DEV(*cudas,0);
        /*TRANSFER*/
        for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
            kx=_K.kerns[gpu];
            /*left part*/
            CUDA_G2G_CP(_K.hiddens[idx].weights,_Kx.hiddens[idx].weights,
                cudas->cuda_n_streams*gpu*M*red,double);
            /*right part*/
            if(gpu==cudas->n_gpu-1) continue;
            CUDA_G2G_CP(_K.hiddens[idx].weights,_Kx.hiddens[idx].weights,
                M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
        }
}
#endif /*_CUBLAS*/
        for(gpu=0;gpu<cudas->n_gpu;gpu++){
            CUDA_SET_DEV(*cudas,gpu);
            CUDA_SYNC();
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
        CUDA_SET_DEV(*cudas,gpu);
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
    CUDA_SET_DEV(*cudas,gpu);
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
    jdx=total_s-1;
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
    CUDA_SET_DEV(*cudas,0);
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
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[0] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[0],_Kx.tmp_gpu,N,double);
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
            CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
                _K.hiddens[0].weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get full delta[idx] from GPU[0]*/
    CUDA_G2G_CP(delta_ptr[idx],_Kx.tmp_gpu,N,double);
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
        CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
            _K.hiddens[0].weights+jdx*M*red,M*red,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
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
    CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
        _K.hiddens[0].weights+jdx*M*red,M*(red+rem),
        double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
    /*TODO: stream copy version*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    CUDA_SET_DEV(*cudas,0);
    /*TRANSFER*/
    for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
        kx=_K.kerns[gpu];
        /*left part*/
        CUDA_G2G_CP(_K.hiddens[0].weights,_Kx.hiddens[0].weights,
            cudas->cuda_n_streams*gpu*M*red,double);
        /*right part*/
        if(gpu==cudas->n_gpu-1) continue;
        CUDA_G2G_CP(_K.hiddens[0].weights,_Kx.hiddens[0].weights,
            M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
    }
}
#else  /*_CUBLAS*/
if((cudas->mem_model!=CUDA_MEM_EXP)||(cudas->n_gpu<2)){
/*>>> all GPU but last one*/
    for(gpu=0;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,moment,delta_ptr[0]+jdx*red,
                _K.in,_K.dw[0]+jdx*M*red,_K.hiddens[0].weights+jdx*M*red);
            CHK_ERR(moment_ger_dw_acc);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,moment,delta_ptr[0]+jdx*red,
            _K.in,_K.dw[0]+jdx*M*red,_K.hiddens[0].weights+jdx*M*red);
        CHK_ERR(moment_ger_dw_acc);
    }
/*>>> last stream*/
    jdx=total_s-1;
    ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,LEARN_RATE,moment,delta_ptr[0]+jdx*red,
        _K.in,_K.dw[0]+jdx*M*red,_K.hiddens[0].weights+jdx*M*red);
    CHK_ERR(moment_ger_dw_acc);
}else{
/*>>> first GPU[0]*/
    CUDA_SET_DEV(*cudas,0);
    for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
        ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,moment,delta_ptr[0]+jdx*red,
            _K.in,_K.dw[0]+jdx*M*red,_K.hiddens[0].weights+jdx*M*red);
        CHK_ERR(moment_ger_dw_acc);
    }
/*>>> next GPUs but the last one*/
    for(gpu=1;gpu<cudas->n_gpu-1;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        kx=_K.kerns[gpu];
        /*1- get full delta[0] from GPU[0]*/
        CUDA_G2G_CP(delta_ptr[0],_Kx.tmp_gpu,N,double);
        /*we don't need to sync (I think)*/
        for(kdx=0;kdx<cudas->cuda_n_streams;kdx++){
            jdx=kdx+gpu*(cudas->cuda_n_streams);
            /*2- calculate weights (vec, dw are GPU-local)*/
            ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
                (M,red,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
                _Kx.in,_Kx.dw[0]+jdx*M*red,_Kx.hiddens[0].weights+jdx*M*red);
            CHK_ERR(moment_ger_dw_acc);
            /*3- transfer back weights to GPU[0]*/
            CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
                _K.hiddens[0].weights+jdx*M*red,M*red,
                double,cudas->cuda_streams[jdx]);
            CHK_ERR(delta_transfer);
        }
    }
/*>>> last GPU*/
    CUDA_SET_DEV(*cudas,gpu);
    kx=_K.kerns[gpu];
    /*1- get full delta[idx] from GPU[0]*/
    CUDA_G2G_CP(delta_ptr[idx],_Kx.tmp_gpu,N,double);
    /*no sync needed (I think)*/
    for(kdx=0;kdx<cudas->cuda_n_streams-1;kdx++){
        jdx=kdx+gpu*(cudas->cuda_n_streams);
        /*2- calculate weights (vec is GPU-local)*/
        ger_dw_acc<<<_KG(red),0,cudas->cuda_streams[jdx]>>>
            (M,red,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
            _Kx.in,_Kx.dw[0]+jdx*M*red,_Kx.hiddens[0].weights+jdx*M*red);
        CHK_ERR(moment_ger_dw_acc);
        /*3- transfer back weights to GPU[0]*/
        CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
            _K.hiddens[0].weights+jdx*M*red,M*red,
            double,cudas->cuda_streams[jdx]);
        CHK_ERR(delta_transfer);
    }
/*>>> last stream*/
    jdx=total_s-1;
    ger_dw_acc<<<_KG(red+rem),0,cudas->cuda_streams[jdx]>>>
        (M,red+rem,LEARN_RATE,moment,_Kx.tmp_gpu+jdx*red,
         _Kx.in,_Kx.dw[0]+jdx*M*red,_Kx.hiddens[0].weights+jdx*M*red);
    CHK_ERR(moment_ger_dw_acc);
    /*3- transfer back weights to GPU[0]*/
    CUDA_G2G_SCP(_Kx.hiddens[0].weights+jdx*M*red,
        _K.hiddens[0].weights+jdx*M*red,M*(red+rem),
        double,cudas->cuda_streams[jdx]);
    CHK_ERR(delta_transfer);
/*>>> put back weights from GPU[0] to all GPUs*/
    /*TODO: stream copy version*/
    /*before that we may need to sync all copies to GPU[0]*/
    for(gpu=1;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
    CUDA_SET_DEV(*cudas,0);
    /*TRANSFER*/
    for(gpu=cudas->n_gpu-1;gpu>0;gpu--){
        kx=_K.kerns[gpu];
        /*left part*/
        CUDA_G2G_CP(_K.hiddens[0].weights,_Kx.hiddens[0].weights,
            cudas->cuda_n_streams*gpu*M*red,double);
        /*right part*/
        if(gpu==cudas->n_gpu-1) continue;
        CUDA_G2G_CP(_K.hiddens[0].weights,_Kx.hiddens[0].weights,
            M*(N-red*cudas->cuda_n_streams*(1+gpu)),double);
    }
}
#endif /*_CUBLAS*/
    for(gpu=0;gpu<cudas->n_gpu;gpu++){
        CUDA_SET_DEV(*cudas,gpu);
        CUDA_SYNC();
    }
/*+++ IV - update error +++*/
    /*update kernel*/
    scuda_ann_forward(kernel,cudas);
    Epr=scuda_ann_error(kernel,train,cudas);
//  fprintf(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ V - cleanup +++*/
    CUDA_SET_DEV(*cudas,0);/*make sure all de-allocation happen on gpu[0]*/
    for(idx=0;idx<(_K.n_hiddens+1);idx++){
        CUDA_FREE(delta_ptr[idx]);
        delta_ptr[idx]=NULL;
    }
    FREE(delta_ptr);
    CHK_ERR(free_1);
    return Ep-Epr;
}

}/*extern "C"*/
