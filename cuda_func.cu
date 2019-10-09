#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common.h"
#include "ann.h"

/*This is a CUDA area for functions*/

#define THREADS_PER_BLOCK 128


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
//		printf("[%d, %d]:\ta=%lf\tb=%lf\tout=%lf\n",\
            blockIdx.y*gridDim.x+blockIdx.x,\
            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
            a[i],b[i],out[i]);
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
        double *gpu_out;
	double _alpha=1.0;
	double _beta =0.0;
        gpu_in=_K.cuda_in;/*no copy?*/
/*+++ I - hiddens +++*/
        for(idx=0;idx<_K.n_hiddens;idx++){
                CUDA_ALLOC(gpu_out,_K.hiddens[idx].n_neurons,DOUBLE);
                CHK_ERR(alloc_1);
                /*GEMV + act*/
                N=_K.hiddens[idx].n_neurons;
                M=_K.hiddens[idx].n_inputs;
		cublasDgemv(cublas_handle,CUBLAS_OP_T,M,N,&_alpha,_K.hiddens[idx].cuda_w,M,gpu_in,1,&_beta,gpu_out,1);
		CHK_ERR(cublas_1);
		sigmoid<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,gpu_out);
                CHK_ERR(kernel_1);
                if(idx>0) CUDA_FREE(gpu_in);/*do not free the original input*/
                CHK_ERR(free_1);
                gpu_in=gpu_out;
        }
//M=_K.output.n_inputs;
//dbg_print<<<(M+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M,gpu_in);
/*+++ II - output +++*/
        N=_K.output.n_neurons;
        M=_K.output.n_inputs;
	cublasDgemv(cublas_handle,CUBLAS_OP_T,M,N,&_alpha,_K.output.cuda_w,M,gpu_in,1,&_beta,_K.cuda_out,1);
	CHK_ERR(cublas_2);
	sigmoid<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,_K.cuda_out);
        CHK_ERR(kernel_2);
        CUDA_FREE(gpu_in);
        CHK_ERR(free_2);
//      cudaDeviceSynchronize();
}
void scuda_ann_forward_cublas(_kernel *kernel,cudastreams *cudas){
	int idx,jdx;
	int M,N,red;
	int max,rem;
	double *gpu_in;
	double *gpu_out;
//	double *gpu_stream[cudas->cuda_n_streams];
        double _alpha=1.0;
        double _beta =0.0;
/*+++ I - hiddens +++*/
#define _GD 8
#define _BD 512
	/*get max*/
	max=_K.output.n_neurons;
	for(idx=0;idx<_K.n_hiddens;idx++)
		if(_K.hiddens[idx].n_neurons>max) max=_K.hiddens[idx].n_neurons;
	if(_K.n_inputs>max)
		CUDA_ALLOC(gpu_in,_K.n_inputs,DOUBLE);
	else
		CUDA_ALLOC(gpu_in,max,DOUBLE);
	CUDA_G2G_CP(_K.cuda_in,gpu_in,_K.n_inputs,DOUBLE);
	CUDA_ALLOC(gpu_out,max,DOUBLE);
	for(idx=0;idx<_K.n_hiddens;idx++){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		red=N/cudas->cuda_n_streams;
		rem=N%cudas->cuda_n_streams;
		for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
			fw_mv_acc<<<_GD,_BD,0,cudas->cuda_streams[jdx]>>>
				(M,red,_K.hiddens[idx].cuda_w+jdx*M*red,gpu_in,gpu_out+jdx*red);
			CHK_ERR(kernel_1);
		}
		if(rem>0){
			/*the following _should_ force a sync*/
			jdx=cudas->cuda_n_streams;
			fw_mv_acc<<<(rem+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>
				(M,rem,_K.hiddens[idx].cuda_w+jdx*M*red,gpu_in,gpu_out+jdx*red);
			CHK_ERR(kernel_1);
		}
		else cudaDeviceSynchronize();
//cudaDeviceSynchronize();
		/*now copy back gpu_out to gpu_in*/
		CUDA_G2G_CP(gpu_out,gpu_in,N,DOUBLE);
		CHK_ERR(sync_1);
	}
//M=_K.output.n_inputs;
//dbg_print<<<(M+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M,gpu_in);
/*+++ II - output +++*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	red=N/cudas->cuda_n_streams;
	rem=N%cudas->cuda_n_streams;
	for(jdx=0;jdx<cudas->cuda_n_streams;jdx++){
		fw_mv_acc<<<_GD,_BD,0,cudas->cuda_streams[jdx]>>>
			(M,red,_K.output.cuda_w+jdx*M*red,gpu_in,_K.cuda_out+jdx*red);
		CHK_ERR(kernel_2);
	}
	if(rem>0){
		jdx=cudas->cuda_n_streams;
		fw_mv_acc<<<(rem+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>
			(M,rem,_K.output.cuda_w+jdx*M*red,gpu_in,_K.cuda_out+jdx*red);
		CHK_ERR(kernel_2);
	}else cudaDeviceSynchronize();
//cudaDeviceSynchronize();
	CHK_ERR(sync_2);
	CUDA_FREE(gpu_in);
	CUDA_FREE(gpu_out);
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
//	dbg_print<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,hidden_vector_ptr[0]);
	CHK_ERR(cublas_1);
	sigmoid<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,hidden_vector_ptr[0]);
	CHK_ERR(kernel_1);
/*^^^ hidden to hidden (if any)*/
	for(idx=1;idx<_K.n_hiddens;idx++){
		N=_K.hiddens[idx].n_neurons;
		M=_K.hiddens[idx].n_inputs;
		CUBLAS_ERR(cublasDgemv(cublas_handle,CUBLAS_OP_T,M,N,&_alpha,_K.hiddens[idx].cuda_w,M,hidden_vector_ptr[idx-1],1,&_beta,hidden_vector_ptr[idx],1));
		CHK_ERR(cublas_2);
		sigmoid<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,hidden_vector_ptr[idx]);
		CHK_ERR(kernel_2);
	}
/*^^^ hidden to output*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	CUBLAS_ERR(cublasDgemv(cublas_handle,CUBLAS_OP_T,M,N,&_alpha,_K.output.cuda_w,M,hidden_vector_ptr[_K.n_hiddens-1],1,&_beta,_K.cuda_out,1));
	CHK_ERR(cublas_3);
	sigmoid<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,_K.cuda_out);
	CHK_ERR(kernel_3);
	/*all done, calculate a preliminary error*/
	N=_K.n_outputs;
	amb<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,tmp_gpu,train,_K.cuda_out);
	CHK_ERR(kernel_4);
	cublasDasum(cublas_handle,N,tmp_gpu,1,&Ep);
	CHK_ERR(cublas_4);
	//cudaDeviceSynchronize();
	Ep*=0.5;
//	printf("TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - DELTAS +++*/
/*^^^ output*/
	N=_K.n_outputs;
	_dsigmoid<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,_K.cuda_out,delta_ptr[_K.n_hiddens]);
	CHK_ERR(kernel_5);
	mul_diff<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,train,_K.cuda_out,delta_ptr[_K.n_hiddens]);
	CHK_ERR(kernel_6);
/*^^^ output to hidden*/
	/*! transposed (of the transposed might be transposed)*/
	N=_K.output.n_neurons;
	M=_K.output.n_inputs;
	cublasDgemv(cublas_handle,CUBLAS_OP_N,M,N,&_alpha,_K.output.cuda_w,M,delta_ptr[_K.n_hiddens],1,&_beta,delta_ptr[_K.n_hiddens-1],1);
	CHK_ERR(cublas_5);
	dsigmoid<<<(M+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M,hidden_vector_ptr[_K.n_hiddens-1],delta_ptr[_K.n_hiddens-1]);
	CHK_ERR(kernel_7);
//cudaDeviceSynchronize();printf("GPU\n");
//dbg_print<<<(M+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M,delta_ptr[_K.n_hiddens-1]);
/*^^^ hidden to hidden (if any)*/
	if(_K.n_hiddens>1){
		for(idx=(_K.n_hiddens-2);idx>0;idx--){
			/*! transposed (of the transposed might be transposed)*/
			N=_K.hiddens[idx+1].n_neurons;
			M=_K.hiddens[idx+1].n_inputs;
			cublasDgemv(cublas_handle,CUBLAS_OP_N,M,N,&_alpha,_K.hiddens[idx+1].cuda_w,M,delta_ptr[idx+1],1,&_beta,delta_ptr[idx],1);
			CHK_ERR(cublas_6);
			dsigmoid<<<(M+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M,hidden_vector_ptr[idx],delta_ptr[idx]);
			CHK_ERR(kernel_8);
		}
		/*add zero*/
		/*! transposed (of the transposed might be transposed)*/
		N=_K.hiddens[1].n_neurons;
		M=_K.hiddens[1].n_inputs;
		cublasDgemv(cublas_handle,CUBLAS_OP_N,M,N,&_alpha,_K.hiddens[1].cuda_w,M,delta_ptr[1],1,&_beta,delta_ptr[0],1);
		CHK_ERR(cublas_7);
		dsigmoid<<<(M+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M,hidden_vector_ptr[0],delta_ptr[0]);
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
	amb<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,tmp_gpu,train,_K.cuda_out);
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