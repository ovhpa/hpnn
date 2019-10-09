#ifndef FUNC_H
#define FUNC_H

double cuda_array_dbg(cublasHandle_t cublas_handle,int n,double *gpu_in);

void cuda_ann_forward_cublas(_kernel *kernel,cublasHandle_t cublas_handle);
void scuda_ann_forward_cublas(_kernel *kernel,cudastreams *cudas);

double cuda_ann_train_cublas(_kernel *kernel,double *train,cudastreams *cudas);

void cuda_ann_act(double *out,int size);
void cuda_ann_dact(double *in,double *out,int size);
void cuda_ann_amb(double *out, double *a,double *b,int size);
void cuda_ann_mul_diff(double *train, double *out, double *res, int size);

void cuda_zero_mv(int m,int n,double *mat,double *vec, double *res);
void cuda_zero_tmv(int m,int n,double *mat,double *vec, double *res);

#endif /*FUNC_H*/
