/*
+++ libhpnn - High Performance Neural Network library - file: cuda_snn.h +++
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
#ifndef CUDA_SNN_H
#define CUDA_SNN_H

__global__ void fw_smax(int n, double dv, double *out);
__global__ void amb_smax(int n, double *res, double *train, double *out);
__global__ void fw_s_acc(int m,int n, double *mat,double *vec,double *res);
__global__ void amb_smax_acc(int n, double *res, double *train, double *out);
__global__ void dv_acc(int n,double *res,double *out);
__global__ void dsmax_diff(int n, double *t, double *o, double *y);


#if __cplusplus
extern "C" {
#endif
void scuda_snn_forward(kernel_ann *kernel,cudastreams *cudas);
double scuda_snn_error(kernel_ann *kernel,double *train,cudastreams *cudas);
double scuda_snn_train(kernel_ann *kernel,double *train,cudastreams *cudas);
double scuda_snn_train_momentum(kernel_ann *kernel,double *train,double moment,
    cudastreams *cudas);
#if __cplusplus
}
#endif

#endif /*CUDA_SNN_H*/
