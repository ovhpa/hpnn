/*
+++ libhpnn - High Performance Neural Network library - file: cuda_ann.h +++
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
#ifndef CUDA_ANN_H
#define CUDA_ANN_H

#if __cplusplus
extern "C" {
#endif

void scuda_ann_allocate(_kernel *kernel,cudastreams *cudas);
void scuda_ann_weights_C2G(_kernel *kernel,cudastreams *cudas);
void scuda_ann_weights_G2C(_kernel *kernel,cudastreams *cudas);

void scuda_ann_forward(_kernel *kernel,cudastreams *cudas);
double scuda_ann_error(_kernel *kernel,double *train,cudastreams *cudas);
double scuda_ann_train(_kernel *kernel,double *train,cudastreams *cudas);
void scuda_ann_raz_momentum(_kernel *kernel,cudastreams *cudas);
double scuda_ann_train_momentum(_kernel *kernel,double *train,double moment,cudastreams *cudas);

#if __cplusplus
}
#endif

#endif /*CUDA_ANN_H*/
