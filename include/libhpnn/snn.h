/*
+++ libhpnn - High Performance Neural Network library - file: snn.h +++
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
#ifndef SNN_H
#define SNN_H

/*SNN (= ANN + SOFTMAX) uses the same _kernel type as ANN*/

/*functions*/
void snn_kernel_run(kernel_ann *kernel);
DOUBLE snn_kernel_train(kernel_ann *kernel,const DOUBLE *train);
DOUBLE snn_kernel_train_momentum(kernel_ann *kernel,const DOUBLE *train,DOUBLE alpha);
DOUBLE snn_train_BP(kernel_ann *kernel,DOUBLE *train_in,DOUBLE *train_out,DOUBLE delta);
DOUBLE snn_train_BPM(kernel_ann *kernel,DOUBLE *train_in,DOUBLE *train_out,DOUBLE alpha,DOUBLE delta);




#endif /*SNN_H*/
