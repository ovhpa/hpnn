/*
+++ libhpnn - High Performance Neural Network library - file: libhpnn.h +++
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
#ifndef LIBHPNN_H
#define LIBHPNN_H
#include <libhpnn/common.h>
/*----------------------------*/
/*+++ library capabilities +++*/
/*----------------------------*/
typedef enum {
	NN_CAP_NONE=0,
	NN_CAP_OMP=(1<<0),
	NN_CAP_MPI=(1<<1),
	NN_CAP_CUDA=(1<<2),
	NN_CAP_CUBLAS=(1<<3),
	/*(1<<4) is reserved for OCL*/
	NN_CAP_PBLAS=(1<<5),
	NN_CAP_SBLAS=(1<<6),
} nn_cap;
/*----------------------------------*/
/*+++ library runtime parameters +++*/
/*----------------------------------*/
typedef struct {
	SHORT nn_verbose;
	BOOL  nn_dry;
	UINT  nn_num_threads;
	UINT  nn_num_blas;
	UINT  nn_num_tasks;
	cudastreams cudas;
} nn_runtime;
/*--------------------------------*/
/*+++ types of neural networks +++*/
/*--------------------------------*/
typedef enum {
	NN_TYPE_ANN = 0,	/*feed-forward, activation base nn*/
	NN_TYPE_LNN = 1,	/*feed-forward, activation (hidden) + linear (output)*/
	NN_TYPE_PNN = 2,	/*feed-forward, probability base nn*/
	NN_TYPE_UKN =-1,	/*unknown*/
} nn_type;
/*-------------------------*/
/*+++ types of training +++*/
/*-------------------------*/
typedef enum {
	NN_TRAIN_BP  = 0,	/*simple back-propagation*/
	NN_TRAIN_BPM = 1,	/*back-propagation with momentum*/
	NN_TRAIN_CG  = 2,	/*conjugate gradients*/
	NN_TRAIN_UKN =-1,	/*unknown*/
} nn_train;
/*-----------------------------*/
/*+++ NN definition handler +++*/
/*-----------------------------*/
typedef struct {
	nn_runtime *rr;		/*each NN should link to the runtime parameters*/
	CHAR     *name;		/*name of the NN*/
	nn_type   type;		/*NN type*/
	BOOL need_init;		/*require initialization*/
	UINT      seed;		/*seed used in case of initialization*/
	void   *kernel;		/*NN kernel*/
	CHAR *f_kernel;		/*kernel filename*/
	nn_train train;		/*training type*/
	CHAR  *samples;		/*samples directory (for training)*/
	CHAR    *tests;		/*tests directory (for validation)*/
} nn_def;
/*------------------*/
/*+++ NN methods +++*/
/*------------------*/
#define _NN(a,b) nn_##a##_##b
/*--------------------------*/
/*+++ initialize library +++*/
/*--------------------------*/
void _NN(inc,verbose)();
void _NN(dec,verbose)();
void _NN(set,verbose)(SHORT verbosity);
void _NN(get,verbose)(SHORT *verbosity);
void _NN(toggle,dry)();
nn_cap _NN(get,capabilities)();
BOOL _NN(init,OMP)();
BOOL _NN(init,MPI)();
BOOL _NN(init,CUDA)();
BOOL _NN(init,BLAS)();
int  _NN(init,all)();
BOOL _NN(deinit,OMP)();
BOOL _NN(deinit,MPI)();
BOOL _NN(deinit,CUDA)();
BOOL _NN(deinit,BLAS)();
int  _NN(deinit,all)();
/*--------------------------*/
/*+++ set/get parameters +++*/
/*--------------------------*/
BOOL _NN(set,omp_threads)(UINT n_threads);
BOOL _NN(get,omp_threads)(UINT *n_threads);
BOOL _NN(set,mpi_tasks)(UINT n_tasks);
BOOL _NN(get,mpi_tasks)(UINT *n_tasks);
BOOL _NN(set,cuda_streams)(UINT n_streams);
BOOL _NN(get,cuda_streams)(UINT *n_streams);
BOOL _NN(set,omp_blas)(UINT n_blas);
BOOL _NN(get,omp_blas)(UINT *n_blas);
cudastreams *_NN(get,cudas)();
/*---------------------*/
/*+++ configuration +++*/
/*---------------------*/
nn_def *_NN(conf,load)(CHAR *filename);
void _NN(conf,dump)(FILE *fp,nn_def *neural);
/*----------------------------*/
/*+++ Access NN parameters +++*/
/*----------------------------*/
UINT _NN(get,n_inputs)(nn_def *neural);
UINT _NN(get,n_hiddens)(nn_def *neural);
UINT _NN(get,n_outputs)(nn_def *neural);
UINT _NN(get,h_neurons)(nn_def *neural,UINT layer);
/*---------------------*/
/*+++ manipulate NN +++*/
/*---------------------*/
BOOL _NN(kernel,generate)(nn_def *neural,UINT n_inputs,UINT n_hiddens,
							UINT n_outputs,UINT *hiddens);
BOOL _NN(kernel,load)(nn_def *neural);
void _NN(kernel,dump)(nn_def *neural, FILE *output);
BOOL _NN(kernel,train)(nn_def *neural);
void _NN(kernel,run)(nn_def *neural);

#endif/*LIBHPNN_H*/
