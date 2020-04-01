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
    nn_cap capability;
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
    NN_TYPE_ANN = 0,    /*feed-forward, activation base nn*/
    NN_TYPE_LNN = 1,    /*feed-forward, activation (hidden) + linear (output)*/
    NN_TYPE_SNN = 2,    /*NN_TYPE_ANN + softmax*/
    NN_TYPE_UKN =-1,    /*unknown*/
} nn_type;
/*-------------------------*/
/*+++ types of training +++*/
/*-------------------------*/
typedef enum {
    NN_TRAIN_BP  = 0,   /*simple back-propagation*/
    NN_TRAIN_BPM = 1,   /*back-propagation with momentum*/
    NN_TRAIN_CG  = 2,   /*conjugate gradients*/
    NN_TRAIN_SPLX =3,   /*simplex optimization*/
    NN_TRAIN_UKN =-1,   /*unknown*/
} nn_train;
#define MIN_BP_ITER 31
#define MAX_BP_ITER 10239
#define MIN_BPM_ITER 31
#define MAX_BPM_ITER 10239
/*-----------------------------*/
/*+++ NN definition handler +++*/
/*-----------------------------*/
typedef struct {
    nn_runtime *rr;     /*each NN should link to the runtime parameters*/
    CHAR     *name;     /*name of the NN*/
    nn_type   type;     /*NN type*/
    BOOL need_init;     /*require initialization*/
    UINT      seed;     /*seed used in case of initialization*/
    void   *kernel;     /*NN kernel*/
    CHAR *f_kernel;     /*kernel filename*/
    nn_train train;     /*training type*/
    CHAR  *samples;     /*samples directory (for training)*/
    CHAR    *tests;     /*tests directory (for validation)*/
} nn_def;
/*------------------*/
/*+++ NN methods +++*/
/*------------------*/
#define _NN(a,b) nn_##a##_##b
/*new defs to take into account verbosity*/
#define NN_DBG(_file,...) do{\
    if((_NN(return,verbose)())>2){\
        _OUT((_file),"NN(DBG): ");\
        _OUT((_file), __VA_ARGS__);\
    }\
}while(0)
#define NN_OUT(_file,...) do{\
    if((_NN(return,verbose)())>1){\
        _OUT((_file),"NN: ");\
        _OUT((_file), __VA_ARGS__);\
    }\
}while(0)
#define NN_COUT(_file,...) do{\
        if((_NN(return,verbose)())>1){\
                _OUT((_file), __VA_ARGS__);\
        }\
}while(0)
#define NN_WARN(_file,...) do{\
    if((_NN(return,verbose)())>0){\
        _OUT((_file),"NN(WARN): ");\
        _OUT((_file), __VA_ARGS__);\
    }\
}while(0)
#define NN_ERROR(_file,...) do{\
    _OUT((_file),"NN(ERR): ");\
    _OUT((_file), __VA_ARGS__);\
}while(0)
#define NN_WRITE _OUT
/*--------------------------*/
/*+++ initialize library +++*/
/*--------------------------*/
void _NN(inc,verbose)();
void _NN(dec,verbose)();
void _NN(set,verbose)(SHORT verbosity);
void _NN(get,verbose)(SHORT *verbosity);
SHORT _NN(return,verbose)();
void _NN(toggle,dry)();
void _NN(get,capabilities)(nn_cap *capabilities);
void _NN(unset,capability)(nn_cap capability);
nn_cap _NN(return,capabilities)();
BOOL _NN(init,OMP)();
BOOL _NN(init,MPI)();
BOOL _NN(init,CUDA)();
BOOL _NN(init,BLAS)();
int _NN(init,all)(UINT init_verbose);
BOOL _NN(deinit,OMP)();
BOOL _NN(deinit,MPI)();
BOOL _NN(deinit,CUDA)();
BOOL _NN(deinit,BLAS)();
int  _NN(deinit,all)();
/*------------------------------*/
/*+++ set/get lib parameters +++*/
/*------------------------------*/
BOOL _NN(set,omp_threads)(UINT n_threads);
BOOL _NN(get,omp_threads)(UINT *n_threads);
int _NN(return,omp_threads)();
/*TODO: we might define a MPI_COMM_xxx for the next 3 functions*/
BOOL _NN(set,mpi_tasks)(UINT n_tasks);
BOOL _NN(get,mpi_tasks)(UINT *n_tasks);
BOOL _NN(get,curr_mpi_task)(UINT *task);
BOOL _NN(set,n_gpu)(UINT n_gpu);
BOOL _NN(get,n_gpu)(UINT *n_gpu);
BOOL _NN(set,cuda_streams)(UINT n_streams);
BOOL _NN(get,cuda_streams)(UINT *n_streams);
BOOL _NN(set,omp_blas)(UINT n_blas);
BOOL _NN(get,omp_blas)(UINT *n_blas);
cudastreams *_NN(return,cudas)();
/*---------------------*/
/*+++ configuration +++*/
/*---------------------*/
void _NN(init,conf)(nn_def *conf);
void _NN(deinit,conf)(nn_def *conf);
void _NN(set,name)(nn_def *conf,const CHAR *name);
void _NN(get,name)(nn_def *conf,CHAR **name);
char *_NN(return,name)(nn_def *conf);
void _NN(set,type)(nn_def *conf,nn_type type);
void _NN(get,type)(nn_def *conf,nn_type *type);
nn_type _NN(return,type)(nn_def *conf);
void _NN(set,need_init)(nn_def *conf,BOOL need_init);
void _NN(get,need_init)(nn_def *conf,BOOL *need_init);
BOOL _NN(return,need_init)(nn_def *conf);
void _NN(set,seed)(nn_def *conf,UINT seed);
void _NN(get,seed)(nn_def *conf,UINT *seed);
UINT _NN(return,seed)(nn_def *conf);
void _NN(set,kernel_filename)(nn_def *conf,CHAR *f_kernel);
void _NN(get,kernel_filename)(nn_def *conf,CHAR **f_kernel);
char *_NN(return,kernel_filename)(nn_def *conf);
void _NN(set,train)(nn_def *conf,nn_train train);
void _NN(get,train)(nn_def *conf,nn_train *train);
nn_train _NN(return,train)(nn_def *conf);
void _NN(set,samples_directory)(nn_def *conf,CHAR *samples);
void _NN(get,samples_directory)(nn_def *conf,CHAR **samples);
char *_NN(return,samples_directory)(nn_def *conf);
void _NN(set,tests_directory)(nn_def *conf,CHAR *tests);
void _NN(get,tests_directory)(nn_def *conf,CHAR **tests);
char *_NN(return,tests_directory)(nn_def *conf);
nn_def *_NN(load,conf)(const CHAR *filename);
void _NN(dump,conf)(nn_def *conf,FILE *fp);
/*----------------------------*/
/*+++ manipulate NN kernel +++*/
/*----------------------------*/
void _NN(free,kernel)(nn_def *conf);
BOOL _NN(generate,kernel)(nn_def *conf,...);
BOOL _NN(load,kernel)(nn_def *conf);
void _NN(dump,kernel)(nn_def *conf, FILE *output);
/*----------------------------*/
/*+++ Access NN parameters +++*/
/*----------------------------*/
UINT _NN(get,n_inputs)(nn_def *conf);
UINT _NN(get,n_hiddens)(nn_def *conf);
UINT _NN(get,n_outputs)(nn_def *conf);
UINT _NN(get,h_neurons)(nn_def *conf,UINT layer);
/*------------------*/
/*+++ sample I/O +++*/
/*------------------*/
BOOL _NN(read,sample)(CHAR *filename,DOUBLE **in,DOUBLE **out);
/*---------------------*/
/*+++ execute NN OP +++*/
/*---------------------*/
BOOL _NN(train,kernel)(nn_def *conf);
void _NN(run,kernel)(nn_def *conf);


#endif/*LIBHPNN_H*/
