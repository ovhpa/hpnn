/*
+++ libhpnn - High Performance Neural Network library - file: snn.c +++
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
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
/*^^^  MPI specific*/
#ifdef _MPI
#include <mpi.h>
#endif
/*^^^ BLAS/MKL specific*/
#if defined (PBLAS) || defined (SBLAS)
#ifndef _MKL
#include <cblas.h>
#else /*_MKL*/
#include <mkl.h>
#include <mkl_cblas.h>
#endif /*_MKL*/
#endif /*PBLAS*/
/*^^^ CUDA specific*/
#ifdef _CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifdef DEBUG
#include <cuda.h>
#endif /*DEBUG*/
#endif
/*^^^ OMP specific*/
#ifdef _OMP
#include <omp.h>
#endif
/*link to the main library*/
#include <libhpnn.h>
#include <libhpnn/ann.h>
#ifdef _CUDA
#include <libhpnn/cuda_ann.h>
#include <libhpnn/cuda_snn.h>
#endif /*_CUDA*/
#include <libhpnn/snn.h>
/*----------------------*/
/*+++ useful defines +++*/
/*----------------------*/
/*^^^ MKL specific*/
#ifdef _MKL
#define _HT mkl_set_num_threads_local(_NN(get,omp_blas)())
#else
#define _HT
#endif
/*^^^ OMP specific*/
#ifdef _OMP
#define _NT num_threads(_NN(return,omp_threads)())
#else
#define _NT
#endif
/*make life easier*/
#define KERN (*kernel)
/*------------------------*/
/*+++ feed-forward run +++*/
/*------------------------*/
void snn_kernel_run(kernel_ann *kernel){
#ifdef   _CUDA
    /*the _NN(run,kernel) is now in charge of transfer(s)*/
    scuda_snn_forward(kernel,_NN(return,cudas)());
#else  /*_CUDA*/
    /*simple, one pass kernel*/
    UINT idx,jdx,M,N;
    DOUBLE dv;
#if !defined (PBLAS) && !defined (SBLAS)
    UINT kdx;
#endif
#ifdef _MPI
    UINT n_streams,stream;
    UINT red,rem;
    _NN(get,mpi_tasks)(&n_streams);
    _NN(get,curr_mpi_task)(&stream);
#endif /*_MPI*/
    /*simple, one pass kernel*/
/*+++ I - input +++*/
    N=KERN.hiddens[0].n_neurons;
    M=KERN.hiddens[0].n_inputs;
#ifdef _MPI
    red=N/n_streams;
    rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
    cblas_dgemv(CblasRowMajor,CblasNoTrans,red,M,
        1.0,KERN.hiddens[0].weights+stream*M*red,M,KERN.in,1,0.,KERN.hiddens[0].vec+stream*red,1);
#define OP_ACT(ix) KERN.hiddens[0].vec[ix+stream*red]=ann_act(KERN.hiddens[0].vec[ix+stream*red])
    UNROLL_OMP_FOR(0,red,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
    /*do the remaining ops without MPI*/
    if(rem>0){
        cblas_dgemv(CblasRowMajor,CblasNoTrans,rem,M,
            1.0,KERN.hiddens[0].weights+n_streams*M*red,M,KERN.in,1,0.,KERN.hiddens[0].vec+n_streams*red,1);
#define OP_ACT(ix) KERN.hiddens[0].vec[ix+n_streams*red]=ann_act(KERN.hiddens[0].vec[ix+n_streams*red])
        UNROLL_OMP_FOR(0,rem,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
    }
#else /*_MPI*/
    cblas_dgemv(CblasRowMajor,CblasNoTrans,N,M,1.0,KERN.hiddens[0].weights,M,KERN.in,1,0.,KERN.hiddens[0].vec,1);
#define OP_ACT(ix) KERN.hiddens[0].vec[ix]=ann_act(KERN.hiddens[0].vec[ix])
    UNROLL_OMP_FOR(0,N,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
#endif /*_MPI*/
#elif defined(SBLAS)
    /*move the parallel mv into a series of vv*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<red;jdx++){
_HT;
        KERN.hiddens[0].vec[jdx+stream*red]=cblas_ddot(
        M,&(KERN.hiddens[0].weights[M*(jdx+stream*red)]),1,KERN.in,1);
        KERN.hiddens[0].vec[jdx+stream*red]=ann_act(KERN.hiddens[0].vec[jdx+stream*red]);
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
if(rem>0){
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<rem;jdx++){
_HT;
        KERN.hiddens[0].vec[jdx+n_streams*red]=cblas_ddot(
        M,&(KERN.hiddens[0].weights[M*(jdx+n_streams*red)]),1,KERN.in,1);
        KERN.hiddens[0].vec[jdx+n_streams*red]=ann_act(KERN.hiddens[0].vec[jdx+n_streams*red]);
    }
}
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<N;jdx++){
_HT;
        KERN.hiddens[0].vec[jdx]=cblas_ddot(
        M,&(KERN.hiddens[0].weights[_2D_IDX(M,jdx,0)]),1,KERN.in,1);
        KERN.hiddens[0].vec[jdx]=ann_act(KERN.hiddens[0].vec[jdx]);
    }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
    for(jdx=0;jdx<red;jdx++){
        KERN.hiddens[0].vec[jdx+stream*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[0].vec[jdx+stream*red]+=KERN.hiddens[0].weights[M*(jdx+stream*red)+ix]*KERN.in[ix]
        UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
        KERN.hiddens[0].vec[jdx+stream*red]=ann_act(KERN.hiddens[0].vec[jdx+stream*red]);
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
        for(jdx=0;jdx<rem;jdx++){
            KERN.hiddens[0].vec[jdx+n_streams*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[0].vec[jdx+n_streams*red]+=KERN.hiddens[0].weights[M*(jdx+n_streams*red)+ix]*KERN.in[ix]
            UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
            KERN.hiddens[0].vec[jdx+n_streams*red]=ann_act(KERN.hiddens[0].vec[jdx+n_streams*red]);
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
    for(jdx=0;jdx<N;jdx++){
        KERN.hiddens[0].vec[jdx]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[0].vec[jdx]+=KERN.hiddens[0].weights[_2D_IDX(M,jdx,ix)]*KERN.in[ix]
        UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
        KERN.hiddens[0].vec[jdx]=ann_act(KERN.hiddens[0].vec[jdx]);
    }
#endif /*_MPI*/
#endif /*PBLAS*/
/*+++ II - hiddens +++*/
    for(idx=1;idx<KERN.n_hiddens;idx++){
        N=KERN.hiddens[idx].n_neurons;
        M=KERN.hiddens[idx].n_inputs;
#ifdef _MPI
        red=N/n_streams;
        rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
        cblas_dgemv(CblasRowMajor,CblasNoTrans,red,M,
        1.0,KERN.hiddens[idx].weights+stream*M*red,M,KERN.hiddens[idx-1].vec,1,0.,KERN.hiddens[idx].vec+stream*red,1);
#define OP_ACT(ix) KERN.hiddens[idx].vec[ix+stream*red]=ann_act(KERN.hiddens[idx].vec[ix+stream*red])
        UNROLL_OMP_FOR(0,red,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
        MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
        if(rem>0){
            cblas_dgemv(CblasRowMajor,CblasNoTrans,rem,M,
            1.0,KERN.hiddens[idx].weights+n_streams*M*red,M,KERN.hiddens[idx-1].vec,1,0.,KERN.hiddens[idx].vec+n_streams*red,1);
#define OP_ACT(ix) KERN.hiddens[idx].vec[ix+n_streams*red]=ann_act(KERN.hiddens[idx].vec[ix+n_streams*red])
            UNROLL_OMP_FOR(0,rem,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
        }
#else /*_MPI*/
        cblas_dgemv(CblasRowMajor,CblasNoTrans,N,M,
            1.0,KERN.hiddens[idx].weights,M,KERN.hiddens[idx-1].vec,1,0.,KERN.hiddens[idx].vec,1);
#define OP_ACT(ix) KERN.hiddens[idx].vec[ix]=ann_act(KERN.hiddens[idx].vec[ix])
        UNROLL_OMP_FOR(0,N,ANN_UNROLL,ACT,jdx);
#undef OP_ACT
#endif /*_MPI*/
#elif defined(SBLAS)
        /*move the parallel mv into a series of vv*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
        for(jdx=0;jdx<red;jdx++){
_HT;
            KERN.hiddens[idx].vec[jdx+stream*red]=cblas_ddot(
            M,&(KERN.hiddens[idx].weights[M*(jdx+stream*red)]),1,KERN.hiddens[idx-1].vec,1);
            KERN.hiddens[idx].vec[jdx+stream*red]=ann_act(KERN.hiddens[idx].vec[jdx+stream*red]);
        }
        MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
        if(rem>0){
#pragma omp parallel for private(jdx) _NT
            for(jdx=0;jdx<rem;jdx++){
_HT;
                KERN.hiddens[idx].vec[jdx+n_streams*red]=cblas_ddot(
                M,&(KERN.hiddens[idx].weights[M*(jdx+n_streams*red)]),1,KERN.hiddens[idx-1].vec,1);
                KERN.hiddens[idx].vec[jdx+n_streams*red]=ann_act(KERN.hiddens[idx].vec[jdx+n_streams*red]);
            }
        }
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
        for(jdx=0;jdx<N;jdx++){
_HT;
            KERN.hiddens[idx].vec[jdx]=cblas_ddot(
            M,&(KERN.hiddens[idx].weights[_2D_IDX(M,jdx,0)]),1,KERN.hiddens[idx-1].vec,1);
            KERN.hiddens[idx].vec[jdx]=ann_act(KERN.hiddens[idx].vec[jdx]);
        }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
        #pragma omp parallel for private(jdx,kdx) _NT
        for(jdx=0;jdx<red;jdx++){
            KERN.hiddens[idx].vec[jdx+stream*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[idx].vec[jdx+stream*red]+=KERN.hiddens[idx].weights[M*(jdx+stream*red)+ix]*KERN.hiddens[idx-1].vec[ix]
            UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
            KERN.hiddens[idx].vec[jdx+stream*red]=ann_act(KERN.hiddens[idx].vec[jdx+stream*red]);
        }
        MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
        if(rem>0){
#pragma omp parallel for private(jdx) _NT
            for(jdx=0;jdx<rem;jdx++){
                KERN.hiddens[idx].vec[jdx+n_streams*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[idx].vec[jdx+n_streams*red]+=KERN.hiddens[idx].weights[M*(jdx+n_streams*red)+ix]*KERN.hiddens[idx-1].vec[ix]
                UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
                KERN.hiddens[idx].vec[jdx+n_streams*red]=ann_act(KERN.hiddens[idx].vec[jdx+n_streams*red]);
            }
        }
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
        for(jdx=0;jdx<N;jdx++){
            KERN.hiddens[idx].vec[jdx]=0.;/*TRAP*/
#define OP_WI(ix) KERN.hiddens[idx].vec[jdx]+=KERN.hiddens[idx].weights[_2D_IDX(M,jdx,ix)]*KERN.hiddens[idx-1].vec[ix]
            UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
            KERN.hiddens[idx].vec[jdx]=ann_act(KERN.hiddens[idx].vec[jdx]);
        }
#endif /*_MPI*/
#endif /*PBLAS*/
    }
/*+++ III - output +++*/
    N=KERN.output.n_neurons;
    M=KERN.output.n_inputs;
    dv=TINY;
#ifdef _MPI
    red=N/n_streams;
    rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
    cblas_dgemv(CblasRowMajor,CblasNoTrans,red,M,
    1.0,KERN.output.weights+stream*M*red,M,KERN.hiddens[KERN.n_hiddens-1].vec,1,0.,KERN.output.vec+stream*red,1);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
        cblas_dgemv(CblasRowMajor,CblasNoTrans,rem,M,
        1.0,KERN.output.weights+n_streams*M*red,M,KERN.hiddens[KERN.n_hiddens-1].vec,1,0.,KERN.output.vec+n_streams*red,1);
    }
    /*SOFTMAX: calculate dv*/
    /* This should be equivalent to BLAS lvl. 1 dasum*/
#pragma omp parallel for private(jdx) reduction(+:dv) _NT
    for(jdx=0;jdx<red;jdx++){
        KERN.output.vec[jdx+stream*red]=exp(KERN.output.vec[jdx+stream*red]-1.0);
        dv+=KERN.output.vec[jdx+stream*red];
    }
    MPI_Allreduce(MPI_IN_PLACE,&dv,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(jdx) reduction(+:dv) _NT
        for(jdx=0;jdx<rem;jdx++){
            KERN.output.vec[jdx+n_streams*red]=exp(KERN.output.vec[jdx+n_streams*red]-1.0);
            dv+=KERN.output.vec[jdx+n_streams*red];
        }
    }
    /*SOFTMAX: calculate output*/
#define OP_SX(ix) KERN.output.vec[ix+stream*red]/=dv;
    UNROLL_OMP_FOR(0,red,ANN_UNROLL,SX,jdx);
#undef OP_SX
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#define OP_SX(ix) KERN.output.vec[ix+n_streams*red]/=dv;
        UNROLL_OMP_FOR(0,rem,ANN_UNROLL,SX,jdx);
#undef OP_SX
    }
#else /*_MPI*/
    /*serial dgemv (no thread support here)*/
    cblas_dgemv(CblasRowMajor,CblasNoTrans,N,M,
        1.0,KERN.output.weights,M,KERN.hiddens[KERN.n_hiddens-1].vec,1,0.,KERN.output.vec,1);
    /*SOFTMAX: calculate dv*/
#pragma omp parallel for private(jdx) reduction(+:dv) _NT
    for(jdx=0;jdx<N;jdx++){
        KERN.output.vec[jdx]=exp(KERN.output.vec[jdx]-1.0);
        dv+=KERN.output.vec[jdx];
    }
    /*SOFTMAX: calculate output*/
#define OP_SX(ix) KERN.output.vec[ix]/=dv;
    UNROLL_OMP_FOR(0,N,ANN_UNROLL,SX,jdx);
#undef OP_SX
#endif /*_MPI*/
#elif defined(SBLAS)
    /*move the mv into a series of vv*/
#ifdef _MPI
#pragma omp parallel for private(jdx) reduction(+:dv) _NT
    for(jdx=0;jdx<red;jdx++){
_HT;
        KERN.output.vec[jdx+stream*red]=cblas_ddot(
        M,&(KERN.output.weights[M*(jdx+stream*red)]),1,KERN.hiddens[KERN.n_hiddens-1].vec,1);
        /*SOFTMAX: calculate dv*/
        KERN.output.vec[jdx+stream*red]=exp(KERN.output.vec[jdx+stream*red]-1.0);
        dv+=KERN.output.vec[jdx+stream*red];
    }
    MPI_Allreduce(MPI_IN_PLACE,&dv,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(jdx) reduction(+:dv) _NT
        for(jdx=0;jdx<rem;jdx++){
_HT;
            KERN.output.vec[jdx+n_streams*red]=cblas_ddot(
            M,&(KERN.output.weights[M*(jdx+n_streams*red)]),1,KERN.hiddens[KERN.n_hiddens-1].vec,1);
            /*SOFTMAX: calculate dv*/
            KERN.output.vec[jdx+n_streams*red]=exp(KERN.output.vec[jdx+n_streams*red]-1.0);
            dv+=KERN.output.vec[jdx+n_streams*red];
        }
    }
    /*SOFTMAX: calculate output*/
#define OP_SX(ix) KERN.output.vec[ix+stream*red]/=dv;
    UNROLL_OMP_FOR(0,red,ANN_UNROLL,SX,jdx);
#undef OP_SX
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#define OP_SX(ix) KERN.output.vec[ix+n_streams*red]/=dv;
        UNROLL_OMP_FOR(0,rem,ANN_UNROLL,SX,jdx);
#undef OP_SX
#else /*_MPI*/
#pragma omp parallel for private(jdx) reduction(+:dv) _NT
    for(jdx=0;jdx<N;jdx++){
_HT;
        KERN.output.vec[jdx]=cblas_ddot(
        M,&(KERN.output.weights[_2D_IDX(M,jdx,0)]),1,KERN.hiddens[KERN.n_hiddens-1].vec,1);
        /*SOFTMAX: calculate dv*/
        KERN.output.vec[jdx]=exp(KERN.output.vec[jdx]-1.0);
        dv+=KERN.output.vec[jdx];
    }
    /*SOFTMAX: calculate output*/
#define OP_SX(ix) KERN.output.vec[ix]/=dv;
    UNROLL_OMP_FOR(0,N,ANN_UNROLL,SX,jdx);
#undef OP_SX
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) reduction(+:dv) _NT
    for(jdx=0;jdx<red;jdx++){
        KERN.output.vec[jdx+stream*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.output.vec[jdx+stream*red]+=KERN.output.weights[M*(jdx+stream*red)+ix]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
        UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
        /*SOFTMAX: calculate dv*/
        KERN.output.vec[jdx+stream*red]=exp(KERN.output.vec[jdx+stream*red]-1.0);
        dv+=KERN.output.vec[jdx+stream*red];
    }
    MPI_Allreduce(MPI_IN_PLACE,&dv,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(jdx,kdx) reduction(+:dv) _NT
        for(jdx=0;jdx<rem;jdx++){
            KERN.output.vec[jdx+n_streams*red]=0.;/*TRAP*/
#define OP_WI(ix) KERN.output.vec[jdx+n_streams*red]+=KERN.output.weights[M*(jdx+n_streams*red)+ix]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
            UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
            /*SOFTMAX: calculate dv*/
            KERN.output.vec[jdx+n_streams*red]=exp(KERN.output.vec[jdx+n_streams*red]-1.0);
            dv+=KERN.output.vec[jdx+n_streams*red];
        }
    }
    /*SOFTMAX: calculate output*/
#define OP_SX(ix) KERN.output.vec[ix+stream*red]/=dv;
    UNROLL_OMP_FOR(0,red,ANN_UNROLL,SX,jdx);
#undef OP_SX
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.vec,red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#define OP_SX(ix) KERN.output.vec[ix+n_streams*red]/=dv;
        UNROLL_OMP_FOR(0,rem,ANN_UNROLL,SX,jdx);
#undef OP_SX
    }
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
    for(jdx=0;jdx<N;jdx++){
        KERN.output.vec[jdx]=0.;/*TRAP*/
#define OP_WI(ix) KERN.output.vec[jdx]+=KERN.output.weights[_2D_IDX(M,jdx,ix)]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
        UNROLL_FOR(0,M,ANN_UNROLL,WI,kdx);
#undef OP_WI
        /*SOFTMAX: calculate dv*/
        KERN.output.vec[jdx]=exp(KERN.output.vec[jdx]-1.0);
        dv+=KERN.output.vec[jdx];
    }
    /*SOFTMAX: calculate output*/
#define OP_SX(ix) KERN.output.vec[ix]/=dv;
    UNROLL_OMP_FOR(0,N,ANN_UNROLL,SX,jdx);
#undef OP_SX
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
//  MPI_Barrier(MPI_COMM_WORLD);//WAIT FOR ALL TASKS BEFORE LEAVING
#endif
    /*done*/
#endif /*_CUDA*/
}
/*-------------------------------*/
/*+++ Train Error Calculation +++*/
/*-------------------------------*/
DOUBLE snn_kernel_train_error(kernel_ann *kernel, const DOUBLE *train){
    DOUBLE Ep=0.;
    UINT idx,N;
#ifdef _MPI
    UINT red,rem;
    UINT n_streams,stream;
    _NN(get,mpi_tasks)(&n_streams);
    _NN(get,curr_mpi_task)(&stream);
#endif /*_MPI*/
    N=KERN.n_outputs;
#ifdef _MPI
    red=N/n_streams;
    rem=N%n_streams;
#pragma omp parallel for private(idx) reduction(+:Ep) _NT
    for(idx=0;idx<red;idx++)
            Ep+=train[idx+stream*red]*log(KERN.output.vec[idx+stream*red]+TINY);
//            +(1.0-train[idx+stream*red])*log(1.0-KERN.output.vec[idx+stream*red]+TINY);
    MPI_Allreduce(MPI_IN_PLACE,&Ep,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if(rem>0) {
        for(idx=0;idx<rem;idx++)
            Ep+=train[idx+n_streams*red]*log(KERN.output.vec[idx+n_streams*red]+TINY);
//            +(1.0-train[idx+n_streams*red])*log(1.0-KERN.output.vec[idx+n_streams*red]+TINY);
    }
#else /*_MPI*/
#pragma omp parallel for private(idx) reduction(+:Ep) _NT
    for(idx=0;idx<N;idx++) if(KERN.output.vec[idx]>0.) Ep+=train[idx]*log(KERN.output.vec[idx]+TINY);
    //           +(1.0-train[idx])*log(1.0-KERN.output.vec[idx]+TINY);
#endif /*_MPI*/
    Ep*=-1.0/(N);
    return Ep;
}
/*------------------------*/
/*+++ Calculate deltas +++*/
/*------------------------*/
void snn_kernel_train_delta(kernel_ann *kernel,const DOUBLE *train, DOUBLE **delta_ptr){
#if !defined (PBLAS) && !defined (SBLAS)
        UINT kdx;
#endif
    UINT N,M;
        UINT idx, jdx;
#ifdef _MPI
    UINT red, rem;
    UINT n_streams,stream;
    MPI_Comm_size(MPI_COMM_WORLD,&n_streams);
    MPI_Comm_rank(MPI_COMM_WORLD,&stream);
#endif /*_MPI*/
/*^^^ output*/
    N=KERN.output.n_neurons;
#ifdef _MPI
    red=N/n_streams;
    rem=N%n_streams;
#define OP_DELTA(ix) delta_ptr[KERN.n_hiddens][ix+stream*red]=\
    (train[ix+stream*red]-KERN.output.vec[ix+stream*red])
    UNROLL_OMP_FOR(0,red,ANN_UNROLL,DELTA,idx);
#undef OP_DELTA
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[KERN.n_hiddens],red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#define OP_DELTA(ix) delta_ptr[KERN.n_hiddens][ix+n_streams*red]=\
    (train[ix+n_streams*red]-KERN.output.vec[ix+n_streams*red])
        UNROLL_OMP_FOR(0,rem,ANN_UNROLL,DELTA,idx);
#undef OP_DELTA
    }
#else /*_MPI*/
#define OP_DELTA(ix) delta_ptr[KERN.n_hiddens][ix]=(train[ix]-KERN.output.vec[ix])
    UNROLL_OMP_FOR(0,N,ANN_UNROLL,DELTA,idx);
#undef OP_DELTA
#endif /*_MPI*/
/*^^^ output to hidden*/
    N=KERN.output.n_neurons;
    M=KERN.output.n_inputs;
#ifdef _MPI
    red=M/n_streams;
    rem=M%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
    cblas_dgemv(CblasRowMajor,CblasTrans,N,red,
        1.0,KERN.output.weights+stream*red,M,delta_ptr[KERN.n_hiddens],1,0.,delta_ptr[KERN.n_hiddens-1]+stream*red,1);
#define OP_DACT(ix) delta_ptr[KERN.n_hiddens-1][ix+stream*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[ix+stream*red])
    UNROLL_OMP_FOR(0,red,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[KERN.n_hiddens-1],red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
        cblas_dgemv(CblasRowMajor,CblasTrans,N,rem,
            1.0,KERN.output.weights+n_streams*red,M,delta_ptr[KERN.n_hiddens],1,0.,delta_ptr[KERN.n_hiddens-1]+n_streams*red,1);
#define OP_DACT(ix) delta_ptr[KERN.n_hiddens-1][ix+n_streams*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[ix+n_streams*red])
        UNROLL_OMP_FOR(0,rem,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
    }
#else /*_MPI*/
    /*! transposed*/
    cblas_dgemv(CblasRowMajor,CblasTrans,N,M,1.0,KERN.output.weights,M,delta_ptr[KERN.n_hiddens],1,0.,delta_ptr[KERN.n_hiddens-1],1);
#define OP_DACT(ix) delta_ptr[KERN.n_hiddens-1][ix]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[ix])
    UNROLL_OMP_FOR(0,M,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
#endif /*_MPI*/
#elif defined(SBLAS)
    /*move the mv into a series of vv*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<red;jdx++){
_HT;
        delta_ptr[KERN.n_hiddens-1][jdx+stream*red]=cblas_ddot(
            N,&(KERN.output.weights[jdx+stream*red]),M,delta_ptr[KERN.n_hiddens],1);
        delta_ptr[KERN.n_hiddens-1][jdx+stream*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx+stream*red]);
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[KERN.n_hiddens-1],red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(jdx) _NT
        for(jdx=0;jdx<rem;jdx++){
_HT;
            delta_ptr[KERN.n_hiddens-1][jdx+n_streams*red]=cblas_ddot(
                N,&(KERN.output.weights[jdx+n_streams*red]),M,delta_ptr[KERN.n_hiddens],1);
            delta_ptr[KERN.n_hiddens-1][jdx+n_streams*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx+n_streams*red]);
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<M;jdx++){
_HT;
        /*since the matrix is transposed incX is the matrix stride!*/
        delta_ptr[KERN.n_hiddens-1][jdx]=cblas_ddot(
        N,&(KERN.output.weights[jdx]),M,&(delta_ptr[KERN.n_hiddens][0]),1);
        delta_ptr[KERN.n_hiddens-1][jdx]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx]);
    }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
    for(jdx=0;jdx<red;jdx++){
#define OP_WD(ix) delta_ptr[KERN.n_hiddens-1][jdx+stream*red]+=\
    KERN.output.weights[_2D_IDX(M,ix,jdx+stream*red)]*delta_ptr[KERN.n_hiddens][ix]
        UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
        delta_ptr[KERN.n_hiddens-1][jdx+stream*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx+stream*red]);
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[KERN.n_hiddens-1],red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
        for(jdx=0;jdx<rem;jdx++){
#define OP_WD(ix) delta_ptr[KERN.n_hiddens-1][jdx+n_streams*red]+=\
    KERN.output.weights[_2D_IDX(M,ix,jdx+n_streams*red)]*delta_ptr[KERN.n_hiddens][ix]
            UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
            delta_ptr[KERN.n_hiddens-1][jdx+n_streams*red]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx+n_streams*red]);
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
    for(jdx=0;jdx<M;jdx++){
#define OP_WD(ix) delta_ptr[KERN.n_hiddens-1][jdx]+=KERN.output.weights[_2D_IDX(M,ix,jdx)]*delta_ptr[KERN.n_hiddens][ix]
        UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
        delta_ptr[KERN.n_hiddens-1][jdx]*=ann_dact(KERN.hiddens[KERN.n_hiddens-1].vec[jdx]);
    }
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
#endif /*_MPI*/
/*^^^ hidden to hidden (if any)*/
    if(KERN.n_hiddens>1){
        for(idx=(KERN.n_hiddens-2);idx>0;idx--){
            N=KERN.hiddens[idx+1].n_neurons;
            M=KERN.hiddens[idx+1].n_inputs;
#ifdef _MPI
            red=M/n_streams;
            rem=M%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
            cblas_dgemv(CblasRowMajor,CblasTrans,N,red,
                 1.0,KERN.hiddens[idx+1].weights+stream*red,M,delta_ptr[idx+1],1,0.,delta_ptr[idx]+stream*red,1);
#define OP_DACT(ix) delta_ptr[idx][ix+stream*red]*=ann_dact(KERN.hiddens[idx].vec[ix+stream*red])
            UNROLL_OMP_FOR(0,red,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
            MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[idx],red,MPI_DOUBLE,MPI_COMM_WORLD);
            if(rem>0){
                cblas_dgemv(CblasRowMajor,CblasTrans,N,rem,
                    1.0,KERN.hiddens[idx+1].weights+n_streams*red,M,delta_ptr[idx+1],1,0.,delta_ptr[idx]+n_streams*red,1);
#define OP_DACT(ix) delta_ptr[idx][ix+n_streams*red]*=ann_dact(KERN.hiddens[idx].vec[ix+n_streams*red])
                UNROLL_OMP_FOR(0,rem,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
            }
#else /*_MPI*/
            /*! transposed*/
            cblas_dgemv(CblasRowMajor,CblasTrans,N,M,1.0,KERN.hiddens[idx+1].weights,M,delta_ptr[idx+1],1,0.,delta_ptr[idx],1);
#define OP_DACT(ix) delta_ptr[idx][ix]*=ann_dact(KERN.hiddens[idx].vec[ix])
            UNROLL_OMP_FOR(0,M,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
#endif /*_MPI*/
#elif defined(SBLAS)
            /*move the mv into a series of vv*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
            for(jdx=0;jdx<red;jdx++){
_HT;
                /*since the matrix is transposed incX is the matrix stride!*/
                delta_ptr[idx][jdx+stream*red]=cblas_ddot(
                N,&(KERN.hiddens[idx+1].weights[jdx+stream*red]),M,delta_ptr[idx+1],1);
                delta_ptr[idx][jdx+stream*red]*=ann_dact(KERN.hiddens[idx].vec[jdx+stream*red]);
            }
            MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[idx],red,MPI_DOUBLE,MPI_COMM_WORLD);
            if(rem>0){
#pragma omp parallel for private(jdx) _NT
                for(jdx=0;jdx<rem;jdx++){
_HT;
                    /*since the matrix is transposed incX is the matrix stride!*/
                    delta_ptr[idx][jdx+n_streams*red]=cblas_ddot(
                    N,&(KERN.hiddens[idx+1].weights[jdx+n_streams*red]),M,delta_ptr[idx+1],1);
                    delta_ptr[idx][jdx+n_streams*red]*=ann_dact(KERN.hiddens[idx].vec[jdx+n_streams*red]);
                }
            }
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
            for(jdx=0;jdx<M;jdx++){
_HT;
                /*since the matrix is transposed incX is the matrix stride!*/
                delta_ptr[idx][jdx]=cblas_ddot(
                N,&(KERN.hiddens[idx+1].weights[_2D_IDX(M,0,jdx)]),M,delta_ptr[idx+1],1);
                delta_ptr[idx][jdx]*=ann_dact(KERN.hiddens[idx].vec[jdx]);
            }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
            for(jdx=0;jdx<red;jdx++){
#define OP_WD(ix) delta_ptr[idx][jdx+stream*red]+=KERN.hiddens[idx+1].weights[_2D_IDX(M,ix,jdx+stream*red)]*delta_ptr[idx+1][ix]
                UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
                delta_ptr[idx][jdx+stream*red]*=ann_dact(KERN.hiddens[idx].vec[jdx+stream*red]);
            }
            MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[idx],red,MPI_DOUBLE,MPI_COMM_WORLD);
            if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
                for(jdx=0;jdx<rem;jdx++){
#define OP_WD(ix) delta_ptr[idx][jdx+n_streams*red]+=KERN.hiddens[idx+1].weights[_2D_IDX(M,ix,jdx+n_streams*red)]*delta_ptr[idx+1][ix]
                    UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
                    delta_ptr[idx][jdx+n_streams*red]*=ann_dact(KERN.hiddens[idx].vec[jdx+n_streams*red]);
                }
            }
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
            for(jdx=0;jdx<M;jdx++){
#define OP_WD(ix) delta_ptr[idx][jdx]+=KERN.hiddens[idx+1].weights[_2D_IDX(M,ix,jdx)]*delta_ptr[idx+1][ix]
                UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
                delta_ptr[idx][jdx]*=ann_dact(KERN.hiddens[idx].vec[jdx]);
            }
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
#endif /*_MPI*/
        }
        /*add zero*/
        N=KERN.hiddens[1].n_neurons;
        M=KERN.hiddens[1].n_inputs;
#ifdef _MPI
        red=M/n_streams;
        rem=M%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
        /*! transposed*/
#ifdef _MPI
        cblas_dgemv(CblasRowMajor,CblasTrans,N,red,
            1.0,KERN.hiddens[1].weights+stream*red,M,delta_ptr[1],1,0.,delta_ptr[0]+stream*red,1);
#define OP_DACT(ix) delta_ptr[0][ix+stream*red]*=ann_dact(KERN.hiddens[0].vec[ix+stream*red])
        UNROLL_OMP_FOR(0,red,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
        MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[0],red,MPI_DOUBLE,MPI_COMM_WORLD);
        if(rem>0){
            cblas_dgemv(CblasRowMajor,CblasTrans,N,rem,
            1.0,KERN.hiddens[1].weights+n_streams*red,M,delta_ptr[1],1,0.,delta_ptr[0]+n_streams*red,1);
#define OP_DACT(ix) delta_ptr[0][ix+n_streams*red]*=ann_dact(KERN.hiddens[0].vec[ix+n_streams*red])
            UNROLL_OMP_FOR(0,rem,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
        }
#else /*_MPI*/
        cblas_dgemv(CblasRowMajor,CblasTrans,N,M,
        1.0,KERN.hiddens[1].weights,M,delta_ptr[1],1,0.,delta_ptr[0],1);
#define OP_DACT(ix) delta_ptr[0][ix]*=ann_dact(KERN.hiddens[0].vec[ix])
        UNROLL_OMP_FOR(0,M,ANN_UNROLL,DACT,jdx);
#undef OP_DACT
#endif /*_MPI*/
#elif defined(SBLAS)
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
        for(jdx=0;jdx<red;jdx++){
_HT;
            /*since the matrix is transposed incX is the matrix stride!*/
            delta_ptr[0][jdx+stream*red]=cblas_ddot(
            N,&(KERN.hiddens[1].weights[_2D_IDX(M,0,jdx+stream*red)]),N,&(delta_ptr[1][0]),1);
            delta_ptr[0][jdx+stream*red]*=ann_dact(KERN.hiddens[0].vec[jdx+stream*red]);
        }
        MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[0],red,MPI_DOUBLE,MPI_COMM_WORLD);
        if(rem>0){
#pragma omp parallel for private(jdx) _NT
            for(jdx=0;jdx<rem;jdx++){
_HT;
                /*since the matrix is transposed incX is the matrix stride!*/
                delta_ptr[0][jdx+n_streams*red]=cblas_ddot(
                N,&(KERN.hiddens[1].weights[_2D_IDX(M,0,jdx+n_streams*red)]),N,&(delta_ptr[1][0]),1);
                delta_ptr[0][jdx+n_streams*red]*=ann_dact(KERN.hiddens[0].vec[jdx+n_streams*red]);
            }
        }
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
        for(jdx=0;jdx<M;jdx++){
_HT;
            /*since the matrix is transposed incX is the matrix stride!*/
            delta_ptr[0][jdx]=cblas_ddot(
            N,&(KERN.hiddens[1].weights[_2D_IDX(M,0,jdx)]),M,&(delta_ptr[1][0]),1);
            delta_ptr[0][jdx]*=ann_dact(KERN.hiddens[0].vec[jdx]);
        }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
        for(jdx=0;jdx<red;jdx++){
#define OP_WD(ix) delta_ptr[0][jdx+stream*red]+=KERN.hiddens[1].weights[_2D_IDX(M,ix,jdx+stream*red)]*delta_ptr[1][ix]
            UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
            delta_ptr[0][jdx+stream*red]*=ann_dact(KERN.hiddens[0].vec[jdx+stream*red]);
        }
        MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,delta_ptr[0],red,MPI_DOUBLE,MPI_COMM_WORLD);
        if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
            for(jdx=0;jdx<rem;jdx++){
#define OP_WD(ix) delta_ptr[0][jdx+n_streams*red]+=KERN.hiddens[1].weights[_2D_IDX(M,ix,jdx+n_streams*red)]*delta_ptr[1][ix]
                UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
                delta_ptr[0][jdx+n_streams*red]*=ann_dact(KERN.hiddens[0].vec[jdx+n_streams*red]);
            }
        }
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
        for(jdx=0;jdx<M;jdx++){
#define OP_WD(ix) delta_ptr[0][jdx]+=KERN.hiddens[1].weights[_2D_IDX(M,ix,jdx)]*delta_ptr[1][ix]
            UNROLL_FOR(0,N,ANN_UNROLL,WD,kdx);
#undef OP_WD
            delta_ptr[0][jdx]*=ann_dact(KERN.hiddens[0].vec[jdx]);
        }
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
#endif /*_MPI*/
    }
}
/*------------------------*/
/*+++ back-propagation +++*/
/*------------------------*/
DOUBLE snn_kernel_train(kernel_ann *kernel,const DOUBLE *train){
#define LEARN_RATE 0.01
#if !defined (PBLAS) && !defined (SBLAS)
    UINT kdx;
#endif
    UINT N,M;
    DOUBLE **delta_ptr;
    UINT idx;
#ifndef PBLAS
    UINT jdx;
#endif
    DOUBLE Ep =0.;
    DOUBLE Epr=0.;
#ifdef _MPI
    UINT red, rem;
    UINT n_streams,stream;
    MPI_Comm_size(MPI_COMM_WORLD,&n_streams);
    MPI_Comm_rank(MPI_COMM_WORLD,&stream);
#endif /*_MPI*/
    /*keep a track of mem*/
    UINT64 allocate=0.;
    ALLOC_REPORT(delta_ptr,KERN.n_hiddens+1,DOUBLE *,allocate);/*+1 for OUTPUT*/
    ALLOC_REPORT(delta_ptr[KERN.n_hiddens],KERN.n_outputs,DOUBLE,allocate);
    for(idx=0;idx<KERN.n_hiddens;idx++)
        ALLOC_REPORT(delta_ptr[idx],KERN.hiddens[idx].n_neurons,DOUBLE,allocate);
/*+++ I - forward is _supposed_ to be done already +++*/
    Ep=snn_kernel_train_error(kernel,train);
//  NN_DBG(stdout,"TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - calculate deltas +++*/
    snn_kernel_train_delta(kernel,train,delta_ptr);
/*+++ III - back propagation +++*/
/*^^^ output*/
    N=KERN.output.n_neurons;
    M=KERN.output.n_inputs;
#ifdef _MPI
    red=N/n_streams;
    rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
    cblas_dger(CblasRowMajor,red,M,LEARN_RATE,delta_ptr[KERN.n_hiddens]+stream*red,
    1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.output.weights+stream*M*red,M);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
        cblas_dger(CblasRowMajor,rem,M,LEARN_RATE,delta_ptr[KERN.n_hiddens]+n_streams*red,
        1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.output.weights+n_streams*M*red,M);
    }
#else /*_MPI*/
    cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[KERN.n_hiddens],1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.output.weights,M);
#endif /*_MPI*/
#elif defined(SBLAS)
    /*move the ger into a series of axpy*/
#ifdef _MPI
#pragma omp parallel for private(idx) _NT
    for(idx=0;idx<red;idx++){
_HT;
        cblas_daxpy(
        M,delta_ptr[KERN.n_hiddens][idx+stream*red]*LEARN_RATE,
        &(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,&(KERN.output.weights[_2D_IDX(M,idx+stream*red,0)]),1);
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(idx) _NT
        for(idx=0;idx<rem;idx++){
_HT;
            cblas_daxpy(
            M,delta_ptr[KERN.n_hiddens][idx+n_streams*red]*LEARN_RATE,
            &(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,
            &(KERN.output.weights[_2D_IDX(M,idx+n_streams*red,0)]),1);
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(idx) _NT
    for(idx=0;idx<N;idx++){
_HT;
        cblas_daxpy(
        M,delta_ptr[KERN.n_hiddens][idx]*LEARN_RATE,
        &(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,
        &(KERN.output.weights[_2D_IDX(M,idx,0)]),1);
    }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(idx,jdx) _NT
    for(idx=0;idx<red;idx++){
#define OP_DH(ix) KERN.output.weights[_2D_IDX(M,idx+stream*red,ix)]+=\
    LEARN_RATE*delta_ptr[KERN.n_hiddens][idx+stream*red]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
        UNROLL_FOR(0,M,ANN_UNROLL,DH,jdx);
#undef OP_DH
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(idx,jdx) _NT
        for(idx=0;idx<rem;idx++){
#define OP_DH(ix) KERN.output.weights[_2D_IDX(M,idx+n_streams*red,ix)]+=\
    LEARN_RATE*delta_ptr[KERN.n_hiddens][idx+n_streams*red]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
            UNROLL_FOR(0,M,ANN_UNROLL,DH,jdx);
#undef OP_DH
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(idx,jdx) _NT
    for(idx=0;idx<N;idx++){
#define OP_DH(ix) KERN.output.weights[_2D_IDX(M,idx,ix)]+=\
    LEARN_RATE*delta_ptr[KERN.n_hiddens][idx]*KERN.hiddens[KERN.n_hiddens-1].vec[ix]
        UNROLL_FOR(0,M,ANN_UNROLL,DH,jdx);
#undef OP_DH
    }
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
//  MPI_Barrier(MPI_COMM_WORLD);//WAIT FOR ALL TASKS
#endif /*_MPI*/
/*^^^ hiddens*/
    for(idx=(KERN.n_hiddens-1);idx>0;idx--){
        N=KERN.hiddens[idx].n_neurons;
        M=KERN.hiddens[idx].n_inputs;
#ifdef _MPI
        red=N/n_streams;
        rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
        cblas_dger(CblasRowMajor,red,M,LEARN_RATE,
        delta_ptr[idx]+stream*red,1,
        KERN.hiddens[idx-1].vec,1,
        KERN.hiddens[idx].weights+stream*M*red,M);
        MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
        if(rem>0){
            cblas_dger(CblasRowMajor,rem,M,LEARN_RATE,
                delta_ptr[idx]+n_streams*red,1,
                KERN.hiddens[idx-1].vec,1,
                KERN.hiddens[idx].weights+n_streams*M*red,M);
        }
#else /*_MPI*/
        cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[idx],1,KERN.hiddens[idx-1].vec,1,KERN.hiddens[idx].weights,M);
#endif /*_MPI*/
#elif defined(SBLAS)
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
        for(jdx=0;jdx<red;jdx++){
_HT;
            cblas_daxpy(M,delta_ptr[idx][jdx+stream*red]*LEARN_RATE,
                &(KERN.hiddens[idx-1].vec[0]),1,&(KERN.hiddens[idx].weights[_2D_IDX(M,jdx+stream*red,0)]),1);
        }
        MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
        if(rem>0){
#pragma omp parallel for private(jdx) _NT
            for(jdx=0;jdx<rem;jdx++){
_HT;
                cblas_daxpy(M,delta_ptr[idx][jdx+n_streams*red]*LEARN_RATE,
                &(KERN.hiddens[idx-1].vec[0]),1,&(KERN.hiddens[idx].weights[_2D_IDX(M,jdx+n_streams*red,0)]),1);
            }
        }
#else /*_MPI*/
        /*move the ger into a series of axpy*/
#pragma omp parallel for private(jdx) _NT
        for(jdx=0;jdx<N;jdx++){
_HT;
            cblas_daxpy(M,delta_ptr[idx][jdx]*LEARN_RATE,
                &(KERN.hiddens[idx-1].vec[0]),1,&(KERN.hiddens[idx].weights[_2D_IDX(M,jdx,0)]),1);
        }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
        for(jdx=0;jdx<red;jdx++){
#define OP_DH(ix) KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx+stream*red,ix)]+=\
    LEARN_RATE*delta_ptr[idx][jdx+stream*red]*KERN.hiddens[idx-1].vec[ix]
            UNROLL_FOR(0,M,ANN_UNROLL,DH,kdx);
#undef OP_DH
        }
        MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
        if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
            for(jdx=0;jdx<rem;jdx++){
#define OP_DH(ix) KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx+n_streams*red,ix)]+=\
    LEARN_RATE*delta_ptr[idx][jdx+n_streams*red]*KERN.hiddens[idx-1].vec[ix]
                UNROLL_FOR(0,M,ANN_UNROLL,DH,kdx);
#undef OP_DH
            }
        }
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
        for(jdx=0;jdx<N;jdx++){
#define OP_DH(ix) KERN.hiddens[idx].weights[_2D_IDX(KERN.hiddens[idx].n_inputs,jdx,ix)]+=\
    LEARN_RATE*delta_ptr[idx][jdx]*KERN.hiddens[idx-1].vec[ix]
            UNROLL_FOR(0,M,ANN_UNROLL,DH,kdx);
#undef OP_DH
        }
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
//  MPI_Barrier(MPI_COMM_WORLD);//WAIT FOR ALL TASKS
#endif /*_MPI*/
    }
    /*add zero*/
    N=KERN.hiddens[0].n_neurons;
    M=KERN.hiddens[0].n_inputs;
#ifdef _MPI
    red=N/n_streams;
    rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
    cblas_dger(CblasRowMajor,red,M,LEARN_RATE,delta_ptr[0]+stream*red,1,KERN.in,1,KERN.hiddens[0].weights+stream*M*red,M);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
        cblas_dger(CblasRowMajor,rem,M,
            LEARN_RATE,delta_ptr[0]+n_streams*red,1,
            KERN.in,1,
            KERN.hiddens[0].weights+n_streams*M*red,M);
    }
#else /*_MPI*/
    cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[0],1,KERN.in,1,KERN.hiddens[0].weights,M);
#endif /*_MPI*/
#elif defined(SBLAS)
    /*move the ger into a series of axpy*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<red;jdx++){
        cblas_daxpy(M,LEARN_RATE*delta_ptr[0][jdx+stream*red],KERN.in,1,&(KERN.hiddens[0].weights[_2D_IDX(M,jdx+stream*red,0)]),1);
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
        for(jdx=0;jdx<rem;jdx++){
            cblas_daxpy(M,LEARN_RATE*delta_ptr[0][jdx+n_streams*red],
            KERN.in,1,&(KERN.hiddens[0].weights[_2D_IDX(M,jdx+n_streams*red,0)]),1);
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<N;jdx++){
        cblas_daxpy(M,LEARN_RATE*delta_ptr[0][jdx],KERN.in,1,&(KERN.hiddens[0].weights[_2D_IDX(M,jdx,0)]),1);
    }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
    for(jdx=0;jdx<red;jdx++){
#define OP_DI(ix) KERN.hiddens[0].weights[_2D_IDX(M,jdx+stream*red,ix)]+=LEARN_RATE*delta_ptr[0][jdx+stream*red]*KERN.in[ix]
        UNROLL_FOR(0,M,ANN_UNROLL,DI,kdx);
#undef OP_DI
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
        for(jdx=0;jdx<rem;jdx++){
#define OP_DI(ix) KERN.hiddens[0].weights[_2D_IDX(M,jdx+n_streams*red,ix)]+=LEARN_RATE*delta_ptr[0][jdx+n_streams*red]*KERN.in[ix]
            UNROLL_FOR(0,M,ANN_UNROLL,DI,kdx);
#undef OP_DI
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
    for(jdx=0;jdx<N;jdx++){
#define OP_DI(ix) KERN.hiddens[0].weights[_2D_IDX(M,jdx,ix)]+=LEARN_RATE*delta_ptr[0][jdx]*KERN.in[ix]
        UNROLL_FOR(0,M,ANN_UNROLL,DI,kdx);
#undef OP_DI
    }
#endif /*_MPI*/
#endif /*PBLAS*/
#ifdef _MPI
//  MPI_Barrier(MPI_COMM_WORLD);//WAIT FOR ALL TASKS
#endif /*_MPI*/
/*+++ IV - update error +++*/
    snn_kernel_run(kernel);
    Epr=snn_kernel_train_error(kernel,train);
//  NN_DBG(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ V - cleanup +++*/
    for(idx=0;idx<(KERN.n_hiddens+1);idx++){
        FREE(delta_ptr[idx]);
        delta_ptr[idx]=NULL;
    }
    FREE(delta_ptr);
    return Ep-Epr;
}
/*---------------------------------*/
/*+++ momentum back-propagation +++*/
/*---------------------------------*/
DOUBLE snn_kernel_train_momentum(kernel_ann *kernel,const DOUBLE *train,DOUBLE alpha){
    UINT idx,N,M;
#ifdef _MPI
    UINT red, rem;
    UINT n_streams,stream;
    MPI_Comm_size(MPI_COMM_WORLD,&n_streams);
    MPI_Comm_rank(MPI_COMM_WORLD,&stream);
#endif /*_MPI*/
#if !defined (PBLAS) && !defined (SBLAS)
    UINT kdx;
#endif
#ifndef PBLAS
    UINT jdx;
#endif
    DOUBLE Ep=0.;
    DOUBLE Epr=0.;
    DOUBLE **delta_ptr;
    /*keep a track of mem*/
    if(!ann_validate_kernel(kernel)) return 0.;
    UINT64 allocate=0.;
    ALLOC_REPORT(delta_ptr,KERN.n_hiddens+1,DOUBLE *,allocate);/*+1 for OUTPUT*/
    ALLOC_REPORT(delta_ptr[KERN.n_hiddens],KERN.n_outputs,DOUBLE,allocate);
    for(idx=0;idx<KERN.n_hiddens;idx++)
        ALLOC_REPORT(delta_ptr[idx],KERN.hiddens[idx].n_neurons,DOUBLE,allocate);
/*+++ I - forward is _supposed_ to be done already +++*/
    Ep=snn_kernel_train_error(kernel,train);
//  NN_DBG(stdout,"TRAINING INITIAL ERROR: %.15f\n",Ep);
/*+++ II - calculate deltas +++*/
    snn_kernel_train_delta(kernel,train,delta_ptr);
/*+++ III - back propagation +++*/
/*^^^ output*/
    N=KERN.output.n_neurons;
    M=KERN.output.n_inputs;
#ifdef _MPI
    red=N/n_streams;
    rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
    cblas_dger(CblasRowMajor,red,M,LEARN_RATE,delta_ptr[KERN.n_hiddens]+stream*red,
    1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.dw[KERN.n_hiddens]+stream*M*red,M);
    cblas_daxpy(red*M,1.0,KERN.dw[KERN.n_hiddens]+stream*M*red,1,KERN.output.weights+stream*M*red,1);
    cblas_dscal(red*M,alpha,KERN.dw[KERN.n_hiddens]+stream*M*red,1);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[KERN.n_hiddens],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
        cblas_dger(CblasRowMajor,rem,M,LEARN_RATE,delta_ptr[KERN.n_hiddens]+n_streams*red,
        1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.dw[KERN.n_hiddens]+n_streams*M*red,M);
        cblas_daxpy(rem*M,1.0,KERN.dw[KERN.n_hiddens]+n_streams*M*red,1,KERN.output.weights+n_streams*M*red,1);
        cblas_dscal(rem*M,alpha,KERN.dw[KERN.n_hiddens]+n_streams*M*red,1);
    }
#else /*_MPI*/
    /*unfortunately dger output can't be scaled*/
    cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[KERN.n_hiddens],
    1,KERN.hiddens[KERN.n_hiddens-1].vec,1,KERN.dw[KERN.n_hiddens],M);
    cblas_daxpy(N*M,1.0,KERN.dw[KERN.n_hiddens],1,KERN.output.weights,1);
    cblas_dscal(N*M,alpha,KERN.dw[KERN.n_hiddens],1);
#endif /*_MPI*/
#elif defined(SBLAS)
    /*move the ger into a series of axpy*/
#ifdef _MPI
#pragma omp parallel for private(idx) _NT
    for(idx=0;idx<red;idx++){
_HT;
        cblas_daxpy(M,delta_ptr[KERN.n_hiddens][idx+stream*red]*LEARN_RATE,
        &(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,
        &(KERN.dw[KERN.n_hiddens][(idx+stream*red)*M]),1);
        cblas_daxpy(M,1.0,
        &(KERN.dw[KERN.n_hiddens][(idx+stream*red)*M]),1,
        &(KERN.output.weights[_2D_IDX(M,idx+stream*red,0)]),1);
        cblas_dscal(M,alpha,&(KERN.dw[KERN.n_hiddens][(idx+stream*red)*M]),1);
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[KERN.n_hiddens],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(idx) _NT
        for(idx=0;idx<rem;idx++){
_HT;
            cblas_daxpy(M,delta_ptr[KERN.n_hiddens][idx+n_streams*red]*LEARN_RATE,
            &(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,
            &(KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M]),1);
            cblas_daxpy(M,1.0,
            &(KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M]),1,
            &(KERN.output.weights[_2D_IDX(M,idx+n_streams*red,0)]),1);
            cblas_dscal(M,alpha,&(KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M]),1);
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(idx) _NT
    for(idx=0;idx<N;idx++){
_HT;
        //dw += LEARN_RATE*delta*y
        cblas_daxpy(M,delta_ptr[KERN.n_hiddens][idx]*LEARN_RATE,
        &(KERN.hiddens[KERN.n_hiddens-1].vec[0]),1,
        &(KERN.dw[KERN.n_hiddens][idx*M]),1);
        //W += dw
        cblas_daxpy(M,1.0,
        &(KERN.dw[KERN.n_hiddens][idx*M]),1,
        &(KERN.output.weights[_2D_IDX(M,idx,0)]),1);
        //dw *= alpha
        cblas_dscal(M,alpha,&(KERN.dw[KERN.n_hiddens][idx*M]),1);
    }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(idx,jdx) _NT
    for(idx=0;idx<red;idx++){
        for(jdx=0;jdx<M;jdx++){
            KERN.dw[KERN.n_hiddens][(idx+stream*red)*M+jdx]+=
                LEARN_RATE*delta_ptr[KERN.n_hiddens][idx+stream*red]*KERN.hiddens[KERN.n_hiddens-1].vec[jdx];
            KERN.output.weights[(idx+stream*red)*M+jdx]+=KERN.dw[KERN.n_hiddens][(idx+stream*red)*M+jdx];
            KERN.dw[KERN.n_hiddens][(idx+stream*red)*M+jdx]*=alpha;
        }
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.output.weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[KERN.n_hiddens],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(idx,jdx) _NT
        for(idx=0;idx<rem;idx++){
            for(jdx=0;jdx<M;jdx++){
                KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M+jdx]+=
                    LEARN_RATE*delta_ptr[KERN.n_hiddens][idx+n_streams*red]*KERN.hiddens[KERN.n_hiddens-1].vec[jdx];
                KERN.output.weights[(idx+n_streams*red)*M+jdx]+=KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M+jdx];
                KERN.dw[KERN.n_hiddens][(idx+n_streams*red)*M+jdx]*=alpha;
            }
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(idx,jdx) _NT
    for(idx=0;idx<N;idx++){
        for(jdx=0;jdx<M;jdx++){
            KERN.dw[KERN.n_hiddens][idx*M+jdx]+=LEARN_RATE*delta_ptr[KERN.n_hiddens][idx]*KERN.hiddens[KERN.n_hiddens-1].vec[jdx];
            KERN.output.weights[_2D_IDX(M,idx,jdx)]+=KERN.dw[KERN.n_hiddens][idx*M+jdx];
            KERN.dw[KERN.n_hiddens][idx*M+jdx]*=alpha;
        }
    }
#endif /*_MPI*/
#endif /*PBLAS*/
/*^^^ hiddens*/
    for(idx=(KERN.n_hiddens-1);idx>0;idx--){
        N=KERN.hiddens[idx].n_neurons;
        M=KERN.hiddens[idx].n_inputs;
#ifdef _MPI
        red=N/n_streams;
        rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
    cblas_dger(CblasRowMajor,red,M,LEARN_RATE,
        delta_ptr[idx]+stream*red,1,KERN.hiddens[idx-1].vec,1,KERN.dw[idx]+stream*M*red,M);
    cblas_daxpy(N*M,1.0,KERN.dw[idx]+stream*M*red,1,KERN.hiddens[idx].weights+stream*M*red,1);
    cblas_dscal(N*M,alpha,KERN.dw[idx]+stream*M*red,1);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[idx],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
        cblas_dger(CblasRowMajor,red,M,LEARN_RATE,
            delta_ptr[idx]+n_streams*red,1,KERN.hiddens[idx-1].vec,1,KERN.dw[idx]+n_streams*M*red,M);
        cblas_daxpy(N*M,1.0,KERN.dw[idx]+n_streams*M*red,1,KERN.hiddens[idx].weights+n_streams*M*red,1);
        cblas_dscal(N*M,alpha,KERN.dw[idx]+n_streams*M*red,1);
    }
#else /*_MPI*/
    cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[idx],1,KERN.hiddens[idx-1].vec,1,KERN.dw[idx],M);
    cblas_daxpy(N*M,1.0,KERN.dw[idx],1,KERN.hiddens[idx].weights,1);
    cblas_dscal(N*M,alpha,KERN.dw[idx],1);
#endif /*_MPI*/
#elif defined(SBLAS)
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<red;jdx++){
_HT;
        cblas_daxpy(M,delta_ptr[idx][jdx+stream*red]*LEARN_RATE,
            KERN.hiddens[idx-1].vec,1,&(KERN.dw[idx][(jdx+stream*red)*M]),1);
        cblas_daxpy(M,1.0,&(KERN.dw[idx][(jdx+stream*red)*M]),1,&(KERN.hiddens[idx].weights[(jdx+stream*red)*M]),1);
        cblas_dscal(M,alpha,&(KERN.dw[idx][(jdx+stream*red)*M]),1);
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[idx],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(jdx) _NT
        for(jdx=0;jdx<rem;jdx++){
_HT;
            cblas_daxpy(M,delta_ptr[idx][jdx+n_streams*red]*LEARN_RATE,
                KERN.hiddens[idx-1].vec,1,&(KERN.dw[idx][(jdx+n_streams*red)*M]),1);
            cblas_daxpy(M,1.0,&(KERN.dw[idx][(jdx+n_streams*red)*M]),1,&(KERN.hiddens[idx].weights[(jdx+n_streams*red)*M]),1);
            cblas_dscal(M,alpha,&(KERN.dw[idx][(jdx+n_streams*red)*M]),1);
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<N;jdx++){
_HT;
        //dw += LEARN_RATE*delta*y
        cblas_daxpy(M,delta_ptr[idx][jdx]*LEARN_RATE,KERN.hiddens[idx-1].vec,1,&(KERN.dw[idx][_2D_IDX(M,jdx,0)]),1);
        //W += dw
        cblas_daxpy(M,1.0,&(KERN.dw[idx][_2D_IDX(M,jdx,0)]),1,&(KERN.hiddens[idx].weights[_2D_IDX(M,jdx,0)]),1);
        //dw *= alpha
        cblas_dscal(M,alpha,&(KERN.dw[idx][_2D_IDX(M,jdx,0)]),1);
    }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx,kdx) _NT
    for(jdx=0;jdx<red;jdx++){
        for(kdx=0;kdx<M;kdx++){
            KERN.dw[idx][(jdx+stream*red)*M+kdx]+=
                LEARN_RATE*delta_ptr[idx][jdx+stream*red]*KERN.hiddens[idx-1].vec[kdx];
            KERN.hiddens[idx].weights[(jdx+stream*red)*M+kdx]+=KERN.dw[idx][(jdx+stream*red)*M+kdx];
            KERN.dw[idx][(jdx+stream*red)*M+kdx]*=alpha;
        }
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[idx].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[idx],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(jdx,kdx) _NT
        for(jdx=0;jdx<rem;jdx++){
            for(kdx=0;kdx<M;kdx++){
                KERN.dw[idx][(jdx+n_streams*red)*M+kdx]+=
                    LEARN_RATE*delta_ptr[idx][jdx+n_streams*red]*KERN.hiddens[idx-1].vec[kdx];
                KERN.hiddens[idx].weights[(jdx+n_streams*red)*M+kdx]+=KERN.dw[idx][(jdx+n_streams*red)*M+kdx];
                KERN.dw[idx][(jdx+n_streams*red)*M+kdx]*=alpha;
            }
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(jdx,kdx) _NT
    for(jdx=0;jdx<N;jdx++){
        for(kdx=0;kdx<M;kdx++){
            KERN.dw[idx][_2D_IDX(M,jdx,kdx)]+=LEARN_RATE*delta_ptr[idx][jdx]*KERN.hiddens[idx-1].vec[kdx];
            KERN.hiddens[idx].weights[_2D_IDX(M,jdx,kdx)]+=KERN.dw[idx][_2D_IDX(M,jdx,kdx)];
            KERN.dw[idx][_2D_IDX(M,jdx,kdx)]*=alpha;
        }
    }
#endif /*_MPI*/
#endif /*PBLAS*/
    }/*idx: hiddens*/
    /*add zero*/
    N=KERN.hiddens[0].n_neurons;
    M=KERN.hiddens[0].n_inputs;
#ifdef _MPI
    red=N/n_streams;
    rem=N%n_streams;
#endif /*_MPI*/
#ifdef PBLAS
#ifdef _MPI
    cblas_dger(CblasRowMajor,red,M,LEARN_RATE,delta_ptr[0]+stream*red,1,KERN.in,1,KERN.dw[0]+stream*M*red,M);
    cblas_daxpy(red*M,1.0,KERN.dw[0]+stream*M*red,1,KERN.hiddens[0].weights+stream*M*red,1);
    cblas_dscal(red*M,alpha,KERN.dw[0]+stream*M*red,1);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[0],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
        cblas_dger(CblasRowMajor,rem,M,LEARN_RATE,delta_ptr[0]+n_streams*red,1,KERN.in,1,KERN.dw[0]+n_streams*M*red,M);
        cblas_daxpy(rem*M,1.0,KERN.dw[0]+n_streams*M*red,1,KERN.hiddens[0].weights+n_streams*M*red,1);
        cblas_dscal(rem*M,alpha,KERN.dw[0]+n_streams*M*red,1);
    }
#else /*_MPI*/
    cblas_dger(CblasRowMajor,N,M,LEARN_RATE,delta_ptr[0],1,KERN.in,1,KERN.dw[0],M);
    cblas_daxpy(N*M,1.0,KERN.dw[0],1,KERN.hiddens[0].weights,1);
    cblas_dscal(N*M,alpha,KERN.dw[0],1);
#endif /*_MPI*/
#elif defined(SBLAS)
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<red;jdx++){
_HT;
        cblas_daxpy(M,delta_ptr[0][jdx+stream*red]*LEARN_RATE,KERN.in,1,&(KERN.dw[0][(jdx+stream*red)*M]),1);
        cblas_daxpy(M,1.0,&(KERN.dw[0][(jdx+stream*red)*M]),1,&(KERN.hiddens[0].weights[(jdx+stream*red)*M]),1);
        cblas_dscal(M,alpha,&(KERN.dw[0][(jdx+stream*red)*M]),1);
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[0],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
        for(jdx=0;jdx<rem;jdx++){
_HT;
            cblas_daxpy(M,delta_ptr[0][jdx+n_streams*red]*LEARN_RATE,KERN.in,1,&(KERN.dw[0][(jdx+n_streams*red)*M]),1);
            cblas_daxpy(M,1.0,&(KERN.dw[0][(jdx+n_streams*red)*M]),1,&(KERN.hiddens[0].weights[(jdx+n_streams*red)*M]),1);
            cblas_dscal(M,alpha,&(KERN.dw[0][(jdx+n_streams*red)*M]),1);
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<N;jdx++){
_HT;
        //dw += LEARN_RATE*delta*y
        cblas_daxpy(M,delta_ptr[0][jdx]*LEARN_RATE,KERN.in,1,&(KERN.dw[0][_2D_IDX(M,jdx,0)]),1);
        //W += dw
        cblas_daxpy(M,1.0,&(KERN.dw[0][_2D_IDX(M,jdx,0)]),1,&(KERN.hiddens[0].weights[_2D_IDX(M,jdx,0)]),1);
        //dw *= alpha
        cblas_dscal(M,alpha,&(KERN.dw[0][_2D_IDX(M,jdx,0)]),1);
    }
#endif /*_MPI*/
#else /*no PBLAS no SBLAS*/
#ifdef _MPI
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<red;jdx++){
        for(kdx=0;kdx<M;kdx++){
            KERN.dw[0][(jdx+stream*red)*M+kdx]+=LEARN_RATE*delta_ptr[0][jdx+stream*red]*KERN.in[kdx];
            KERN.hiddens[0].weights[(jdx+stream*red)*M+kdx]+=KERN.dw[0][(jdx+stream*red)*M+kdx];
            KERN.dw[0][(jdx+stream*red)*M+kdx]*=alpha;
        }
    }
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.hiddens[0].weights,M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,KERN.dw[0],M*red,MPI_DOUBLE,MPI_COMM_WORLD);
    if(rem>0){
#pragma omp parallel for private(jdx) _NT
        for(jdx=0;jdx<rem;jdx++){
            for(kdx=0;kdx<M;kdx++){
                KERN.dw[0][(jdx+n_streams*red)*M+kdx]+=LEARN_RATE*delta_ptr[0][jdx+n_streams*red]*KERN.in[kdx];
                KERN.hiddens[0].weights[(jdx+n_streams*red)*M+kdx]+=KERN.dw[0][(jdx+n_streams*red)*M+kdx];
                KERN.dw[0][(jdx+n_streams*red)*M+kdx]*=alpha;
            }
        }
    }
#else /*_MPI*/
#pragma omp parallel for private(jdx) _NT
    for(jdx=0;jdx<N;jdx++){
        for(kdx=0;kdx<M;kdx++){
            KERN.dw[0][_2D_IDX(M,jdx,kdx)]+=LEARN_RATE*delta_ptr[0][jdx]*KERN.in[kdx];
            KERN.hiddens[0].weights[_2D_IDX(M,jdx,kdx)]+=KERN.dw[0][_2D_IDX(M,jdx,kdx)];
            KERN.dw[0][_2D_IDX(M,jdx,kdx)]*=alpha;
        }
    }
#endif /*_MPI*/
#endif /*PBLAS*/
/*+++ IV - update error +++*/
    snn_kernel_run(kernel);
    Epr=snn_kernel_train_error(kernel,train);
//  NN_DBG(stdout,"TRAINING UPDATED ERROR: %.15f\n",Epr);
/*+++ IV - cleanup +++*/
    for(idx=0;idx<(KERN.n_hiddens+1);idx++){
        FREE(delta_ptr[idx]);
        delta_ptr[idx]=NULL;
    }
    FREE(delta_ptr);
    return Ep-Epr;
}

/*--------------------------*/
/* train SNN sample with BP */
/*--------------------------*/
DOUBLE snn_train_BP(kernel_ann *kernel,DOUBLE *train_in,DOUBLE *train_out,DOUBLE delta){
/*typical values delta=0.000001*/
    BOOL is_ok;
    UINT   idx;
    UINT  iter;
    UINT max_p;
    UINT p_trg;
    DOUBLE dEp;
    DOUBLE *ptr;
    DOUBLE probe;
#ifdef _CUDA
    cudastreams *cudas=_NN(return,cudas)();
    DOUBLE *train_gpu;
    CUDA_SET_DEV(*cudas,0);/*make sure all transfer happen to gpu[0]*/
    CUDA_C2G_CP(train_in,KERN.in,KERN.n_inputs,DOUBLE);
    if((cudas->mem_model==CUDA_MEM_EXP)&&(cudas->n_gpu>1)){
        kernel_ann *kx; 
        /*distribute input to other GPUs*/
        for(int gpu=1;gpu<cudas->n_gpu;gpu++){
            /*copy*/
            kx=KERN.kerns[gpu];
            CUDA_G2G_CP(KERN.in,kx->in,KERN.n_inputs,DOUBLE);
        }
    }
if(cudas->mem_model==CUDA_MEM_CMM){
    CUDA_ALLOC_MM(train_gpu,KERN.n_outputs,DOUBLE);
}else{
    CUDA_ALLOC(train_gpu,KERN.n_outputs,DOUBLE);
}
    CUDA_C2G_CP(train_out,train_gpu,KERN.n_outputs,DOUBLE);
    scuda_snn_forward(kernel,cudas);
    dEp=scuda_snn_error(kernel,train_gpu,cudas);
#else  /*_CUDA*/
    /*copy input*/
    ARRAY_CP(train_in,KERN.in,KERN.n_inputs);
    snn_kernel_run(kernel);/*also FILL vec*/
    dEp=snn_kernel_train_error(kernel,train_out);
#endif /*_CUDA*/
    NN_COUT(stdout," init=%15.10f",dEp);
    iter=0;
    do{
        iter++;
#ifdef _CUDA
        dEp=(DOUBLE)scuda_snn_train(kernel,train_gpu,_NN(return,cudas)());
if(cudas->mem_model!=CUDA_MEM_CMM){
        /*we have to sync output.cuda_v -> out*/
        CUDA_SET_DEV(*cudas,0);/*make sure transfer happen from GPU[0]*/
        CUDA_G2C_CP(kernel->tmp_cpu,kernel->output.vec,KERN.n_outputs,DOUBLE);
        CUDA_SYNC();/*<-useful?*/
        ptr=kernel->tmp_cpu;
}else{
        /*CMM can use GPU memory directly*/
        cudaMemPrefetchAsync(kernel->output.vec,
            kernel->n_outputs*sizeof(DOUBLE),cudaCpuDeviceId,NULL);
        ptr=kernel->output.vec;
}
#else /*_CUDA*/
        dEp=snn_kernel_train(kernel,train_out);
        ptr=kernel->output.vec;
#endif /*_CUDA*/
        /*1- determine max_p, p_trg*/
        probe=-1.0;max_p=0;p_trg=0;
        for(idx=0;idx<KERN.n_outputs;idx++){
            if(probe<ptr[idx]){
                probe=ptr[idx];
                max_p=idx;
            }
            if(train_out[idx]==1.0) p_trg=idx;
        }
        /*2- match*/
        is_ok=(max_p==p_trg);
        if(iter==1){
            /*determine if we get a good answer at first try*/
            if(is_ok==TRUE) NN_COUT(stdout," OK");
            else NN_COUT(stdout," NO");
        }
        if(iter>10239) break;/*failsafe number of wrong iteration*/
    }while((dEp > delta)||(!(is_ok==TRUE)));
    NN_COUT(stdout," N_ITER=%8i",iter);
    if(is_ok==TRUE) NN_COUT(stdout," SUCCESS!\n");
    else NN_COUT(stdout," FAIL!\n");
    fflush(stdout);
#ifdef _CUDA
    CUDA_FREE(train_gpu);
#endif /*_CUDA*/
    return dEp;
}
/*---------------------------*/
/* train SNN sample with BPM */
/*---------------------------*/
DOUBLE snn_train_BPM(kernel_ann *kernel,DOUBLE *train_in,DOUBLE *train_out,DOUBLE alpha,DOUBLE delta){
/*typical values alpha=0.2 delta=0.00001*/
    BOOL is_ok;
    UINT   idx;
    UINT  iter;
    UINT max_p;
    UINT p_trg;
    DOUBLE dEp;
    DOUBLE *ptr;
    DOUBLE probe;
    ann_raz_momentum(kernel);
#ifdef _CUDA
    cudastreams *cudas=_NN(return,cudas)();
    DOUBLE *train_gpu;
    CUDA_SET_DEV(*cudas,0);/*make sure all transfer happen to gpu[0]*/
    CUDA_C2G_CP(train_in,KERN.in,KERN.n_inputs,DOUBLE);
    if((cudas->mem_model==CUDA_MEM_EXP)&&(cudas->n_gpu>1)){
        kernel_ann *kx; 
        /*distribute input to other GPUs*/
        for(int gpu=1;gpu<cudas->n_gpu;gpu++){
            /*copy*/
            kx=KERN.kerns[gpu];
            CUDA_G2G_CP(KERN.in,kx->in,KERN.n_inputs,DOUBLE);
        }
    }
if(cudas->mem_model==CUDA_MEM_CMM){
    CUDA_ALLOC_MM(train_gpu,KERN.n_outputs,DOUBLE);
}else{
    CUDA_ALLOC(train_gpu,KERN.n_outputs,DOUBLE);
}
    CUDA_C2G_CP(train_out,train_gpu,KERN.n_outputs,DOUBLE);
    scuda_snn_forward(kernel,cudas);
    dEp=scuda_snn_error(kernel,train_gpu,cudas);
#else  /*_CUDA*/
    /*copy input*/
    ARRAY_CP(train_in,KERN.in,KERN.n_inputs);
    snn_kernel_run(kernel);/*also FILL vec*/
    dEp=snn_kernel_train_error(kernel,train_out);
#endif /*_CUDA*/
    NN_COUT(stdout," init=%15.10f",dEp);
    iter=0;
    do{
        iter++;
#ifdef _CUDA
        dEp=(DOUBLE)scuda_snn_train_momentum(kernel,train_gpu,alpha,_NN(return,cudas)());
if(cudas->mem_model!=CUDA_MEM_CMM){
        /*we have to sync output.cuda_v -> out*/
        CUDA_SET_DEV(*cudas,0);/*make sure transfer happen from GPU[0]*/
        CUDA_G2C_CP(kernel->tmp_cpu,kernel->output.vec,KERN.n_outputs,DOUBLE);
        CUDA_SYNC();/*<-useful?*/
        ptr=kernel->tmp_cpu;
}else{
        /*CMM can use GPU memory directly*/
        /*TODO: require prefetch?*/
        ptr=kernel->output.vec;
}
#else /*_CUDA*/
        dEp=snn_kernel_train_momentum(kernel,train_out,alpha);
        ptr=kernel->output.vec;
#endif /*_CUDA*/
        /*1- determine max_p, p_trg*/
        probe=-1.0;max_p=0;p_trg=0;
        for(idx=0;idx<KERN.n_outputs;idx++){
            if(probe < ptr[idx]){
                probe = ptr[idx];
                max_p = idx;
            }
            if(train_out[idx]==1.0) p_trg = idx;
        }
        /*2- match*/
        is_ok=(max_p == p_trg);
        if(iter==1){
            /*determine if we get a good answer at first try*/
            if(is_ok==TRUE) NN_COUT(stdout," OK");
            else NN_COUT(stdout," NO");
        }
        if(iter>10239) break;/*failsafe number of wrong iteration*/ 
    }while((dEp > delta)||(!(is_ok==TRUE)));
    NN_COUT(stdout," N_ITER=%8i",iter);
    if(is_ok==TRUE) NN_COUT(stdout," SUCCESS!\n");
    else NN_COUT(stdout," FAIL!\n");
    fflush(stdout);
#ifdef _CUDA
    CUDA_FREE(train_gpu);
#endif /*_CUDA*/
    return dEp;
}

#undef KERN
