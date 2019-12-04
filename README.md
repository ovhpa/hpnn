# hpnn
High performance neural networks

The hpnn is an (academic) essay on trying to understand and develop a simple framework for the optimization and training of artificial neural networks (ANN).
Plain C language is combined with numerical analysis tools and parallel programing over CPUs and GPUs to acheive some reasonable optimization.

This work is yet at a very, very early stage.

## Compile

1/ Select or create a `make.xxx` in the arch folder.
2/ replace the line that says `include arch/make.gcc.linux` with your `make.xxx` file:
```
include arch/make.xxx
```
3/ type:
```
make all
```
In the main directory (where the `Makefile` file is).

There is also a `debug` target (for.. debug) and a glib `gall` target for compatibility, with an associated `gdebug` for debuging glib target.
All target can be made using:
```
make everything
```

It is recomended to use or modify one of the available targets:
make.gcc.openblas: uses the openblas _parallel_ library with the gcc compiler. 
arch/make.icc.linux: uses the mkl _parallel_ library with the intel compiler.


## Basic ANN type (NN\_TYPE\_ANN)

The feed-forward simple ANN is the most simple topology. It is composed of:
* inputs: any number of inputs, which must be consistent with sample and test files, used for training and testing ANN, respectively;
* any number of layers, each of them composed of any number of neurons;
* an output layer, composed of a number of neurons equals to the number of outputs.

Basic shematics for NN\_TYPE\_ANN


The hpnn is optimized for:
- [x] Serial processing, using BLAS (lvl. 2);
- [x] openMP (multicore) processing, using parallel BLAS (openblas, intel mkl);
- [x] CUDA (multi-GPU) processing, using CUBLAS;
- [ ] MPI multi-nodes (and multi CPUs), using OPENMPI (v. 4)

A tutorial is available: tutorial.bash


_YES_ even the README.md is not finished...






