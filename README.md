# hpnn
High performance neural networks

The hpnn library is an (academic) essay on trying to develop a simple, yet optimized, library for the "on-the-fly" training of artificial neural networks (ANNs).
The goal is to provide any given program an access to the creation, training, and use of ANN through a very simplistic API, creating ANNs as the program perform its regular calculations.

The library is written in plain C, and optimized for use in parallel systems with several layers:
* MPI - for the inter-nodes computations
* OpenMP - for the intra-nodes computations
* CUDA - for GPGPU computations

Some other technologies may be added later.

This work is still at a very, very early stage!
Some _demonstration_ wrapper are provided which shows an example of integration with the library: these are not the programs you are looking for ;)
* train_nn: demonstrate how to train a network using MPI/OpenMP/CUDA, taking training set from a sample directory, and creating a kernel.opt file which contains the definition of the optimized ANN.
* run_nn: demonstrate how to run a specific ANN against a testing set from a test directory.

Additionally, in order to produce a valid set of input/ouput to feed an ANN, a tutorial presentation is available which will download an X-ray diffraction (XRD) database and create two programs to format it into a simple format. It consist in a script that will do most of the demonstration by itself. The full explanation of this tutorial is available here.
This tutorial is provided in order to understand libhpnn API, it is by no mean a guide of good practice in producing reasonable model for XRD predicting the geometry of a crystal using XRD pattern.
-> in case you are interested by such application, I recommend reading the following paper: 
P.M. Vecsei, K. Choo, J. Chang, and T. Neupert, Phys. Rev. B 99, 245120 (2019). A preprint is available [here](https://arxiv.org/abs/1812.05625)

## License

![logo](https://www.gnu.org/graphics/gplv3-or-later.png)
libhpnn is a free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
libhpnn is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public [License](LICENSE) for more details.
libhpnn should include a copy of the GNU General Public License.  If not, see <[https://www.gnu.org/licenses/](https://www.gnu.org/licenses/)>.


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
* any number of layers (at least 1), each of them composed of any number of neurons;
* an output layer, composed of a number of neurons equals to the number of outputs.

Basic schematics for NN\_TYPE\_ANN

X

The hpnn is optimized for:
- [x] Serial processing, using BLAS (lvl. 2);
- [x] openMP (multicore) processing, using parallel BLAS (openblas, intel mkl);
- [x] CUDA (multi-GPU) processing, using CUBLAS;
- [x] MPI multi-nodes (and multi CPUs), using OPENMPI (v. 4)
- [x] MPI/OpenMP combination (for multi-nodes/multi-CPUs/multi-cores configurations.
- [ ] MPI/CUDA for GPGPU capable multi-nodes (work in progress).



_YES_ even the README.md is not finished...






