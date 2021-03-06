# hpnn

High performance neural networks.\
currently: [v. 0.2](https://github.com/ovhpa/hpnn/releases/tag/v0.2) (_very_ alpha - don't expect anything but troubles).

Please check the [Wiki](https://github.com/ovhpa/hpnn/wiki) for implementation details.

## Introduction

The hpnn library is an (academic) essay on trying to develop a simple, yet optimized, library for the "on-the-fly" training of artificial neural networks (ANNs).
The goal is to provide any given program an access to the creation, training, and use of ANN through a very simplistic API, creating ANNs as the program perform its regular calculations.

The novelty (or oddness) of our method is that the training of ANNs is aimed to be performed while the actual calculation are performed, for example, if we consider a normal ANN creation and optimization such as:

![regular](res/regular.png)

A database is build using the many output produced by running a program (in blue).
Then, an ANN is trained with the data from the database (in green).
When the training is deem good enough, a final ANN can be use (in orange) to produce a result (in red) from input almost instantly.
The quality of the output of course depend on the quality of the database.
Usually, the training process is quite long, and obtaining the database can be problematic: high quality database usually require a huge amount of time and/or some significant financial.
Note that in such "classic" scenario, the demonstration `train_nn` and `run_nn` programs from libhpnn can be used.

The goal libhpnn is trying to achieve can be viewed as:

![hpnn](res/hpnn.png)

In this case, an ANN in enriched on the fly each time the program is run.
As the ANN quality improved, the ANN can be used together with the program as a was to optimize its performance.
Of course, the ANN part has to be "corrected" by the program on each run until such correction prove unnecessary.
For example, if one consider using an ANN in place of a self consistent (iterative) calculation:
* at first the ANN is poorly optimized, the ANN part is then no better than a random initialization of the calculation, and the whole calculation should take a little _longer_ than it would without the ANN part.
* as the ANN improved, the iterative part will reduce, since the initialization will be a "better" guess (than a random one). In this part, the training and preparation of ANN will challenge the reduction of the iterative part.
* when the ANN is getting efficient, the iterative part will reduce significantly (ideally tending to a single iteration). At that point the training of ANN will also be reduce and can even be switched off. The remaining iterative part will then act as a "safety net" for the (still possible) failures of the ANN.

The HPNN library is developed "from scratch" (which means from accessible books and literature) and is provided _as is_ in the hope that it will be helpful (see [License](LICENSE)).

The library is written in plain C, and optimized for use in parallel systems with several layers:
* MPI - for the inter-nodes computations
* OpenMP - for the intra-nodes computations
* CUDA - for GPGPU computations

Combination of the above is also provided: MPI/OpenMP and MPI/CUDA. Some other technologies (ie. openCL) may be added later.

This work is still at a very, very early stage!
Some _demonstration_ wrapper are provided which shows an example of integration with the library: these are not the programs you are looking for ;)
* `train_nn`: demonstrate how to train a network using MPI/OpenMP/CUDA, taking training set from a sample directory, and creating a kernel.opt file which contains the definition of the optimized ANN.
* `run_nn`: demonstrate how to run a specific ANN against a testing set from a test directory.

Additionally, some tutorials are provided, to understand libhpnn API.

1- RRUFF X-ray diffraction (XRD) database, shows how to format inputs and outputs for use with libhpnn. This tutorial will produce a ANN for ascribing the space group symmetry of a crystal based only on its experimental XRD pattern (given a given temperature parameter). This short tutorial uses `train_nn` and `run_nn` programs.
2- incoming tutorials...

The tutorials are detailed [here](tutorials/README.md).


## License ![logo](https://www.gnu.org/graphics/gplv3-or-later.png)

* libhpnn is a free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
* libhpnn is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public [License](LICENSE) for more details.
* libhpnn should include a copy of the GNU General Public License.  If not, see <[https://www.gnu.org/licenses/](https://www.gnu.org/licenses/)>.


## Compile / Install

Usually the `autogen.sh` script will find the best option for compiling libhpnn.
In such case, it will produce a `configure` executable, which can be run to prepare the Makefiles.
After that, given that no error is encountered, a `make all` command will compile the library and a `make install` will install it into the directory specified by the `--prefix=` option (which default to /usr/local on GNU/Linux). Depending on where the installation is performed, the latter installation could require elevated privilege (such as root).

Additionally, before the `make install` step, one may want to test the libhpnn library. The `make check` command will perform several tests, depending on the library capability. Should one of the test FAIL, please check external compilers and library (C compiler, MPI, openMP, CUDA) prior to filling a BUG report.

One may want to use specific compiler, libraries, or capabilities settings for the compilation of libhpnn. In such cases, please consult the [install](INSTALL) file.

