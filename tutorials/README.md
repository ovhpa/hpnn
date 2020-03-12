# The Tutorials section

This section will contain some basic tutorial in order to demonstrate the use of libhpnn.
The examples that are provided are, by no means, intended to be _meaningful_ ones.
Hopefully, though, it could lead to applications that will be.

## API tutorials

### RRUFF X-ray diffraction database

This simple tutorial relies on the RRUFF X-ray diffraction (XRD) database.
An extra program is required to convert from the original database format to the simple input/ouput format used by libhpnn.

This tutorial is provided in order to understand libhpnn API, it is by no mean a guide of good practice in producing reasonable model for XRD predicting the geometry of a crystal using XRD pattern.
-> in case you are interested by such application, I recommend reading the following paper: 
P.M. Vecsei, K. Choo, J. Chang, and T. Neupert, Phys. Rev. B 99, 245120 (2019). A preprint is available [here](https://arxiv.org/abs/1812.05625)

A tutorial is provided for this type in the form of a bash script, [tutorial.bash](tutorials/ann/tutorial.bash).\
Following is a walk-through of what the script is doing. 

The tutorial relies on two executable programs that are described thereafter: `run_nn` and `train_nn`, for running and training a NN, respectively.

After `libhpnn` have been successfully prepared, the [tutorial.bash](tutorials/ann/tutorial.bash) script can be launched.\
Here are the step that are executed in the script:

1. The RRUFF database is downloaded (on press of the `Y` key, be careful it is quite sensitive).
2. The files are renamed to follow a more practical R[number].txt scheme - in which number it the experiment number.
3. files are process by the `pdif` program that will translate it to a raw format that can be read by `train_nn` and `run_nn`.
4. A first training is done from a randomly generated NN kernel.
5. 10 round of optimization are then performed on this kernel.
6. Finally a run of the kernel on data is performed to check if it was optimized.

In this really simplistic example, we have chose to represent an XRD diffraction pattern as a 1D data. Each value represent the integral value of XRD counts between a regular interval between 5 and 90 degree. The resulting array produced by `pdif` have 850 values, which means each value correspond to the integration over a 0.1 degree interval.

The output arrays was chosen to represent the space-group of the crystal system for which the XRD pattern was recorded. There are 230 space group considered, each of them are given the value -1.0 (indicating a wrong result) or 1.0 (positive result).

To acheive better "results", the temperature was added as a parameter, and we chose to add it directly to the input array. Therefore the input array have 851 values, the first one being temperature.

Note that the temperature was added as a relative value, that is a value in Kelvin divided by 273.15 which is 0C in Kelvin. It is generally a good idea to use relative values and to make sure all values in the input array are within the same order of magnitude. To this effect, the value of XRD count integrals is also scaled so that the maximum is 1.0 at the highest.

After training the `run_nn` program shows that training have been successful by correctly ascribing each structure its space group (minus some few failure).


*Of course, there is now meaning in training and running a NN kernel with the exact same sample!*\
This is just a demonstration of a process flow (training/running) intended to understand the functionning of `libhpnn`.

NOTE: by changing the line 472 of the `file__dif.c` file from
```
		else fprintf(dest," -1.0");
```
to
```
		else fprintf(dest," 0.0");
```
and the line 130 of the `tutorial.bash` file from
```
echo "[type] ANN" >> nn.conf
```
to
```
echo "[type] SNN" >> nn.conf
```
one can actually use the SOFTMAX type (NN_TYPE_SNN) of neural network.\
In such case, the output is given as a 'probabilistic-like' result, with the guessed ascribed value given as a percent. 


## RUN and TRAIN programs

The `train_nn` and `run_nn` programs are the most basic use of libhpnn, and are not part of tutorials, but it can be useful to examine how they are linked to the library.\
_Since our goal is not to train a batch of samples efficiently, but to optimize a NN 'on the fly', this 2 programs are provided as an illustration only._ If you are looking for an *optimized* way to train a NN using batch techniques, please see other (very performant) programs such as [chainer](https://github.com/chainer/chainer) and others.

### The RUN program

In `run_nn` we only need a very minimal process to be executed:
1. initialize the library;
2. load the ANN and its parameters;
3. apply the ANN to the samples' input;
4. verify wether the output produce by the ANN is consistent with the one from the sample file.
5. de-initialize the library and exit.

#### 1. initialization

The library is first initialized using the global `_NN(init,all)(0)` call.
With this, each element (OMP, MPI, CUDA, BLAS) of the library will be initialized, depending on the library capabilities (ie. how the library was compiled). Note that, should libhpnn be linked to a program that is already initializing elements on its own, a selective initialization would have to be done.
Once initialized, the program adjusts the number of threads to the values provided by user. The variables `n_o`, `n_b`, and `n_s` are given using the `-O`, `-B`, and `-S` switches of `run_nn`, respectively. They are passed to libhpnn using:
```
_NN(set,omp_threads)(n_o);
_NN(set,omp_blas)(n_b);
_NN(set,cuda_streams)(n_s);
```
which sets the number of threads for OpenMP, BLAS, and CUDA use, respectively.\
Note that we do not need to set the number of threads that are going to be used by MPI (when available). This number is usually set at runtime by using a command in the form `mpirun -np X run_nn`, in which X is to be replaced by the number of MPI threads.
In a future implementation of libhpnn, it will be possible to _decrease_ that number using `_NN(set,mpi_tasks)(n_t)`, however the `run_nn` program doesn't require this.

#### 2. loading ANN 

The ANN is loaded indirectly, by the use of a configuration file. This configuration file will define a variable of `nn_def` type.
```
nn_def *neural;
```
After initialization has been done, the configuration can be loaded by libhpnn.
```
neural=_NN(load,conf)(nn_filename);
```
In this process, the file containing the ANN definition, and the location of the test directory containing the samples to test are provided.\
The exact content of this configuration file is provided here. 
For example, the content of the file can be:
```
[name] example
[type] ANN
[init] generate
[seed] 10958
[input] 851
[hidden] 64 64
[output] 230
[train] BPM
[sample_dir] ./samples
[test_dir] ./tests
```
in which we will briefly describe each field.

`[name]` is the optional name of the ANN (any text is allowed).\
`[type]` is the type of ANN used. Here ANN refers to `NN_TYPE_ANN`.
Please check the [Wiki](https://github.com/ovhpa/hpnn/wiki/ANN) for details on each ANN type.\
`[init]` should either be the name of the ANN kernel or the word `generate` if a start from a randomly generated neural network is required.\
`[seed]` is the seed that will be used to initialize the random number generator. If that value is zero, seed will be initialized with a number depending on the date - which mean that two consecutive runs will leads to different results.\
`[input]` is the number of input values used in the sample files and in the kernel definition.\
`[hidden]` is the number of neurons in each hidden layer. In above example, there are 2 hidden layers, each containing 64 neurons.\
`[output]` is the number of output values used in the sample files and in the kernel definition.\
`[train]` is the selected training type. Note that for the `run_nn` program, this field will not be used, but it will be checked for correctness. `BPM` here stands for 'back-propagation with momentum' training types. A description for each type can be found in the [Wiki](https://github.com/ovhpa/hpnn/wiki).\
`[sample_dir]` is the directory which contains the sample files used for training the ANN. It is not checked with the `run_nn` programs.\
`[test_dir]` is the directory containing the sample files for testing the ANN. Each file in that directory will be tested by `run_nn`.

#### 3. running ANN

#### 4. verify output

#### 5. de-initialization


### the TRAIN program

Compared to the `run_nn` program, `train_nn` require slightly different steps:
1. initialize the library;
2. load (or generate) the ANN and load its parameters;
3. train ANN with all samples; so that ANN applied to each sample produce is close to its intended output.
4. de-initialize the library and exit.

#### 1. initialization

#### 2. loading ANN

#### 3. Train the ANN

#### 5. de-initialization


