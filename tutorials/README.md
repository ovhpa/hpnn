# The Tutorials section

This section will contain some basic tutorial in order to demonstrate the use of libhpnn.
The examples that are provided are, by no means, intended to be _meaningful_ ones.
Hopefully, though, it will lead to application that will be.

I will try to provide a basic tutorial for all different types of NN.

Another interesting - yet simple - example of interface with the hpnn library API is the tests programs that are provided in the tests directory. Unlike tutorials, however, there will be no explanation on the simple way they are linked to the library, but user might gain some insights by looking at the code itself.


## Basic ANN type (NN\_TYPE\_ANN)

This it the most basic, feedforward, NN.
It consists of a vector of inputs, one or several hidden layers, and one ouput layer.
each layer, being it hidden or output, has the following structure:
* a matrix containing the layers weights, and
* a vector containing the intermediate result.
In the case of output layer, obviously, the intermediate result is actually the output result of the NN.



