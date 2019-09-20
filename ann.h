#ifndef ANN_H
#define ANN_H

#define _2D_IDX(len,j,i) (len)*(j)+i

#ifndef ANN_UNROLL 
#define ANN_UNROLL 4
#endif /*ANN_UNROLL*/

#define DBG_TRACE(array,N) do{\
        acc=0.;\
        for(rdx=0;rdx<(N);rdx++) acc+=(array)[rdx];\
        fprintf(stdout,"#DBG: acc=%.15f\n",acc);\
}while(0)


/*deprecated?*/
typedef struct {
	UCHAR tag;		/*activation*/
	DOUBLE *weights;	/*weights*/
} _neuron;

typedef struct {
        UINT n_neurons;         /*number of neurons*/
//	_neuron *neurons;
	UINT n_inputs;          /*number of inputs*/
	DOUBLE *weights;	/*all weights for this layer*/
} _layer;

typedef struct {
	CHAR *name;		/*ANN name*/
        UINT n_inputs;		/*number of inputs*/
        DOUBLE *in;		/*input vector*/
        UINT n_hiddens;		/*number of hidden layers*/
        _layer *hiddens;	/*hidden layers*/
	UINT n_outputs;		/*number of outputs*/
        _layer output;		/*output layers*/
	DOUBLE *out;		/*output vector*/
	DOUBLE **dw;		/*weights momentum (when relevant)*/
} _kernel;

void ann_set_verbose();
void ann_set_dry();
int ann_init();
_kernel *ann_load(CHAR *f_kernel);
_kernel *ann_generate(UINT *seed,UINT n_inputs,UINT n_hiddens,UINT n_outputs,UINT *hiddens);
void ann_dump(_kernel *kernel,FILE *out);
void ann_kernel_run(_kernel *kernel);
DOUBLE ann_kernel_train(_kernel *kernel,const DOUBLE *train);
void ann_momentum_init(_kernel *kernel);
void ann_raz_momentum(_kernel *kernel);
void ann_empty_momentum(_kernel *kernel);
DOUBLE ann_kernel_train_momentum(_kernel *kernel,const DOUBLE *train,DOUBLE alpha);
DOUBLE ann_train_BPM(_kernel *kernel,DOUBLE *train_in,DOUBLE *train_out,DOUBLE alpha,DOUBLE delta);

#endif /*ANN_H*/
