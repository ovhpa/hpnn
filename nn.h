#ifndef NN_H
#define NN_H

typedef enum {
	NN_TYPE_ANN = 0,	/*feed-forward, activation base nn*/
	NN_TYPE_LNN = 1,	/*feed-forward, activation (hidden) + linear (output)*/
	NN_TYPE_PNN = 2,	/*feed-forward, probability base nn*/
	NN_TYPE_UKN =-1,	/*unknown*/
} nn_type;

typedef enum {
	NN_TRAIN_BP  = 0,	/*simple back-propagation*/
	NN_TRAIN_BPM = 1,	/*back-propagation with momentum*/
	NN_TRAIN_CG  = 2,	/*conjugate gradients*/
	NN_TRAIN_UKN =-1,	/*unknown*/
} nn_train;

typedef struct {
	CHAR     *name;		/*name of the NN*/
	nn_type   type;		/*NN type*/
	BOOL need_init;		/*require initialization*/
	UINT      seed;		/*seed used in case of initialization*/
	void   *kernel;		/*NN kernel (weights,input,output)*/
	CHAR *f_kernel;		/*kernel filename*/
	nn_train train;		/*training type*/
	CHAR  *samples;		/*samples directory (for training)*/
	CHAR    *tests;		/*tests directory (for validation)*/
} nn_def;

#define _NN(a,b) nn_##a##_##b
/*init*/
void _NN(inc,verbose)();
void _NN(toggle,dry)();
int _NN(init,all)();
int _NN(deinit,all)();
#ifdef _CUDA
#ifdef _CUBLAS
cublasHandle_t _NN(get,cuda_handle)();
#else /*_CUBLAS*/
int _NN(get,cuda_handle)();
#endif /*_CUBLAS*/
cudastreams *_NN(get,cudas)();
void _NN(set,cuda_streams)(UINT n_streams);
#endif /*_CUDA*/
#ifdef _OMP
void _NN(set,omp_threads)(UINT n);
UINT _NN(get,omp_threads)();
void _NN(set,omp_blas)(UINT n);
UINT _NN(get,omp_blas)();
#endif /*_OMP*/

/*ACCESS*/
UINT _NN(get,n_inputs)(nn_def *neural);
UINT _NN(get,n_hiddens)(nn_def *neural);
UINT _NN(get,n_outputs)(nn_def *neural);
UINT _NN(get,h_neurons)(nn_def *neural,UINT layer);
/*GENERAL*/
BOOL _NN(kernel,generate)(nn_def *neural,UINT n_inputs,UINT n_hiddens,UINT n_outputs,UINT *hiddens);
BOOL _NN(kernel,load)(nn_def *neural);

void _NN(kernel,dump)(nn_def *neural, FILE *output);


BOOL _NN(kernel,train)(nn_def *neural);

void _NN(kernel,run)(nn_def *neural);


#endif /*NN_H*/
