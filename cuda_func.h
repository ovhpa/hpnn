#ifndef FUNC_H
#define FUNC_H

void scuda_ann_forward_cublas(_kernel *kernel,cudastreams *cudas);
double scuda_ann_error(_kernel *kernel,double *train,cudastreams *cudas);
double scuda_ann_train_cublas(_kernel *kernel,double *train,cudastreams *cudas);

#endif /*FUNC_H*/
