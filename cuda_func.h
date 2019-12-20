#ifndef FUNC_H
#define FUNC_H

void scuda_ann_allocate(_kernel *kernel,cudastreams *cudas);
void scuda_ann_weights_C2G(_kernel *kernel,cudastreams *cudas);
void scuda_ann_weights_G2C(_kernel *kernel,cudastreams *cudas);

void scuda_ann_forward(_kernel *kernel,cudastreams *cudas);
double scuda_ann_error(_kernel *kernel,double *train,cudastreams *cudas);
double scuda_ann_train(_kernel *kernel,double *train,cudastreams *cudas);
void scuda_ann_raz_momentum(_kernel *kernel,cudastreams *cudas);
double scuda_ann_train_momentum(_kernel *kernel,double *train,double moment,cudastreams *cudas);

#endif /*FUNC_H*/
