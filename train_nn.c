#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

/* Artificial Neuron Network preparation and training. */
/* --------------------- Hubert Okadome Valencia, 2019 */

/*we need to define atom_symb, space_groups at least once per program*/
#define NEED_ATOM_LIST
#define NEED_SG_LIST
#include "common.h"
#undef NEED_ATOM_LIST
#undef NEED_SG_LIST
#include "ann.h"
#include "nn.h"
#include "file.h"

void dump_help(){
	fprintf(stdout,"***********************************\n");
	fprintf(stdout,"usage:  train_nn [-options] [input]\n");
	fprintf(stdout,"***********************************\n");
	fprintf(stdout,"options:\n");
	fprintf(stdout,"-h \tdisplay this help;\n");
	fprintf(stdout,"-v \tincrease verbosity;\n");
	fprintf(stdout,"-x \tdiscard results.\n");
#ifdef _OMP
	fprintf(stdout,"-O \tnumber of openMP threads.\n");
	fprintf(stdout,"-B \tnumber of BLAS threads (MKL).\n");
#endif
#ifdef _CUDA
	fprintf(stdout,"-S \tnumber of CUDA streams.\n");
#endif
	fprintf(stdout,"***********************************\n");
	fprintf(stdout,"input:     neural network .def file\n");
	fprintf(stdout,"contains the network definition and\n");
	fprintf(stdout,"topology. May contain weight values\n");
	fprintf(stdout,"or context for a random generation.\n");
	fprintf(stdout,"***********************************\n");
	fprintf(stdout,"Code released 'as is' within GPLv3.\n");
	fprintf(stdout,"here: https://github.com/ovhpa/hpnn\n");
	fprintf(stdout,"- project started 2019~   -- OVHPA.\n");
	fprintf(stdout,"***********************************\n");
}
int main (int argc, char *argv[]){
	UINT  idx, jdx;
	FILE   *output;
#ifdef _OMP
	UINT  n_o, n_b;
#endif /*_OMP*/
#ifdef _CUDA
	UINT n_s;
#endif /*_CUDA*/
#if defined (_OMP) || defined (_CUDA)
	CHAR *tmp,*ptr;
#endif
	CHAR *nn_filename = NULL;
	nn_def    *neural = NULL;
/*parse arguments*/
	if(argc<2) {
		/*This is the default: neural network definition is taken from file nn.conf */
		STRDUP("./nn.conf",nn_filename);
	}else{
		/*find some switch, if any*/
		idx=1;
		while(idx<argc){
			if(argv[idx][0]=='-'){
				/*switch detected*/
				jdx=1;
				while(ISGRAPH(argv[idx][jdx])){
					switch (argv[idx][jdx]){
					case 'h':
						dump_help();
						return 0;
					case 'v':
						_NN(inc,verbose)();
						jdx++;
						break;
					case 'x':
						_NN(toggle,dry)();
						jdx++;
						break;
#ifdef _OMP
					case 'O':
						tmp=&(argv[idx][jdx]);
						if(!ISGRAPH(*(tmp+1))){
							 /*we are having separated -O N*/
							idx++;
							tmp=&(argv[idx][0]);
							SKIP_BLANK(tmp);
							if(!ISDIGIT(*(tmp))){
								fprintf(stderr,"syntax error: bad -O parameter!\n");
								dump_help();
								return 1;
							}
						}else{
							/*we have -ON*/
							if(!ISDIGIT(*(tmp+1))){
								fprintf(stderr,"syntax error: bad -O parameter!\n");
								dump_help();
								return 1;
							}
							tmp++;
						}
						GET_UINT(n_o,tmp,ptr);
						if(n_o==0){
							fprintf(stderr,"syntax error: bad -O parameter!\n");
							dump_help();
							return 1;
						}
						_NN(set,omp_threads)(n_o);
						goto next_arg;/*no combination is allowed*/
					case 'B':
						tmp=&(argv[idx][jdx]);
						if(!ISGRAPH(*(tmp+1))){
							/*we are having separated -B N*/
							idx++;
							tmp=&(argv[idx][0]);
							SKIP_BLANK(tmp);
							if(!ISDIGIT(*(tmp))){
								fprintf(stderr,"syntax error: bad -B parameter!\n");
								dump_help();
								return 1;
							}
						}else{
							/*we have -BN*/
							if(!ISDIGIT(*(tmp+1))){
								fprintf(stderr,"syntax error: bad -B parameter!\n");
								dump_help();
								return 1;
							}
							tmp++;
						}
						GET_UINT(n_b,tmp,ptr);
						if(n_b==0){
							fprintf(stderr,"syntax error: bad -B parameter!\n");
							dump_help();
							return 1;
						}
						_NN(set,omp_blas)(n_b);
						goto next_arg;/*no combination is allowed*/
#endif /*_OMP*/
#ifdef _CUDA
					case 'S':
						tmp=&(argv[idx][jdx]);
						if(!ISGRAPH(*(tmp+1))){
							/*we are having separated -S N*/
							idx++;
							tmp=&(argv[idx][0]);
							SKIP_BLANK(tmp);
							if(!ISDIGIT(*(tmp))){
								fprintf(stderr,"syntax error: bad -S parameter!\n");
								dump_help();
								return 1;
							}
						}else{
							/*we have -SN*/
							if(!ISDIGIT(*(tmp+1))){
								fprintf(stderr,"syntax error: bad -S parameter!\n");
								dump_help();
								return 1;
							}
							tmp++;
						}
						GET_UINT(n_s,tmp,ptr);
						if(n_s==0){
							fprintf(stderr,"syntax error: bad -S parameter!\n");
							dump_help();
							return 1;
						}
						_NN(set,cuda_streams)(n_s);
						goto next_arg;/*no combination is allowed*/
#endif /*_CUDA*/
					default:
						fprintf(stderr,"syntax error: unrecognized option!\n");
						dump_help();
						return 1;
					}
				}
			}else{
				/*not a switch, then must be a file name!*/
				STRDUP(argv[idx],nn_filename);
				goto next_arg;
				/*rest of the command line is ignored!*/
			}
next_arg:
		idx++;
		}
	}
	if(nn_filename==NULL) STRDUP("./nn.conf",nn_filename);
	/*initialize ann*/
	_NN(init,all)();
	/*load configuration file*/
	neural=read_conf(nn_filename);
	if(neural==NULL) {
		fprintf(stderr,"FAILED to read NN configuration file! (ABORTING)\n");
		return 1;
	}
	/*setup done save a temporary kernel*/
	output=fopen("kernel.tmp","w");
	if(output==NULL){
		fprintf(stderr,"FAILED to open kernel.tmp for WRITE!\n");
		return 1;
	}
	_NN(kernel,dump)(neural,output);
	fclose(output);
	/*perform training*/
	if(!_NN(kernel,train)(neural)){
		fprintf(stderr,"FAILED to train kernel!\n");
		return 1;
	}
	/*save the trained kernel*/
	output=fopen("kernel.opt","w");
	if(output==NULL){
		fprintf(stderr,"FAILED to open kernel.tmp for WRITE!\n");
		return 1;
	}
	_NN(kernel,dump)(neural,output);
	fclose(output);	
	/*dump config*/
//	dump_conf(stdout,neural);
	/*deinit*/
	_NN(deinit,all)();
	return 0;
}
