/*
 * run_nn.c
 *
 * Copyright (C) 2019 - Hubert Valencia
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

/* Artificial Neuron Network ---------------------- preparation and training. */
/* -------------------------------------------- Hubert Okadome Valencia, 2019 */

#include <libhpnn.h>

void dump_help(){
	fprintf(stdout,"****************************************\n");
	fprintf(stdout," usage: run_nn [-options] [input]       \n");
	fprintf(stdout,"****************************************\n");
	fprintf(stdout,"options:                               *\n");
	fprintf(stdout,"-h \tdisplay this help;                *\n");
	fprintf(stdout,"-v \tincrease verbosity;               *\n");
/*^^^ for openMP calculation ^^^*/
#ifdef _OMP
	fprintf(stdout,"-O \tnumber of openMP threads.         *\n");
	fprintf(stdout,"-B \tnumber of BLAS threads (MKL).     *\n");
#endif
/*^^^ CUDA specific ^^^*/
#ifdef _CUDA
	fprintf(stdout,"-S \tnumber of CUDA streams.           *\n");
#endif /*_CUDA*/
	fprintf(stdout,"****************************************\n");
	fprintf(stdout,"input: neural network def file contains*\n");
	fprintf(stdout,"neural network definitions & topologies*\n");
	fprintf(stdout," -can contain weight values or seed for*\n");
	fprintf(stdout,"the random generation of weights.      *\n");
	fprintf(stdout,"****************************************\n");
	fprintf(stdout,"Code released 'as is' /GPLv3, available*\n");
	fprintf(stdout,"here: https://github.com/ovhpa/hpnn    *\n");
	fprintf(stdout,"- project started 2019~       -- OVHPA.*\n");
	fprintf(stdout,"****************************************\n");
}
int main (int argc, char *argv[]){
	UINT idx,jdx;
	nn_def *neural;
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
/*parse arguments*/
	if(argc>1){
		idx=1;
		while(idx<argc){
			if(argv[idx][0]=='-'){
				/*switch detected*/
				jdx=1;
				while(ISGRAPH(argv[idx][jdx])){
					switch (argv[idx][jdx]){
					case 'h':
						dump_help();
						return 0;/*nothing happen after help*/
					case 'v':
						_NN(inc,verbose)();
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
	if(nn_filename==NULL){
		/*default config file*/
		STRDUP("./nn.conf",nn_filename);
	}
	/*initialize ann*/
	_NN(init,all)();
	/*load configuration file*/
	neural=_NN(conf,load)(nn_filename);
	if(neural==NULL) {
		_OUT(stderr,"FAILED to read NN configuration file! (ABORTING)\n");
		return 1;
	}
	/*setup done, run kernel*/
	_NN(kernel,run)(neural);
	/*deinit*/
	_NN(deinit,all)();
	return 0;
}
