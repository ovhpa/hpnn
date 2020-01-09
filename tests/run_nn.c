/*
+++ libhpnn - High Performance Neural Network library - run_nn test application +++
    Copyright (C) 2019  Okadome Valencia Hubert

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>
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
	_OUT(stdout,"****************************************\n");
	_OUT(stdout," usage: run_nn [-options] [input]       \n");
	_OUT(stdout,"****************************************\n");
	_OUT(stdout,"options:                               *\n");
	_OUT(stdout,"-h \tdisplay this help;                *\n");
	_OUT(stdout,"-v \tincrease verbosity;               *\n");
/*^^^ for openMP calculation ^^^*/
#ifdef _OMP
	_OUT(stdout,"-O \tnumber of openMP threads.         *\n");
	_OUT(stdout,"-B \tnumber of BLAS threads (MKL).     *\n");
#endif
/*^^^ CUDA specific ^^^*/
#ifdef _CUDA
	_OUT(stdout,"-S \tnumber of CUDA streams.           *\n");
#endif /*_CUDA*/
	_OUT(stdout,"****************************************\n");
	_OUT(stdout,"input: neural network def file contains*\n");
	_OUT(stdout,"neural network definitions & topologies*\n");
	_OUT(stdout," -can contain weight values or seed for*\n");
	_OUT(stdout,"the random generation of weights.      *\n");
	_OUT(stdout,"****************************************\n");
	_OUT(stdout,"Code released 'as is' /GPLv3, available*\n");
	_OUT(stdout,"here: https://github.com/ovhpa/hpnn    *\n");
	_OUT(stdout,"- project started 2019~       -- OVHPA.*\n");
	_OUT(stdout,"****************************************\n");
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
	/*init all*/
	_NN(init,all)();
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
						_NN(deinit,all)();
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
_OUT(stderr,"syntax error: bad -O parameter!\n");
								dump_help();
								goto FAIL;
							}
						}else{
							/*we have -ON*/
							if(!ISDIGIT(*(tmp+1))){
_OUT(stderr,"syntax error: bad -O parameter!\n");
								dump_help();
								goto FAIL;
							}
							tmp++;
						}
						GET_UINT(n_o,tmp,ptr);
						if(n_o==0){
_OUT(stderr,"syntax error: bad -O parameter!\n");
							dump_help();
							goto FAIL;
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
_OUT(stderr,"syntax error: bad -B parameter!\n");
								dump_help();
								goto FAIL;
							}
						}else{
							/*we have -BN*/
							if(!ISDIGIT(*(tmp+1))){
_OUT(stderr,"syntax error: bad -B parameter!\n");
								dump_help();
								goto FAIL;
							}
							tmp++;
						}
						GET_UINT(n_b,tmp,ptr);
						if(n_b==0){
_OUT(stderr,"syntax error: bad -B parameter!\n");
							dump_help();
							goto FAIL;
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
_OUT(stderr,"syntax error: bad -S parameter!\n");
								dump_help();
								goto FAIL;
							}
						}else{
							/*we have -SN*/
							if(!ISDIGIT(*(tmp+1))){
_OUT(stderr,"syntax error: bad -S parameter!\n");
								dump_help();
								goto FAIL;
							}
							tmp++;
						}
						GET_UINT(n_s,tmp,ptr);
						if(n_s==0){
_OUT(stderr,"syntax error: bad -S parameter!\n");
							dump_help();
							goto FAIL;
						}
						_NN(set,cuda_streams)(n_s);
						goto next_arg;/*no combination is allowed*/
#endif /*_CUDA*/
					default:
						_OUT(stderr,"syntax error: unrecognized option!\n");
						dump_help();
						goto FAIL;
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
	/*load configuration file*/
	neural=_NN(conf,load)(nn_filename);
	if(neural==NULL) {
		_OUT(stderr,"FAILED to read NN configuration file! (ABORTING)\n");
		goto FAIL;
	}
	/*setup done, run kernel*/
	_NN(kernel,run)(neural);
	/*deinit*/
	_NN(deinit,all)();
	return 0;
FAIL:
	_NN(deinit,all)();
	return -1;
}
