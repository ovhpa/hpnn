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
	fprintf(stdout,"usage:    run_nn [-options] [input]\n");
	fprintf(stdout,"***********************************\n");
	fprintf(stdout,"options:\n");
	fprintf(stdout,"-h \tdisplay this help;\n");
	fprintf(stdout,"-v \tincrease verbosity;\n");
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
	UINT idx,jdx;
	nn_def *neural;
	CHAR *nn_filename = NULL;
/*parse arguments*/
	if(argc<2) {
		/*This is the default: neural network definition is taken from file nn.conf */
		STRDUP("./nn.conf",nn_filename);
	}else{
		/*find some switch, if any*/
		for(idx=1;idx<argc;idx++){
			if(argv[idx][0]=='-'){
				/*switch detected*/
				jdx=1;
				while(ISGRAPH(argv[idx][jdx])){
					switch (argv[idx][jdx]){
					case 'h':
						dump_help();
						return 0;
					case 'v':
						ann_set_verbose();
						jdx++;
						break;
					default:
						fprintf(stderr,"syntax error: unrecognized option!\n");
						dump_help();
						return 1;
					}
				}
			}else{
				/*not a switch, then must be a file name!*/
				STRDUP(argv[idx],nn_filename);
				/*rest of the command line is ignored!*/
			}
		}
	}
	/*load configuration file*/
	neural=read_conf(nn_filename);
	if(neural==NULL) {
		fprintf(stderr,"FAILED to read NN configuration file! (ABORTING)\n");
		return 1;
	}
	/*setup done, run kernel*/
	_NN(kernel,run)(neural);
	return 0;
}
