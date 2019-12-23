/*
 * prepare_dif.c
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
/* ^^^^^^^^^^^^^^^^^^^^^^^This is an example processing RRUFF XRD database[1] */
/* -------------------------------------------- Hubert Okadome Valencia, 2019 */

/*[1] http://rruff.info/about/about_general.php */

/*we need to define atom_symb, space_groups at least once per program*/
#define NEED_ATOM_LIST
#define NEED_SG_LIST
#include "atom.def"
#include "sg.def"
#undef NEED_ATOM_LIST
#undef NEED_SG_LIST
#include "file_dif.h"

void dump_help(){
	fprintf(stdout,"********************************************\n");
	fprintf(stdout,"usage: pdif rruff_directory -i n_in -o n_out\n");
	fprintf(stdout,"********************************************\n");
	fprintf(stdout,"rruff_directory: where dif and raw directory\n");
	fprintf(stdout,"are located.\n");
	fprintf(stdout,"-i n_in: number of input samples -MANDATORY!\n");
	fprintf(stdout,"This is the interval number in which the XRD\n");
	fprintf(stdout,"diffraction pattern will be interpolated NOT\n");
	fprintf(stdout,"the number of input for neural network.  The\n");
	fprintf(stdout,"difference is extra input is added to the NN\n");
	fprintf(stdout,"for relative temperature (T/T0 ; T0=273.15).\n");
	fprintf(stdout,"-i n_out: number of outputs -ALSO MANDATORY!\n");
	fprintf(stdout,"It corresponds to  the neural network number\n");
	fprintf(stdout,"of outputs and SHOULD BE 230, ie. the number\n");
	fprintf(stdout,"of studied space groups to be found by NN...\n");
	fprintf(stdout,"********************************************\n");
	fprintf(stdout,"The default is that created the samples will\n");
	fprintf(stdout,"be written to the 'samples' directory, which\n");
	fprintf(stdout,"can be changed with the -s option.\n");
	fprintf(stdout,"********************************************\n");
	fprintf(stdout,"This code is released 'as is', within GPLv3.\n");
	fprintf(stdout,"available at:  https://github.com/ovhpa/hpnn\n");
	fprintf(stdout,"- project started 2019~            -- OVHPA.\n");
	fprintf(stdout,"********************************************\n");
}


int main (int argc, char *argv[]){
	UINT     is_ok;
	UINT  idx, jdx;
	UINT  n_inputs;
	UINT n_outputs;
	CHAR *ptr,*tmp;
	_dif      *dif;
	FILE  *dest_file;
	DIR_S *directory;
	CHAR  *rruff_dir;
	CHAR *sample_dir;
	CHAR   *curr_dir;
	CHAR  *curr_file;
	/*init*/
	n_inputs=0;
	n_outputs=0;
	curr_dir=NULL;
	rruff_dir=NULL;
	sample_dir=NULL;
	/*parse arguments*/
        if(argc<3) {
		/*minimum argument list is rruff_directory n_in n_out*/
		dump_help();
		return 1;
        }
	for(idx=1;idx<argc;idx++){
		if(argv[idx][0]=='-'){
			/*switch detected*/
			jdx=1;
			while(ISGRAPH(argv[idx][jdx])){
				switch (argv[idx][jdx]){
				case 'h':
					dump_help();
					return 0;
				case 'i':
					tmp=&(argv[idx][jdx]);
					if(!ISGRAPH(*(tmp+1))){
						/*we are having separated -i N*/
						idx++;
						tmp=&(argv[idx][0]);
						SKIP_BLANK(tmp);
						if(!ISDIGIT(*tmp)){
							fprintf(stderr,"syntax error: bad -i parameter!\n");
							dump_help();
							return 1;
						}
					}else{
						/*we have -iN*/
						if(!ISDIGIT(*(tmp+1))){
							fprintf(stderr,"syntax error: bad -i parameter!\n");
							dump_help();
							return 1;
						}
						tmp++;
					}
					GET_UINT(n_inputs,tmp,ptr);
					if(n_inputs==0){
						fprintf(stderr,"syntax error: bad -i parameter!\n");
						dump_help();
						return 1;
					}
					n_inputs+=1;/*temperature*/
					goto end_loop;/*no combination (-ih) is allowed*/
				case 'o':
					tmp=&(argv[idx][jdx]);
					if(!ISGRAPH(*(tmp+1))){
						/*we are having separated -o N*/
						idx++;
						tmp=&(argv[idx][0]);
						SKIP_BLANK(tmp);
						if(!ISDIGIT(*(tmp))){
							fprintf(stderr,"syntax error: bad -o parameter!\n");
							dump_help();
							return 1;
						}
					}else{
						/*we have -oN*/
						if(!ISDIGIT(*(tmp+1))){
							fprintf(stderr,"syntax error: bad -o parameter!\n");
							dump_help();
							return 1;
						}
						tmp++;
					}
					GET_UINT(n_outputs,tmp,ptr);
					if(n_outputs==0){
						fprintf(stderr,"syntax error: bad -o parameter!\n");
						dump_help();
						return 1;
					}
					goto end_loop;/*no combination (-oh) is allowed*/
				case 's':
					/*the sample directory*/
					tmp=&(argv[idx][jdx]);
					if(!ISGRAPH(*(tmp+1))){
						/*we are having separated -s dir*/
						idx++;
						tmp=&(argv[idx][0]);
						SKIP_BLANK(tmp);
					}else{
						/*we have -sdir*/
						tmp++;
					}
					STRDUP(tmp,sample_dir);
					goto end_loop;/*no combination (-sh) is allowed*/
				default:
					fprintf(stderr,"syntax error: unrecognized option!\n");
					dump_help();
					return 1;
				}
			}
		}else{
			/*this _have_ to be rruff directory*/
			if(rruff_dir!=NULL) {
				/*we already registered it!*/
				fprintf(stderr,"syntax error: too many parameters!\n");
				dump_help();
				return 1;
			}
			STRDUP(argv[idx],rruff_dir);
		}
end_loop:
		jdx=0;
	}
	if(sample_dir==NULL) STRDUP("./samples",sample_dir);
fprintf(stdout,">> received: %s -i %i -o %i -s %s\n",
		rruff_dir,n_inputs,n_outputs,sample_dir);
	/*check sample directory*/
	OPEN_DIR(directory,sample_dir);
	if(directory==NULL){
		fprintf(stderr,"ERROR: can't open directory: %s\n",sample_dir);
		return 1;
	}
	CLOSE_DIR(directory,is_ok);
	if(is_ok){
		fprintf(stderr,"ERROR: trying to close %s directory. IGNORED\n",
			sample_dir);
	}
	/*process*/
	STRCAT(curr_dir,rruff_dir,"/dif/");
	if(curr_dir==NULL){
		fprintf(stderr,"ERROR: STRCAT trouble?!\n");
		return 1;
	}
	OPEN_DIR(directory,curr_dir);
	if(directory==NULL){
		fprintf(stderr,"ERROR: can't open directory: %s\n",curr_dir);
		return 1;
	}
	FILE_FROM_DIR(directory,curr_file);
	while(curr_file!=NULL){
		/*process file!*/
		fprintf(stdout,"Processing file: %s\n",curr_file);
/**/
		STRCAT(tmp,curr_dir,curr_file);
		dif=read_dif(tmp);
		if(dif==NULL){
			fprintf(stderr,"ERROR:  reading %s file! SKIP\n",curr_file);
			FILE_FROM_DIR(directory,curr_file);
			continue;
		}
		if(dif->lambda==0.710730){
			fprintf(stderr,"ERROR:  file %s has wavelength of 0.710730! SKIP\n",
				curr_file);
			FILE_FROM_DIR(directory,curr_file);
			continue;
		}
		FREE(tmp);
		STRCAT(ptr,rruff_dir,"/raw/");
		STRCAT(tmp,ptr,curr_file);
		if(!read_raw(tmp,dif)){
			fprintf(stderr,"ERROR: reading %s file! SKIP\n",tmp);
			FILE_FROM_DIR(directory,curr_file);
			continue;
		}
		FREE(ptr);
		FREE(tmp);
		STRCAT(ptr,sample_dir,"/");
		STRCAT(tmp,ptr,curr_file);
		dest_file=fopen(tmp,"w");
		if(dest_file==NULL){
			fprintf(stderr,"ERROR: opening %s sample file for WRITE!\n",tmp);
			FILE_FROM_DIR(directory,curr_file);
			continue;
		}
		if(!dif_2_sample(dif,dest_file,n_inputs,n_outputs)){
			fprintf(stderr,"ERROR: writting %s sample file!\n",tmp);
		}
		fclose(dest_file);
		FREE(ptr);
		FREE(tmp);
		FREE(dif);
/**/
		FREE(curr_file);
		FILE_FROM_DIR(directory,curr_file);
	}
	CLOSE_DIR(directory,is_ok);
	if(is_ok){
		fprintf(stderr,"ERROR: trying to close %s directory. IGNORED\n",
			curr_dir);
	}
	return 0;
}
