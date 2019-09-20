#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "common.h"

#include "nn.h"

#include "file.h"

#define STR_CLEAN(pointer) do{\
	CHAR *_ptr=pointer;\
	while(*_ptr!='\0'){\
		if(*_ptr=='\t') *_ptr='\0';\
		if(*_ptr==' ') *_ptr='\0';\
		if((*_ptr=='\n')||(*_ptr=='#')) *_ptr='\0';\
		else _ptr++;\
	}\
}while(0)

/*load neural network definition file*/
nn_def *read_conf(CHAR *filename){
#define FAIL read_conf_fail
	PREP_READLINE();
	CHAR *line=NULL;
	CHAR *ptr,*ptr2;
	UINT *parameter;
	UINT *n_hiddens;
	UINT64 allocate;
	nn_def  *neural;
//	_kernel *kernel;
	BOOL is_ok;
	UINT   idx;
	FILE   *fp;
	/*init*/
	allocate=0;
	n_hiddens=NULL;
	ALLOC_REPORT(neural,1,nn_def,allocate);
	neural->need_init=FALSE;
	neural->train=NN_TRAIN_UKN;
	neural->type=NN_TYPE_UKN;
	ALLOC(parameter,3,UINT);
	/**/
	fp=fopen(filename,"r");
	if(!fp){
		fprintf(stderr,"Error opening configuration file: %s\n",filename);
		goto FAIL;
	}
	READLINE(fp,line);/*first line is usually a comment*/
	do{
		ptr=STRFIND("[name",line);
		if(ptr!=NULL){
			/*get name {any string}*/
			ptr+=6;SKIP_BLANK(ptr);
			/*process line (remove comment and \n)*/
			STR_CLEAN(ptr);
			STRDUP_REPORT(ptr,neural->name,allocate);
		}
		ptr=STRFIND("[type",line);
		if(ptr!=NULL){
			/*get type {"ANN", "PNN", ...}*/
			ptr+=6;SKIP_BLANK(ptr);
			switch (*ptr){
			case 'L':
				neural->type=NN_TYPE_LNN;
				break;
			case 'P':
				neural->type=NN_TYPE_PNN;
				break;
			case 'A':
				/*fallthrough*/
			default:
				neural->type=NN_TYPE_ANN;
				break;
			}
		}
		ptr=STRFIND("[init",line);
		if(ptr!=NULL){
			/*get init {"generate", "file"}*/
			ptr+=6;SKIP_BLANK(ptr);
			if((STRFIND("generate",line)!=NULL)||(STRFIND("GENERATE",line)!=NULL)){
fprintf(stdout,"NN generating kernel!\n");
				neural->need_init=TRUE;
			}else{
fprintf(stdout,"NN loading kernel!\n");
				neural->need_init=FALSE;
				STR_CLEAN(ptr);
				STRDUP_REPORT(ptr,neural->f_kernel,allocate);
				if(neural->f_kernel==NULL){
					fprintf(stderr,"Malformed NN configuration file!\n");
					fprintf(stderr,"keyword: init, can't read filename: %s\n",ptr);
					goto FAIL;
				}
			}
		}
		ptr=STRFIND("[seed",line);
		if(ptr!=NULL){
			ptr+=6;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				fprintf(stderr,"Malformed NN configuration file!\n");
				fprintf(stderr,"keyword: seed, value: %s\n",ptr);
				goto FAIL;
			}
			GET_UINT(neural->seed,ptr,ptr2);
		}
		ptr=STRFIND("[input",line);
		if(ptr!=NULL){
			/*get number of inputs {integer}*/
			ptr+=7;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				fprintf(stderr,"Malformed NN configuration file!\n");
				fprintf(stderr,"keyword: input, value: %s\n",ptr);
				goto FAIL;
			}
			GET_UINT(parameter[0],ptr,ptr2);
			/*input can be 0 or even missing (it then will be set while loading kernel)*/
		}
		ptr=STRFIND("[hidden",line);
		if(ptr!=NULL){
			/*get number of hidden neurons {integer x n_layers}*/
			ptr+=8;SKIP_BLANK(ptr);
			/*count the number of integers -> n_hiddens*/
			if(!ISDIGIT(*ptr)) {
				fprintf(stderr,"Malformed NN configuration file!\n");
				fprintf(stderr,"keyword: hidden, value: %s\n",ptr);
				goto FAIL;
				return FALSE;
			}
			parameter[1]=1;ptr2=ptr;
			while(ISGRAPH(*ptr)){
				while(ISDIGIT(*ptr)) ptr++;
				SKIP_BLANK(ptr);
				if(ISDIGIT(*ptr)) parameter[1]++;
				else break;
			}
			/*now get each hidden layer n_neurons*/
			ALLOC_REPORT(n_hiddens,parameter[1],UINT,allocate);
			ptr=ptr2;
			for(idx=0;idx<parameter[1];idx++){
				GET_DOUBLE(n_hiddens[idx],ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
				ptr=ptr2+1;SKIP_BLANK(ptr);
			}
		}
		ptr=STRFIND("[output",line);
		if(ptr!=NULL){
			/*get the number of output {integer}*/
			ptr+=8;SKIP_BLANK(ptr);
			if(!ISDIGIT(*ptr)) {
				fprintf(stderr,"Malformed NN configuration file!\n");
				fprintf(stderr,"keyword: output, value: %s\n",ptr);
				goto FAIL;
			}
			GET_UINT(parameter[2],ptr,ptr2);
		}
		ptr=STRFIND("[train",line);
		if(ptr!=NULL){
			/*get the training type {"BP","BPM","CG" ...}*/
			ptr+=7;SKIP_BLANK(ptr);
			switch (*ptr){
				case 'B':
					if(*(ptr+2)=='M') neural->train=NN_TRAIN_BPM;
					else neural->train=NN_TRAIN_BP;
					break;
				case 'C':
					neural->train=NN_TRAIN_CG;
					break;
				default:
					neural->train=NN_TRAIN_UKN;
			}
		}
		ptr=STRFIND("[sample_dir",line);
		if(ptr!=NULL){
			/*get the sample directory {"dir"}*/
			ptr+=12;SKIP_BLANK(ptr);
			STR_CLEAN(ptr);
			STRDUP_REPORT(ptr,neural->samples,allocate);
		}
		ptr=STRFIND("[test_dir",line);
		if(ptr!=NULL){
			/*get the test directory {"dir"}*/
			ptr+=10;SKIP_BLANK(ptr);
			STR_CLEAN(ptr);
			STRDUP_REPORT(ptr,neural->tests,allocate);
		}
		READLINE(fp,line);
	}while(!feof(fp));
	/*checks*/
	if(neural->type==NN_TYPE_UKN){
		fprintf(stderr,"Malformed NN configuration file!\n");
		fprintf(stderr,"keyword: type; unknown or missing...\n");
		goto FAIL;
	}
	if(neural->need_init==TRUE){
		if(parameter[0]==0){
			fprintf(stderr,"Malformed NN configuration file!\n");
			fprintf(stderr,"keyword: input; wrong or missing...\n");
			goto FAIL;
		}
		if(parameter[1]==0){
			fprintf(stderr,"Malformed NN configuration file!\n");
			fprintf(stderr,"keyword: hidden; wrong or missing...\n");
			goto FAIL;
		}
		if(parameter[2]==0){
			fprintf(stderr,"Malformed NN configuration file!\n");
			fprintf(stderr,"keyword: output; wrong or missing...\n");
			goto FAIL;
		}
		is_ok=_NN(kernel,generate)(neural,parameter[0],parameter[1],parameter[2],n_hiddens);
		if(!is_ok){
			fprintf(stderr,"FAILED to generate NN kernel!\n");
			fprintf(stderr,"keyword: type; unsupported...\n");
			goto FAIL;
		}
	}else{
		is_ok=_NN(kernel,load)(neural);
		if(!is_ok){
			fprintf(stderr,"FAILED to load the NN kernel!\n");
			goto FAIL;
		}
	}
	if(neural->kernel==NULL){
		fprintf(stderr,"Initialization or load of NN kernel FAILED!\n");
		goto FAIL;
	}
	FREE(parameter);
	FREE(n_hiddens);
	fclose(fp);
fprintf(stdout,"NN definition allocation: %lu (bytes)\n",allocate);
	return neural;
read_conf_fail:
	FREE(neural->name);neural->name=NULL;
	FREE(neural->samples);neural->samples=NULL;
	FREE(neural->tests);neural->tests=NULL;
	FREE(neural);
	FREE(parameter);
	FREE(n_hiddens);
	return NULL;
#undef FAIL
}

void dump_conf(FILE *fp,nn_def *neural){
	UINT n_hiddens;
	UINT idx;
	if(fp==NULL) return;
	fprintf(fp,"# NN configuration\n");
	fprintf(fp,"[name] %s\n",neural->name);
	switch(neural->type){
		case NN_TYPE_LNN:
			fprintf(fp,"[type] LNN\n");
			break;
		case NN_TYPE_PNN:
			fprintf(fp,"[type] PNN\n");
			break;
		case NN_TYPE_ANN:
		default:
			fprintf(fp,"[type] ANN\n");
	}
	if(neural->need_init) fprintf(fp,"[init] generate\n");
	else fprintf(fp,"[init] %s\n",neural->f_kernel);
	fprintf(fp,"[seed] %i\n",neural->seed);
	fprintf(fp,"[inputs] %i\n",_NN(get,n_inputs)(neural));
	n_hiddens=_NN(get,n_hiddens)(neural);
	fprintf(fp,"[hiddens] ");
	for(idx=0;idx<n_hiddens;idx++){
		fprintf(fp,"%i ",_NN(get,h_neurons)(neural,idx));
	}
	fprintf(fp,"\n");
	fprintf(fp,"[outputs] %i\n",_NN(get,n_outputs)(neural));
	switch(neural->train){
		case NN_TRAIN_BP:
			fprintf(fp,"[train] BP\n");
			break;
		case NN_TRAIN_BPM:
			fprintf(fp,"[train] BPM\n");
			break;
		case NN_TRAIN_CG:
			fprintf(fp,"[train] CG\n");
			break;
		default:
			fprintf(fp,"[train] none\n");
	}

	if(neural->samples!=NULL) fprintf(fp,"[sample_dir] %s\n",neural->samples);
	if(neural->tests!=NULL) fprintf(fp,"[test_dir] %s\n",neural->tests);
}



