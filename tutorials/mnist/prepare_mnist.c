#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <libhpnn/common.h>

/*This translate the compact MNIST file into individual
 * input output format for use with train_nn and run_nn
 * example programs.                            --OVHPA */
/*+++ MNIST STRUCTURE +++*/
typedef struct {
	uint32_t magic;
	uint32_t  size;
	unsigned char label;
	uint32_t pixel_rows;
	uint32_t pixel_cols;
	unsigned char *pixels;
} mnist_data;
#define N_PX (data.pixel_rows*data.pixel_cols)
#define PX_MAX 255
/*------------*/
/*+++ HELP +++*/
/*------------*/
void dump_help(){
        fprintf(stdout,"********************************************\n");
        fprintf(stdout,"usage: pmnist samples_dir tests_dir         \n");
        fprintf(stdout,"********************************************\n");
        fprintf(stdout,"samples_dir: where the training samples will\n");
	fprintf(stdout,"be written.\n");
	fprintf(stdout,"tests_dir: where the testing samples will be\n");
	fprintf(stdout,"written.\n");
        fprintf(stdout,"********************************************\n");
        fprintf(stdout,"The default MNIST files should be renamed to\n");
        fprintf(stdout,"train_images from    train-images-idx3-ubyte\n");
        fprintf(stdout,"train_labels from    train-labels-idx1-ubyte\n");
	fprintf(stdout,"test_images  from     t10k-images-idx3-ubyte\n");
	fprintf(stdout,"test_labels  from     t10k-labels-idx1-ubyte\n");
        fprintf(stdout,"********************************************\n");
        fprintf(stdout,"This code is released 'as is', within GPLv3.\n");
        fprintf(stdout,"available at:  https://github.com/ovhpa/hpnn\n");
        fprintf(stdout,"- project started 2019~            -- OVHPA.\n");
        fprintf(stdout,"********************************************\n");
}
/*--------------------*/
/*+++ WRITE output +++*/
/*--------------------*/
void write_output(FILE *sample_f,mnist_data data){
	int idx;
	fprintf(sample_f,"[input] %i\n",(data.pixel_rows*data.pixel_cols));
	fprintf(sample_f,"%7.5f",(double) data.pixels[0]);
	for(idx=1;idx<(N_PX);idx++) fprintf(sample_f," %7.5f",(double) data.pixels[idx]);
	fprintf(sample_f,"\n");
	fprintf(sample_f,"[output] %i  #%d\n",10,data.label);
	if(data.label == 0) fprintf(sample_f,"1.0");
	else fprintf(sample_f,"-1.0");
	for(idx=1;idx<10;idx++) 
		if(data.label == idx) fprintf(sample_f," 1.0");
		else fprintf(sample_f," -1.0");
	fprintf(sample_f,"\n");
}
/*-------------------*/
/*+++ MAIN pmnist +++*/
/*-------------------*/
int main (int argc, char *argv[]){
	char  *label_nm;
	char  *image_nm;
	char *sample_nm;
	char *sample_wd;
	char   *test_wd;
	FILE   *label_f;
	FILE   *image_f;
	FILE  *sample_f;
	mnist_data data;
	int   index = 0;
	char s_name[13];
	uint32_t magic2;
	uint32_t  size2;
	/**/
/*>>> inputs*/
	if(argv[1][0]=='-'){
		if(argv[1][1]=='h'){
			dump_help();
			return 0;
		}
		if((argv[1][1]=='-')&&(argv[1][2]=='h')){
			dump_help();
			return 0;
		}
		_OUT(stderr,"ERROR invalid argument!\n");
	}
	if(argc<2) {/*2 args minimum*/
		_OUT(stderr,"ERROR not enough arguments!\n");
		dump_help();
		return 1;
	}
	STRDUP(argv[1],sample_wd);
	STRDUP(argv[2],test_wd);
	_OUT(stdout,"processing sample database into %s directory.\n",sample_wd);
	_OUT(stdout,"processing   test database into %s directory.\n",test_wd);
/*>>> initialize!*/
	STRDUP("./train_labels",label_nm);
	STRDUP("./train_images",image_nm);
	/*init data*/
	data.magic = 0;
	data.size = 0;
	data.label = 0;
	data.pixel_rows = 0;
	data.pixel_cols = 0;
	data.pixels = NULL;
/*>>> open files!*/
	label_f = fopen(label_nm,"rb");
	if(label_f == NULL){
		_OUT(stderr,"FAILED to open label file %s for READ!\n",label_nm);
		return -1;
	}
	image_f = fopen(image_nm,"rb");
	if(image_f == NULL){
		_OUT(stderr,"FAILED to open image file %s for READ!\n",image_nm);
		return -1;
	}
#define _READ(file,var) do{\
	int is_ok = fread(&(var),sizeof(var),1,file);\
	if(is_ok != 1) {\
		_OUT(stderr,"READ FAIL: var %s\n",QUOTE(var));\
		return -1;\
	}\
}while(0)
/*ignore error*/
#define _READ_i(file,var) fread(&(var),sizeof(var),1,file)
#define _READ_N(file,array,type,nb) do{\
	int is_ok = fread(array,sizeof(type),nb,file);\
	if(is_ok != nb) {\
		_OUT(stderr,"READ FAIL: var %s read %d of %d requested\n",QUOTE(var),is_ok,nb);\
		return -1;\
	}\
}while(0)
#define _SWP_32(num) ((num>>24)&0xff)|((num<<8)&0xff0000)|((num>>8)&0xff00)|((num<<24)&0xff000000)
	/*label magic and size*/
	_READ(label_f,data.magic);data.magic = _SWP_32(data.magic);
	_READ(label_f,data.size);data.size = _SWP_32(data.size);
	/*image magic and size*/
	_READ(image_f,magic2);magic2 = _SWP_32(magic2);
	_READ(image_f,size2);size2 = _SWP_32(size2);
	if(data.size != size2){
		_OUT(stderr,"ERROR: different set size!\n-- %s has %d and %s has %d",
			label_nm,data.size,image_nm,size2);
		return -1;
	}
	_OUT(stdout,"# Opened samples label=%X image=%X\n",data.magic,magic2);
	_READ(image_f,data.pixel_rows);data.pixel_rows = _SWP_32(data.pixel_rows);
	_READ(image_f,data.pixel_cols);data.pixel_cols = _SWP_32(data.pixel_cols);
	size2 = data.pixel_rows*data.pixel_cols;
	if(size2 == 0) {
		_OUT(stderr,"ERROR: pixel size is 0: rows=%d cols=%d!\n",
		data.pixel_rows,data.pixel_cols);
		return -1;
	}
/*>>> allocate pixels*/
	ALLOC(data.pixels,size2,unsigned char);
/*>>> process data*/
	_READ(label_f,data.label);/*first label*/
	while((!feof(label_f))&&(!feof(image_f))){
		index++;
		if(data.label>9){
			_OUT(stderr,"ERROR: label out of boundaries!\n");
			continue;/*<- one data corrupt?*/
		}
		/*image data*/
		_READ_N(image_f,data.pixels,unsigned char,size2);
		/*Prepare sample name*/
		sprintf(s_name,"/s%05d.txt",index);s_name[12]='\0';
		STRCAT(sample_nm,sample_wd,s_name);
		/*open output*/
		sample_f = fopen(sample_nm,"w");
		if(sample_f == NULL){
			_OUT(stderr,"FAILED to open sample %s for WRITE!\n",sample_nm);
			return -1;
		}
		/*put sample in output*/
		write_output(sample_f,data);
		/*close output*/
		fclose(sample_f);
		FREE(sample_nm);
		/*read next label (if any)*/
		_READ_i(label_f,data.label);
	}
/*>>> close files*/
	fclose(image_f);
	fclose(label_f);
/*>>> samples done, now tests*/
        STRDUP("./test_labels",label_nm);
        STRDUP("./test_images",image_nm);
/*>>> open files!*/
        label_f = fopen(label_nm,"rb");
        if(label_f == NULL){
                _OUT(stderr,"FAILED to open label file %s for READ!\n",label_nm);
                return -1;
        }
        image_f = fopen(image_nm,"rb");
        if(image_f == NULL){
                _OUT(stderr,"FAILED to open image file %s for READ!\n",image_nm);
                return -1;
        }
        /*label magic and size*/
        _READ(label_f,data.magic);data.magic = _SWP_32(data.magic);
        _READ(label_f,data.size);data.size = _SWP_32(data.size);
        /*image magic and size*/
        _READ(image_f,magic2);magic2 = _SWP_32(magic2);
        _READ(image_f,size2);size2 = _SWP_32(size2);
        if(data.size != size2){
                _OUT(stderr,"ERROR: different set size!\n-- %s has %d and %s has %d",
                        label_nm,data.size,image_nm,size2);
                return -1;
        }
        _OUT(stdout,"# Opened tests label=%X image=%X\n",data.magic,magic2);
        _READ(image_f,data.pixel_rows);data.pixel_rows = _SWP_32(data.pixel_rows);
        _READ(image_f,data.pixel_cols);data.pixel_cols = _SWP_32(data.pixel_cols);
        size2 = data.pixel_rows*data.pixel_cols;
        if(size2 == 0) {
                _OUT(stderr,"ERROR: pixel size is 0: rows=%d cols=%d!\n",
                data.pixel_rows,data.pixel_cols);
                return -1;
        }
/*>>> re-allocate pixels*/
	FREE(data.pixels);
        ALLOC(data.pixels,size2,unsigned char);
/*>>> process data*/
        _READ(label_f,data.label);/*first label*/
/*>>> process data*/
        _READ(label_f,data.label);/*first label*/
        while((!feof(label_f))&&(!feof(image_f))){
                index++;
                if(data.label>9){
                        _OUT(stderr,"ERROR: label out of boundaries!\n");
                        continue;/*<- one data corrupt?*/
                }
                /*image data*/
                _READ_N(image_f,data.pixels,unsigned char,size2);
                /*Prepare sample name*/
                sprintf(s_name,"/s%05d.txt",index);s_name[12]='\0';
                STRCAT(sample_nm,test_wd,s_name);
                /*open output*/
                sample_f = fopen(sample_nm,"w");
                if(sample_f == NULL){
                        _OUT(stderr,"FAILED to open sample %s for WRITE!\n",sample_nm);
                        return -1;
                }
                /*put sample in output*/
                write_output(sample_f,data);
                /*close output*/
                fclose(sample_f);
                FREE(sample_nm);
                /*read next label (if any)*/
                _READ_i(label_f,data.label);
        }
/*>>> close files*/
        fclose(image_f);
        fclose(label_f);
/*>>> de-init all*/
	FREE(data.pixels);
	FREE(label_nm);
	FREE(image_nm);
	FREE(sample_wd);
	FREE(test_wd);
	return 0;
}



