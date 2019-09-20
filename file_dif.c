#include <stdio.h>
#include <stdlib.h>

#include "common.h"

#include "file_dif.h"

#define DIF (*dif)

/*read a RRUFF database DIF file*/
_dif *read_dif(CHAR *filename){
#define FAIL read_dif_fail
	PREP_READLINE();
	CHAR *line=NULL;
	CHAR *ptr,*ptr2;
	DOUBLE *pt, *pi;
	int is_ok;
	_atm  *at;
	FILE  *fp;
	UINT  idx;
	UINT  jdx;
	_dif *dif;
	/*init arrays*/
	pt=NULL;
	pi=NULL;
	at=NULL;
	line=NULL;
	ALLOC(dif,1,_dif);
	/*try to get the most of diff file*/
	fp=fopen(filename,"r+");/*+ is here to assert opening directory*/
	if(!fp) {
		fprintf(stderr,"Error opening file: %s\n",filename);
		return NULL;
	}
	READLINE(fp,line);/*line 1: name*/
/*there are 4 files: R060187, R060699, R060349, and R060508 that do not contain full set information*/
	if(STRFIND("R060187",line) != NULL) goto FAIL;
	if(STRFIND("5.000",line) != NULL) goto FAIL;
/*there are also 823 files that do not contain ATOM information -- supposedly available in literature.*/
	ptr=&(line[0]);SKIP_BLANK(ptr);
	idx=0;ptr2=ptr;
	while(ISGRAPH(*ptr)) {
		ptr++;
		idx++;
	}
	if(idx==0){
		/*noname structure*/
		ALLOC(DIF.name,4,CHAR);
		DIF.name[0]='?';
		DIF.name[1]='?';
		DIF.name[2]='?';
		DIF.name[3]='\0';/*unnecessary*/
	}else{
		ALLOC(DIF.name,idx+1,CHAR);
		for(jdx=0;jdx<idx;jdx++) DIF.name[jdx]=ptr2[jdx];
		DIF.name[idx]='\0';/*unnecessary*/
	}
	DIF.temp=273.15+25.0;	/*AKA room temperature!*/
	DIF.space=0;		/*AKA unknown <- not P1*/
	DIF.natoms=0;		/*if stay at zero dif file is invalid!*/
	DIF.atoms=NULL;		/*unnecessary*/
	DIF.lambda=1.541838;	/*all dif files have lambda = 1.541838*/
	DIF.n_peaks=0;		/*if stay at zero dif file is invalid!*/
	DIF.pk_t=NULL;		/*unnecessary*/
	DIF.pk_i=NULL;		/*unnecessary*/
	do{
		READLINE(fp,line);
		if(STRFIND("Sample",line)!=NULL){
			/*try to get temperature*/
			ptr=NULL;
			ptr=STRFIND("T =",line);
			if(ptr!=NULL){
				ptr+=4;
				GET_DOUBLE(DIF.temp,ptr,ptr2);
				/*deg C or K <- AFAIK there is no other unit.*/
				if(ptr2==NULL){
					/*no unit? <- never seen that*/
					/*anyway, assume Celsius*/
					DIF.temp+=273.15;
				}else{
					ptr2++;
					if(*ptr2!='K'){
						DIF.temp+=273.15;
					}
				}
			}
		}
		ptr=STRFIND("CELL PARAMETERS:",line);
		if(ptr != NULL){
			/*get crystal lattice parameters <- mandatory!*/
			ptr+=16;
			SKIP_BLANK(ptr);
			GET_DOUBLE(DIF.cl.a,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
			ptr=ptr2+1;SKIP_BLANK(ptr);
			GET_DOUBLE(DIF.cl.b,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
			ptr=ptr2+1;SKIP_BLANK(ptr);
			GET_DOUBLE(DIF.cl.c,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
			ptr=ptr2+1;SKIP_BLANK(ptr);
			GET_DOUBLE(DIF.cl.alpha,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
			ptr=ptr2+1;SKIP_BLANK(ptr);
			GET_DOUBLE(DIF.cl.beta,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
			ptr=ptr2+1;SKIP_BLANK(ptr);
			GET_DOUBLE(DIF.cl.gamma,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
		}
		ptr=STRFIND("SPACE GROUP",line);
		if(ptr!=NULL){
			/*one dif file, R060879, has an incorrect "SPACE GROUP #:" instead of just "SPACE GROUP:"*/
			ptr+=11;
			if(*ptr!=':') ptr++;
			ptr+=2;
			/*the space group is given as a string, we need to convert it to an integer...*/
/*because this is WAY too long*/
#define SG_IS_EQ(pointer,index,is_ok) do{\
	SHORT _i=0;\
	do{\
		if(*(pointer+_i)==space_groups_hm[index][_i]) is_ok=TRUE;\
		else {\
			is_ok=FALSE;\
			break;\
		}\
		_i++;\
	}while(ISGRAPH(*(pointer+_i)));\
	if(ISGRAPH(space_groups_hm[index][_i])) is_ok=FALSE;\
}while(0)
			idx=NUM_GROUPS;
			while(idx>0){
				idx--;
				SG_IS_EQ(ptr,idx,is_ok);
				if(is_ok){
					DIF.space=space_groups_id[idx];
					break;
				}
			}
if(DIF.space==0) fprintf(stdout,"#DBG: NO_space group = %s\n",ptr);
		}
		ptr=STRFIND("ATOM",line);
		if(ptr!=NULL){
			/*read in atom positions*/
			READLINE(fp,line);
			ptr=&(line[0]);SKIP_BLANK(ptr);
while((!ISDIGIT(*ptr))&&(ISGRAPH(*ptr))){
			/*start from last atom for a simplier match*/
/*because this is WAY too long*/
#define ATM_IS_EQ(pointer,idx,is_ok) do{\
	if(*pointer!=atom_symb[idx][0]) is_ok=FALSE;\
	else if((ISSPACE(*(pointer+1)))&&(atom_symb[idx][1]=='\0')) is_ok=TRUE;\
	else if(*(pointer+1)==atom_symb[idx][1]) is_ok=TRUE;\
	else is_ok=FALSE;\
}while(0)
			for(idx=MAX_ATOMS;idx>0;idx--){
				ATM_IS_EQ(ptr,idx,is_ok);
				if(is_ok){
					/*we have a hit, given:*/
					/*I before In*/
					if((idx==53)&&(*(ptr+1)=='n')) idx=49;
					/*S before Si*/
					if((idx==16)&&(*(ptr+1)=='i')) idx=14;
					/*B before Be*/
					if((idx==5)&&(*(ptr+1)=='e')) idx=4;
					DIF.natoms++;
					ALLOC(at,DIF.natoms,_atm);
					ATM_CP(DIF.atoms,at,DIF.natoms-1);
					at[DIF.natoms-1].Z=idx;
					ptr+=2;
					GET_DOUBLE(at[DIF.natoms-1].x,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].y,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].z,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].occ,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].B,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					FREE(DIF.atoms);
					DIF.atoms=at;at=NULL;
					break;
				}
			}
			if(idx<0){
				/*we missed it, but there are special "atomic" types:*/
				/*OH -> O from OH*/
				/*Wa -> O from OH2*/
				/*Ow -> O from OH2*/
				/*Oh -> O from OH*/
				if(((*ptr=='O')&&(*(ptr+1)=='H'))
					||((*ptr=='W')&&(*(ptr+1)=='a'))
					||((*ptr=='O')&&(*(ptr+1)=='w'))
					||((*ptr=='O')&&(*(ptr+1)=='h'))){
					DIF.natoms++;
					ALLOC(at,DIF.natoms,_atm);
					ATM_CP(DIF.atoms,at,DIF.natoms-1);
					at[DIF.natoms-1].Z=8;/*O*/
					ptr+=2;
					GET_DOUBLE(at[DIF.natoms-1].x,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].y,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].z,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].occ,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].B,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					FREE(DIF.atoms);
					DIF.atoms=at;at=NULL;
				}else{
					/*we have a unknown response... X?*/
					DIF.natoms++;
					ALLOC(at,DIF.natoms,_atm);
					ATM_CP(DIF.atoms,at,DIF.natoms-1);
					at[DIF.natoms-1].Z=0;/*X->unknown*/
					ptr+=2;
					GET_DOUBLE(at[DIF.natoms-1].x,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].y,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].z,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].occ,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					ptr=ptr2+1;SKIP_BLANK(ptr);
					GET_DOUBLE(at[DIF.natoms-1].B,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
					FREE(DIF.atoms);
					DIF.atoms=at;at=NULL;
				}
			}
			/*go to next line/atom*/
			READLINE(fp,line);
			ptr=&(line[0]);SKIP_BLANK(ptr);
}
		}
		ptr=STRFIND("WAVELENGTH",line);
		if(ptr!=NULL){
			/*should be 1.541838 anyway*/
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			if(!ISDIGIT(*ptr)) continue;/*let's be permissive*/
			GET_DOUBLE(DIF.lambda,ptr,ptr2);
/* lambda is now a parameter
			if(DIF.lambda!=1.541838){
				fprintf(stdout,"#WARNING: expected 1.541838, but got %lf\n",DIF.lambda);
			}
*/
		}
		ptr=STRFIND("2-THETA",line);
		if(ptr!=NULL){
			/*read peak data*/
			READLINE(fp,line);
			ptr=&(line[0]);SKIP_BLANK(ptr);
			while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
			while(ISDIGIT(*ptr)){
				DIF.n_peaks++;
				ALLOC(pt,DIF.n_peaks,DOUBLE);
				ARRAY_CP(DIF.pk_t,pt,DIF.n_peaks-1);
				ALLOC(pi,DIF.n_peaks,DOUBLE);
				ARRAY_CP(DIF.pk_i,pi,DIF.n_peaks-1);
				GET_DOUBLE(pt[DIF.n_peaks-1],ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
				ptr=ptr2+1;SKIP_BLANK(ptr);
				GET_DOUBLE(pi[DIF.n_peaks-1],ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
				FREE(DIF.pk_t);
				FREE(DIF.pk_i);
				DIF.pk_t=pt;pt=NULL;
				DIF.pk_i=pi;pi=NULL;
				READLINE(fp,line);
				ptr=&(line[0]);SKIP_BLANK(ptr);
			}
		}
	}while(!feof(fp));
	/*check requirements*/
//	if(DIF.natoms==0) goto FAIL;//most LACK this infomation anyway
	if(DIF.n_peaks==0) goto FAIL;
	FREE(pt);
	FREE(pi);
	FREE(at);
	FREE(line);
	fclose(fp);
	return dif;
read_dif_fail:
	/*something has gone wrong, safely bail*/
	FREE(pt);
	FREE(pi);
	FREE(at);
	FREE(line);
	FREE(DIF.name);
	FREE(DIF.atoms);
	FREE(DIF.pk_t);
	FREE(DIF.pk_i);
	return NULL;
#undef FAIL
}
BOOL read_raw(CHAR *filename,_dif *dif){
/*read the raw part of XRD*/
#define FAIL read_raw_fail
	PREP_READLINE();
	CHAR *line=NULL;
	CHAR *ptr,*ptr2;
	DOUBLE p_t, p_i;
	DOUBLE  *tmp_pt;
	UINT  nb_points;
	DOUBLE acc;
	FILE  *fp;
	/*init arrays*/
	line=NULL;
	/*try to get the most of diff file*/
	fp=fopen(filename,"r+");/*+ is here to assert opening directory*/
	if(!fp) {
		fprintf(stderr,"Error opening file: %s\n",filename);
		return FALSE;
	}
	/*skip # lines*/
	DIF.raw._t=NULL;
	DIF.raw._i=NULL;
	do{
		READLINE(fp,line);
	}while((!ISDIGIT(line[0]))&&(!feof(fp)));
	if(feof(fp)) return FALSE;
	nb_points=0;acc=0.;
	do{
		ptr=&(line[0]);SKIP_BLANK(ptr);
		GET_DOUBLE(p_t,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
		ptr=ptr2+1;while(!(ISDIGIT(*ptr))&&(*ptr!='\n')&&(*ptr!='\0')) ptr++;
		GET_DOUBLE(p_i,ptr,ptr2);ASSERT_GOTO(ptr2,FAIL);
		/*we have pi,pt*/
		nb_points++;
		ALLOC(tmp_pt,nb_points,DOUBLE);
		ARRAY_CP(DIF.raw._t,tmp_pt,nb_points-1);
		tmp_pt[nb_points-1]=p_t;
		FREE(DIF.raw._t);
		DIF.raw._t=tmp_pt;
		tmp_pt=NULL;
		ALLOC(tmp_pt,nb_points,DOUBLE);
		ARRAY_CP(DIF.raw._i,tmp_pt,nb_points-1);
		tmp_pt[nb_points-1]=p_i;
		FREE(DIF.raw._i);
		DIF.raw._i=tmp_pt;
		tmp_pt=NULL;
		acc+=p_i;
read_raw_fail:
	/*we are permissive on read failure of some data*/
		READLINE(fp,line);
	}while(!feof(fp));
	DIF.raw.n_sample=nb_points;
	return TRUE;
#undef FAIL
}
/*print a _dif structure on screen*/
void dump_dif(const _dif *dif){
UINT idx;
fprintf(stdout,"      %s\n",DIF.name);
fprintf(stdout,"*** This is a sample for training ANN ***\n");
fprintf(stdout,"      Sample: T = %lf K\n",DIF.temp);
fprintf(stdout,"\n");
fprintf(stdout,"      CELL PARAMETERS: %lf  %lf  %lf   %lf  %lf  %lf\n",
	DIF.cl.a,DIF.cl.b,DIF.cl.c,DIF.cl.alpha,DIF.cl.beta,DIF.cl.gamma);
/*find _a_ space group*/
idx=0;
do{
	idx++;
}while((space_groups_id[(UINT) DIF.space]!=DIF.space)&&(idx<NUM_GROUPS));
if(idx<NUM_GROUPS) fprintf(stdout,"      SPACE GROUP: %s (%i)\n",space_groups_hm[idx],(UINT) DIF.space);
else fprintf(stdout,"      SPACE GROUP: %i\n",(UINT) DIF.space);

fprintf(stdout,"\n");
fprintf(stdout,"           ATOM        X         Y         Z     OCCUPANCY  ISO(B)\n");
for(idx=0;idx<DIF.natoms;idx++){
	fprintf(stdout,"            %s",atom_symb[DIF.atoms[idx].Z]);
	if(atom_symb[DIF.atoms[idx].Z][1]=='\0') fprintf(stdout," ");
	fprintf(stdout,"     %.5f   %.5f   %.5f",DIF.atoms[idx].x,DIF.atoms[idx].y,DIF.atoms[idx].z);
	fprintf(stdout,"     %.3f     %.3f\n",DIF.atoms[idx].occ,DIF.atoms[idx].B);
}
fprintf(stdout,"\n");
fprintf(stdout,"            X-RAY WAVELENGTH:     %lf\n",DIF.lambda);
fprintf(stdout,"\n");
fprintf(stdout,"               2-THETA      INTENSITY\n");
for(idx=0;idx<DIF.n_peaks;idx++){
	fprintf(stdout,"                %.2f        %6.2f\n",DIF.pk_t[idx],DIF.pk_i[idx]);
}


fprintf(stdout,"*** END of sample (training ANN) ***\n");
}

BOOL dif_2_sample(const _dif *dif,FILE *dest,UINT n_inputs,UINT n_outputs){
	/*write a sample file from a dif (+raw)*/
	/*to be used as a training input in NN.*/
	UINT   idx, jdx;
	DOUBLE acc, max;
	DOUBLE max_i=0.;
	DOUBLE interval;
	DOUBLE *samples;
	/*------------*/
	if((dif==NULL)||(dest==NULL)||(n_inputs==0)||(n_outputs==0)) return FALSE;
	ALLOC(samples,n_inputs,DOUBLE);

	/*start writting*/
	fprintf(dest,"[input] %i\n",n_inputs);
	/*temperature input <- have to be relative ie. T/T0*/
	/*integrate input*/
	interval=(MAX_THETA-MIN_THETA)/(n_inputs-1);
	max=MIN_THETA+interval;
	jdx=0;
	/*ignore values below MIN_THETA*/
	while((DIF.raw._t[jdx]<MIN_THETA)&&(jdx<DIF.raw.n_sample)) jdx++;
	/*starts sampling at 1 (temperature is idx=0)*/
	for(idx=1;idx<n_inputs;idx++){
		acc=0.;
		while((DIF.raw._t[jdx]<max)&&(jdx<DIF.raw.n_sample)) {
			acc+=DIF.raw._i[jdx];
			jdx++;
		}
		max+=interval;
		samples[idx]=acc;
		if(acc>max_i) max_i=acc;
	}
	if(max_i==0.) {
		FREE(samples);
		return FALSE;/*this is NOT OK*/
	}
	/*now print, temperature have to be relative ie. T/T0*/
	fprintf(dest,"%7.5f",DIF.temp/273.15);
	for(idx=1;idx<n_inputs;idx++) {
		samples[idx]/=max_i;
		fprintf(dest," %7.5f",samples[idx]);
	}
	fprintf(dest,"\n");
	/*done, now process output (here space group)*/
	fprintf(dest,"[output] %i\n",n_outputs);
	if(DIF.space==1) fprintf(dest,"1.0");
	else fprintf(dest,"-1.0");
	for(idx=1;idx<n_outputs;idx++){
		if(idx==DIF.space-1) fprintf(dest," 1.0");
		else fprintf(dest," -1.0");
	}
	fprintf(dest,"\n");
	/*all done!*/
	FREE(samples);
	return TRUE;
}





#undef DIF




