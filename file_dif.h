#ifndef FILE_DIF_H
#define FILE_DIF_H

#ifndef COMMON_H
#error "require include common.h BEFORE file_dif.h"
#endif /*COMMON_H*/

#define MIN_THETA 5.
#define MAX_THETA 90.

typedef struct {
	DOUBLE a,b,c;
	DOUBLE alpha,beta,gamma;
} _cr_latt;
typedef struct {
	UCHAR Z;                /*atomic number*/
	DOUBLE x,y,z;           /*atom position*/
	DOUBLE occ;             /*atom occupancy*/
	DOUBLE B;               /*temperature factor (ass. isotropy)*/
} _atm;
#define ATM_CP(src,dest,n) do{\
	if((src)!=NULL){\
		UINT _i;\
		for(_i=0;_i<(n);_i++){\
			dest[_i].Z=src[_i].Z;\
			dest[_i].x=src[_i].x;\
			dest[_i].y=src[_i].y;\
			dest[_i].z=src[_i].z;\
			dest[_i].occ=src[_i].occ;\
			dest[_i].B=src[_i].B;\
		}\
	}\
}while(0)
typedef struct {
	UINT n_sample;
	DOUBLE *_t;		/*2theta*/
	DOUBLE *_i;		/*intensity*/
} _dif_raw;
typedef struct {
	/*general definition*/
	CHAR  *name;            /*name of the structure, unused*/
	/*Sample definition*/
	DOUBLE temp;            /*temperature*/
	/*crystal structure*/
	_cr_latt cl;            /*crystal lattice*/
	UCHAR space;            /*space group: [0,255] 0-> unkown >230 -> undefined*/
	UINT natoms;            /*number of atoms*/
	_atm *atoms;            /*each atom definition*/
	/*diffraction*/
	DOUBLE lambda;
	/*XRD peaks*/
	UINT n_peaks;           /*peak number*/
	DOUBLE *pk_t;           /*2theta (peak)*/
	DOUBLE *pk_i;           /*intensity (peak)*/
	_dif_raw raw;		/*raw spectra*/
} _dif;
/*functions*/
_dif *read_dif(CHAR *filename);
BOOL read_raw(CHAR *filename,_dif *dif);
void dump_dif(const _dif *dif);
BOOL dif_2_sample(const _dif *dif,FILE *dest,UINT n_inputs,UINT n_outputs);



#endif /*FILE_DIF_H*/
