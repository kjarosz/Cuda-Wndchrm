/***************************************************************************
* ======================================================================
* Computer Vision/Image Processing Tool Project - Dr. Scott Umbaugh SIUE
* ======================================================================
*
*             File Name: CVIPtexture.h
*           Description: contains function prototypes, type names, constants,
*			 etc. related to libdataserv (Data Services Toolkit.)
*         Related Files: Imakefile, cvip_pgmtexture.c
*   Initial Coding Date: 6/19/96
*           Portability: Standard (ANSI) C
*             Credit(s): Steve Costello
*                        Southern Illinois University @ Edwardsville
*
** Copyright (C) 1993 SIUE - by Gregory Hance.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
****************************************************************************/
#ifndef _CVIP_texture
#define _CVIP_texture

#include "device_launch_parameters.h"



#define RADIX 2.0
#define EPSILON 0.000000001

#define SIGN(x,y) ((y)<0 ? -fabs(x) : fabs(x))
#define SWAP(a,b) {y=(a);(a)=(b);(b)=y;}
#define PGM_MAXMAXVAL 255



typedef unsigned char u_int8_t;

typedef struct  {
	float ASM;           /*  (1) Angular Second Moment */
	float contrast;      /*  (2) Contrast */
	float correlation;   /*  (3) Correlation */
	float variance;      /*  (4) Variance */
	float IDM;		       /*  (5) Inverse Diffenence Moment */
	float sum_avg;	     /*  (6) Sum Average */
	float sum_var;	     /*  (7) Sum Variance */
	float sum_entropy;	 /*  (8) Sum Entropy */
	float entropy;	     /*  (9) Entropy */
	float diff_var;	     /* (10) Difference Variance */
	float diff_entropy;	 /* (11) Diffenence Entropy */
	float meas_corr1;	   /* (12) Measure of Correlation 1 */
	float meas_corr2;	   /* (13) Measure of Correlation 2 */
	float max_corr_coef; /* (14) Maximal Correlation Coefficient */
} TEXTURE;



/* 
[0] -> 0 degree, 
[1] -> 45 degree, 
[2] -> 90 degree, 
[3] -> 135 degree,
[4] -> average, 
[5] -> range (max - min) 
*/
__device__ int       Extract_Texture_Features(TEXTURE *Texture, double **tone_matrix, double **buffer_matrix, double *vector_buffer, 
                                                       int distance, int angle, u_int8_t **grays, int rows, int cols, int max_val);



// Spacial Dependence matrix calculations
__device__ double** CoOcMat_Angle_0(double** tone_matrix, int distance, u_int8_t **grays, int rows, int cols, int* tone_LUT, int tone_count);
__device__ double** CoOcMat_Angle_45(double** tone_matrix, int distance, u_int8_t **grays, int rows, int cols, int* tone_LUT, int tone_count);
__device__ double** CoOcMat_Angle_90(double** tone_matrix, int distance, u_int8_t **grays, int rows, int cols, int* tone_LUT, int tone_count);
__device__ double** CoOcMat_Angle_135(double** tone_matrix, int distance, u_int8_t **grays, int rows, int cols, int* tone_LUT, int tone_count);



// Auxiliary
__device__ double   f1_asm(double **P, int Ng);
__device__ double   f2_contrast(double **P, int Ng);
__device__ double   f3_corr(double **P, double *px, int Ng);
__device__ double   f4_var(double **P, int Ng);
__device__ double   f5_idm(double **P, int Ng);
__device__ double   f6_savg(double **P, double *Pxpy, int Ng);
__device__ double   f7_svar(double **P, double *Pxpy, int Ng, double S);
__device__ double   f8_sentropy(double **P, double *Pxpy, int Ng);
__device__ double   f9_entropy(double **P, int Ng);
__device__ double   f10_dvar(double **P, double *Pxpy, int Ng);
__device__ double   f11_dentropy(double **P, double *Pxpy, int Ng);
__device__ double   f12_icorr(double **P, double *Pxpy, int Ng);
__device__ double   f13_icorr(double **P, double *Pxpy, int Ng);
__device__ double   f14_maxcorr(double **P, double **Q_buf, double *Pxpy,int Ng);

#endif
