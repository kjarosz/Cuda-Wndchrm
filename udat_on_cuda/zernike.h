#ifndef ZERNIKE_H
#define ZERNIKE_H



#include "image_matrix.h"
#include "cuda_runtime.h"



__device__ double factorial     (double n);
__device__ double image_moments (pix_data *image, int width, int height, int x, int y);
__device__ void   znl           (long n, long l, double *X, double *Y, double *P, 
                                 int size, double *out_r, double *out_i);
__device__ void   zernike       (pix_data **images, int *widths, int *heights, int *depths, 
                                 int size, double *d, double *r, double *zvalues, 
                                 long *output_size);



#endif // ZERNIKE_H