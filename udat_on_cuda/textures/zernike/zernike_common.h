#ifndef zernike_commonH
#define zernike_commonH

#include <cuda_runtime.h>

#include "../../cmatrix.h"

__host__ __device__ double factorial(double n);
__host__ __device__ double mb_imgmoments(ImageMatrix *image, int x, int y);
__host__ __device__ void   mb_Znl(long n, long l, double *X, double *Y, double *P, int size, double *out_r, double *out_i)

#endif // zernike_commonH