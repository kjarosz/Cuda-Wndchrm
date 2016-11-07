/*
 * Included in OME by Tom Macura so that Michael Boland's mb_Znl.cpp doesn't 
 * require system specific complex arithmetic library.
 *
 * http://www.csounds.com/developers/html/complex_8c-source.html
 * Code from Press, Teukolsky, Vettering and Flannery
 * Numerical Recipes in C, 2nd Edition, Cambridge 1992.
*/

#ifndef _CUDA_COMPLEX_H_
#define _CUDA_COMPLEX_H_

#ifndef _FCOMPLEX_DECLARE_T_
typedef struct FCOMPLEX {double r,i;} fcomplex;
#define _FCOMPLEX_DECLARE_T_
#endif /* _FCOMPLEX_DECLARE_T_ */

#include <cuda_runtime.h>

#define CUDA_ABS(x) ((x) >= 0 ? (x) : -(x))

__host__ __device__ fcomplex Cadd(fcomplex a, fcomplex b);
__host__ __device__ fcomplex Csub(fcomplex a, fcomplex b);
__host__ __device__ fcomplex Cmul(fcomplex a, fcomplex b);
__host__ __device__ fcomplex Complex(double re, double im);
__host__ __device__ fcomplex Conjg(fcomplex z);
__host__ __device__ fcomplex Cdiv(fcomplex a, fcomplex b);
__host__ __device__ double Cabs(fcomplex z);
__host__ __device__ fcomplex Csqrt(fcomplex z);
__host__ __device__ fcomplex RCmul(double x, fcomplex a);
__host__ __device__ fcomplex Rpolar (double rho, double theta);

#endif /* _CUDA_COMPLEX_H_ */
