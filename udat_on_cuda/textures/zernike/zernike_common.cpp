#include "zernike_common.h"

#include <cuda_device_runtime_api.h>
#include <thrust/complex.h>



/* ************************************************************************* */
   __host__ __device__  double  factorial(double n)
/* ************************************************************************* */
{
  if (n<0)
    return 0;

  double ans = 1;
  for (int a = 1; a <= n; a++)
    ans = ans*a;

  return ans;
}



/* ************************************************************************* */
   __host__ __device__ double mb_imgmoments(ImageMatrix **image, int x, int y)
/* ************************************************************************* */
// Calculates the moment MXY for IMAGE
/* ************************************************************************* */
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  double sum = 0;
  /* Generate a matrix with the y coordinates of each pixel. */
  for (int col = 0; col < image[i]->width; col++) 
  {
    for (int row = 0; row < image[i]->height; row++)
    {
      double xStuff = pow((double)(col + 1), (double)x);

      if (y != 0)
      {
        if (x == 0)
          xStuff = pow((double)(row + 1), (double)y);
        else
          xStuff = pow((double)(col + 1), (double)y) * xStuff;
      }

      sum += xStuff * image[i]->pixel(col, row, 0).intensity;
    }
  }

  return sum;
}



/* ************************************************************************* */
   __host__ __device__ void mb_Znl(long n, long l, double *X, double *Y, 
                                   double *P, int size, double *out_r, 
                                   double *out_i)
/* ************************************************************************* */
// Zernike moment generating function.  The moment of degree n and
// angular dependence l for the pixels defined by coordinate vectors
// X and Y and intensity vector P.  X, Y, and P must have the same
// length
/* ************************************************************************* */
{
  double x, y, p;   /* individual values of X, Y, P */
  int i, m;
  double* preal;             /* Real part of return value */
  double* pimag;             /* Imag part of return value */

  // Accumulator for complex moments
  thrust::complex<float> sum = thrust::complex<float>(0.0, 0.0);
  for (i = 0; i < size; i++) {
    x = X[i];
    y = Y[i];
    p = P[i];

    /* Inner sum in Zernike calculations */
    thrust::complex<float> Vnl = thrust::complex<float>(0.0, 0.0);
    for (m = 0; m <= (n - l) / 2; m++) {
      double tmp = (pow((double)-1.0, (double)m)) * (factorial(n - m)) /
        (factorial(m) * (factorial((n - 2.0*m + l) / 2.0)) *
        (factorial((n - 2.0*m - l) / 2.0))) *
        (pow(sqrt(x*x + y*y), (double)(n - 2 * m)));

      Vnl = Vnl + tmp * /* convert angles to Rpolar */ Rpolar(1.0, l*atan2(y, x));
      /*
      NOTE: This function did not work with the following:
      ...pow((x*x + y*y), (double)(n/2 -m))...
      perhaps pow does not work properly with a non-integer
      second argument.
      'not work' means that the output did not match the 'old'
      Zernike calculation routines.
      */
    }

    /* sum += p * conj(Vnl) ; */
    sum = sum + p * Conjg(Vnl);
  }

  /* sum *= (n+1)/3.14159265 ; */
  sum = RCmul((n + 1) / 3.14159265, sum);


  /* Assign the returned value */
  *out_r = sum.r;
  *out_i = sum.i;
}
