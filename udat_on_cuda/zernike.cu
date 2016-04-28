#include "zernike.h"
#include "signatures.h"



#include <thrust/complex.h>



/* ************************************************************************* */
   __device__  double  factorial(double n)
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
   __device__ double image_moments(pix_data *image, int width, int height,
                                   int x, int y)
/* ************************************************************************* */
// Calculates the moment MXY for IMAGE
/* ************************************************************************* */
{
  double sum = 0;
  /* Generate a matrix with the y coordinates of each pixel. */
  for (int col = 0; col < width; col++) 
  {
    for (int row = 0; row < height; row++)
    {
      double xStuff = pow((double)(col + 1), (double)x);

      if (y != 0)
      {
        if (x == 0)
          xStuff = pow((double)(row + 1), (double)y);
        else
          xStuff = pow((double)(col + 1), (double)y) * xStuff;
      }

      sum += xStuff * get_pixel(image, col, row, 0).intensity;
    }
  }

  return sum;
}



/* ************************************************************************* */
   __device__ void znl(long n, long l, double *X, double *Y, 
                       double *P, int size, double *out_r, 
                       double *out_i)
/* ************************************************************************* */
// Zernike moment generating function.  The moment of degree n and
// angular dependence l for the pixels defined by coordinate vectors
// X and Y and intensity vector P.  X, Y, and P must have the same
// length
/* ************************************************************************* */
{
  // Accumulator for complex moments
  thrust::complex<double> sum = thrust::complex<double>(0.0, 0.0);
  thrust::complex<double> ctemp = thrust::complex<double>(0.0, 0.0);
  for (int i = 0; i < size; i++) 
  {
    double x = X[i];
    double y = Y[i];
    double p = P[i];

    /* Inner sum in Zernike calculations */
    thrust::complex<double> Vnl = thrust::complex<double>(0.0, 0.0);
    for (int m = 0; m <= (n - l) / 2; m++) {
      double tmp = (pow((double)-1.0, (double)m)) * (factorial(n - m)) /
        (factorial(m) * (factorial((n - 2.0*m + l) / 2.0)) *
        (factorial((n - 2.0*m - l) / 2.0))) *
        (pow(sqrt(x*x + y*y), (double)(n - 2 * m)));

      double tmp2 = l * atan2(y, x);
      ctemp.real(cos(tmp2));
      ctemp.imag(sin(tmp2));
      /* convert angles to Rpolar */ // Rpolar(1.0, tmp2);
      Vnl = Vnl + tmp * ctemp;  

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
    sum = sum + thrust::conj<double>(Vnl) * p;
  }

  /* sum *= (n+1)/3.14159265 ; */
  sum = ((n + 1) / 3.14159265) * sum;


  /* Assign the returned value */
  *out_r = sum.real();
  *out_i = sum.imag();
}



/* ************************************************************************* */
   __global__ 
   void zernike (pix_data **images, 
                 int *widths, int *heights, int *depths, 
                 double *D, double *R, 
                 double *zvalues, long *output_size)
/* ************************************************************************* */
{  
   int img_idx = blockDim.x * blockIdx.x + threadIdx.x;

   if (D[img_idx] <= 0) 
      D[img_idx] = 15;

   int rows, cols;
   if (R[img_idx] <= 0)
   {  
      rows = heights[img_idx];
      cols = widths[img_idx];
      R[img_idx] = rows/2;
   }

   /* Normalize the coordinates to the center of mass and normalize
      pixel distances using the maximum radius argument (R) */
   double moment10 = image_moments(images[img_idx], widths[img_idx], heights[img_idx], 1, 0);
   double moment00 = image_moments(images[img_idx], widths[img_idx], heights[img_idx], 0, 0);
   double moment01 = image_moments(images[img_idx], widths[img_idx], heights[img_idx], 0, 1);

   double *Y = new double[rows*cols];
   double *X = new double[rows*cols];
   double *P = new double[rows*cols];

   /* Find all non-zero pixel coordinates and values */
   int size=0;
   double psum=0;
   for (int y=0; y < rows; y++)
   {
     for (int x=0; x < cols; x++)
     {
        pix_data pixel = get_pixel(images[img_idx], widths[img_idx], heights[img_idx], x, y, 0);
        if (pixel.intensity!=0)
        {  
           Y[size] = y+1;
           X[size] = x+1;
           P[size] = pixel.intensity;
           psum   += pixel.intensity;
           size++;
        }
     }
   }

   int size2=0;
   for (int a=0; a < size; a++)
   {  
      X[size2] = (X[a]-moment10/moment00)/R[img_idx];
      Y[size2] = (Y[a]-moment01/moment00)/R[img_idx];
      P[size2] = P[a]/psum;
      if ( sqrt( X[size2] * X[size2] + Y[size2] * Y[size2] ) <= 1) 
         size2=size2++;
   }

   int size3=0;
   for (int n=0; n <= D[img_idx]; n++)
   {
     for (int l=0; l <= n ; l++)
     {
       if (((n-l) % 2) ==0)
       {  
          double preal,pimag;
          znl(n,l,X,Y,P,size2,&preal,&pimag);
          zvalues[img_idx * MAX_OUTPUT_SIZE + size3++]=fabs(sqrt(preal*preal+pimag*pimag));
       }
     }
   }
   output_size[img_idx] = size3;

   delete Y;
   delete X;
   delete P;

}
