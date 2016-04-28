#include "zernike.h"



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
  for (int i = 0; i < size; i++) {
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
   __device__ 
   void zernike (pix_data **images, 
                 int *widths, int *heights, int *depths, int count, 
                 double *d, double *r, double *zvalues, 
                 long *output_size)
/* ************************************************************************* */
{
   int n,l;
   int size2,size3,a;

   int *size = new int[count];
   int *size2 = new int[count];
   int *size3 = new int[count];

   int *rows = new int[count];
   int *cols = new int[count];

   double **X = new double*[count];
   double **Y = new double*[count];
   double **P = new double*[count];

   double *psum = new double[count];
   for (int i = 0; i < count; i++)
   {
      psum[i] = 0;

      if (d[i] <= 0) 
        d[i] = 15;

      if (r[i] <= 0)
      {  
         rows[i] = images[i].height;
         cols[i] = images[i].width;
         r[i]    = rows[i] / 2;
      }

      Y[i] = new double[rows[i]*cols[i]];
      X[i] = new double[rows[i]*cols[i]];
      P[i] = new double[rows[i]*cols[i]];

      /* Find all non-zero pixel coordinates and values */
      size[i] = 0;
      for (int y = 0; y < rows[i]; y++)
      {
        for (int x = 0; x < cols[i]; x++)
        {
           if (images[i].pixel(x,y,0).intensity != 0)
           {  
              Y[i][size[i]] = y+1;
              X[i][size[i]] = x+1;
              P[i][size[i]] = images[i].pixel(x,y,0).intensity;
              psum[i] += images[i].pixel(x,y,0).intensity;
              size[i]++;
           }
        }
      }
   }

   /* Normalize the coordinates to the center of mass and normalize
      pixel distances using the maximum radius argument (R) */
   double *moment10 = new double[count];
   double *moment00 = new double[count];
   double *moment01 = new double[count];

   mb_imgmoments(images, 1, 0, moment10);
   mb_imgmoments(images, 0, 0, moment00);
   mb_imgmoments(images, 0, 1, moment01);

   for(int i = 0; i < count; i++)
   {
      size2[i]=0;
      for (int a = 0; a < size; a++)
      { 
         X[size2] = (X[a] - moment10[i] / moment00[i]) / r;
         Y[size2] = (Y[a] - moment01[i] / moment00[i]) / r;
         P[size2] = P[a] / psum;
         if (sqrt(X[size2]*X[size2]+Y[size2]*Y[size2])<=1) size2=size2++;
      }
   }

   delete [] moment10;
   delete [] moment00;
   delete [] moment01;

   size3=0;
   for (n=0;n<=D;n++)
     for (l=0;l<=n;l++)
       if (((n-l) % 2) ==0)
       {  double preal,pimag;
          mb_Znl(n,l,X,Y,P,size2,&preal,&pimag);
          zvalues[size3++]=fabs(sqrt(preal*preal+pimag*pimag));
       }
   *output_size=size3;

   for (int i = 0; i < count; i++)
   {
      delete [] Y[i];
      delete [] X[i];
      delete [] P[i];
   }

   delete [] Y;
   delete [] X;
   delete [] P;

   delete [] rows;
   delete [] cols;

   delete [] size;
   delete [] size2;
   delete [] size3;
}