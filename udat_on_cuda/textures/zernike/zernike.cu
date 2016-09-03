/****************************************************************************/
/*                                                                          */
/*                                                                          */
/*                              mb_Znl.c                                    */
/*                                                                          */
/*                                                                          */
/*                           Michael Boland                                 */
/*                            09 Dec 1998                                   */
/*                                                                          */     
/*  Revisions:                                                              */
/*  9-1-04 Tom Macura <tmacura@nih.gov> modified to make the code ANSI C    */
/*         and work with included complex arithmetic library from           */
/*         Numerical Recepies in C instead of using the system's C++ STL    */
/*         Libraries.                                                       */
/*                                                                          */
/*  1-29-06 Lior Shamir <shamirl (-at-) mail.nih.gov> modified "factorial"  */
/*  to a loop, replaced input structure with ImageMatrix class.             */
/****************************************************************************/


//---------------------------------------------------------------------------

#pragma hdrstop

#include "../../cuda_signatures.h"
#include "zernike.h"

#include <thrust/complex.h>

//---------------------------------------------------------------------------

__device__ double factorial(double n)
{ 
  if (n < 0) 
    return 0;

  double ans = 1;
  for (int a = 1; a <= n; a++)
    ans *= a;
  return ans;
}

/* mb_imgmoments
   calculates the moment MXY for IMAGE

*/
__device__ double mb_imgmoments(pix_data *pixels, int width, int height, int x, int y, double *xcoords)
{ 
  /* Generate a matrix with the x coordinates of each pixel. */
  for (int row = 0; row < height; row++)
    for (int col = 0; col < width; col++)
      xcoords[row*width + col] = pow(col + 1, x);

  double sum = 0;
  /* Generate a matrix with the y coordinates of each pixel. */
  for (int col = 0; col < width; col++) 
  {
    for (int row = 0; row < height; row++)
    {
       if (y != 0)
       {  
         if (x == 0) 
           xcoords[row*width + col] = pow(row + 1, y);
         else
           xcoords[row*width + col] = pow(col + 1, y) * xcoords[row*width + col];
       }
       sum += xcoords[row*width + col] * get_pixel(pixels, width, height, col, row, 0).intensity;
    }
  }

  return sum;
}



__device__ thrust::complex<double> Rpolar(double rho, double theta)
{
  return thrust::complex<double>(
      rho * cos(theta),
      rho * sin(theta)
    );
}


/* mb_Znl
  Zernike moment generating function.  The moment of degree n and
  angular dependence l for the pixels defined by coordinate vectors
  X and Y and intensity vector P.  X, Y, and P must have the same
  length
*/
__device__ void mb_Znl(long n, long l, double *X, double *Y, double *P, int size, double *out_r, double *out_i)
{ 
  // Accumulator for complex moments
  thrust::complex<double> sum = thrust::complex<double> (0.0, 0.0);
  for(int i = 0 ; i < size ; i++) 
  {
    double x = X[i] ;
    double y = Y[i] ;
    double p = P[i] ;

    // Inner sum in Zernike calculations
    thrust::complex<double> Vnl = thrust::complex<double> (0.0, 0.0);
    for(int m = 0; m <= (n-l)/2; m++) 
    {
      double tmp = pow(double(-1.0), double(m)) * factorial(n-m) /
				( factorial(m) * factorial((n - 2.0*m + l) / 2.0) * factorial((n - 2.0*m - l) / 2.0) ) *
				( pow(sqrt(X[i]*X[i] + Y[i]*Y[i]), double(n - 2*m)) );

      Vnl = Vnl + tmp * Rpolar(1.0, l * atan2(Y[i], X[i])) ;
      /*
       NOTE: This function did not work with the following:
        ...pow((x*x + y*y), (double)(n/2 -m))...
        perhaps pow does not work properly with a non-integer
        second argument.
       'not work' means that the output did not match the 'old'
        Zernike calculation routines.
      */
    }

    sum = sum + p * thrust::conj<double>(Vnl);
  }

  /* sum *= (n+1)/3.14159265 ; */
  sum = ((n+1)/3.14159265) * sum;


  /* Assign the returned value */
  *out_r = sum.real() ;
  *out_i = sum.imag() ;
}


__global__ void cuda_zernike(CudaImages images, ZernikeData data)
{  
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;


  int rows, cols;
  if (data.D[thread_idx] <= 0) 
    data.D[thread_idx] = 15;

  if (data.R[thread_idx] <= 0)
  {  
    rows = images.heights[thread_idx];
    cols = images.widths[thread_idx];
    data.R[thread_idx] = rows / 2;
  }

  /* Find all non-zero pixel coordinates and values */
  double psum = 0;

  int size = 0;
  for (int y=0; y < rows; y++) 
  {
    for (int x=0; x < cols; x++) 
    {
      pix_data pixel = get_pixel(images.pixels[thread_idx], 
                                 images.widths[thread_idx], images.heights[thread_idx], 
                                 x, y, 0);
      if (pixel.intensity != 0)
      {  
        data.Y[thread_idx][size] = y+1;
        data.X[thread_idx][size] = x+1;
        data.P[thread_idx][size] = pixel.intensity;
        psum += pixel.intensity;
        size++;
      }
    }
  }

  /* Normalize the coordinates to the center of mass and normalize
     pixel distances using the maximum radius argument (R) */
  double moment10 = mb_imgmoments(images.pixels[thread_idx], images.widths[thread_idx], images.heights[thread_idx], 
                                  1, 0, data.xcoords[thread_idx]);
  double moment00 = mb_imgmoments(images.pixels[thread_idx], images.widths[thread_idx], images.heights[thread_idx], 
                                  0, 0, data.xcoords[thread_idx]);
  double moment01 = mb_imgmoments(images.pixels[thread_idx], images.widths[thread_idx], images.heights[thread_idx], 
                                  0, 1, data.xcoords[thread_idx]);

  int size2 = 0;
  for (int a = 0; a < size; a++)
  { 
    data.X[thread_idx][size2] = (data.X[thread_idx][a] - moment10/moment00)/data.R[thread_idx];
    data.Y[thread_idx][size2] = (data.Y[thread_idx][a] - moment01/moment00)/data.R[thread_idx];
    data.P[thread_idx][size2] = data.P[thread_idx][a] / psum;

    double squareX = data.X[thread_idx][size2] * data.X[thread_idx][size2];
    double squareY = data.Y[thread_idx][size2] * data.Y[thread_idx][size2];
    double radius = sqrt( squareX + squareY );
    if (radius <= 1.0) 
      size2 = size2++;
  }

  int size3 = 0;
  for (int n = 0; n <= data.D[thread_idx]; n++) 
  {
    for (int l = 0; l <= n; l++) 
    {
      if (((n - l) % 2) == 0)
      {  
        double preal, pimag;
        mb_Znl(n, l, data.X[thread_idx], data.Y[thread_idx], data.P[thread_idx], size2, &preal, &pimag);
        data.zvalues[thread_idx][size3++] = fabs(sqrt(preal*preal + pimag*pimag));
      }
    }
  }
  data.output_size[thread_idx] = size3;
}



ZernikeData cuda_allocate_zernike_data(const CudaImages &images)
{
  ZernikeData zdata;

  cudaMalloc(&zdata.D,    images.count * sizeof(double));
  cudaMemset(zdata.D,  0, images.count * sizeof(double));

  cudaMalloc(&zdata.R,    images.count * sizeof(double));
  cudaMemset(zdata.R,  0, images.count * sizeof(double));

  double **Y             = new double*[images.count];
  double **X             = new double*[images.count];
  double **P             = new double*[images.count];
  double **xcoords       = new double*[images.count];
  double **output_arrays = new double*[images.count];

  for(int i = 0; i < images.count; i++)
  {
    long arr_size = images.widths[i] * images.heights[i] * sizeof(double);
    cudaMalloc(&Y[i], arr_size);
    cudaMalloc(&X[i], arr_size);
    cudaMalloc(&P[i], arr_size);
    cudaMalloc(&xcoords[i], arr_size);
    cudaMalloc(&output_arrays[i], MAX_OUTPUT_SIZE * sizeof(double));
  }

  cudaMalloc(&zdata.X, images.count * sizeof(double *));
  cudaMemcpy(zdata.X, X, images.count * sizeof(double *), cudaMemcpyHostToDevice);
  delete [] X;

  cudaMalloc(&zdata.Y, images.count * sizeof(double *));
  cudaMemcpy(zdata.Y, Y, images.count * sizeof(double *), cudaMemcpyHostToDevice);
  delete [] Y;

  cudaMalloc(&zdata.P, images.count * sizeof(double *));
  cudaMemcpy(zdata.P, P, images.count * sizeof(double *), cudaMemcpyHostToDevice);
  delete [] P;

  cudaMalloc(&zdata.xcoords, images.count * sizeof(double *));
  cudaMemcpy(zdata.xcoords, xcoords, images.count * sizeof(double *), cudaMemcpyHostToDevice);
  delete [] xcoords;

  cudaMalloc(&zdata.zvalues, images.count * sizeof(double *));
  cudaMemcpy(zdata.zvalues, output_arrays, images.count * sizeof(double *), cudaMemcpyHostToDevice);
  delete [] output_arrays;

  cudaMalloc(&zdata.output_size, images.count * sizeof(long));

  return zdata;
}



std::vector<FileSignatures> cuda_get_zernike_signatures(const std::vector<ImageMatrix *> &images, 
                                                        const ZernikeData &data, int image_count)
{
  long *output_size = new long[image_count];
  cudaMemcpy(output_size, data.output_size, image_count * sizeof(long), cudaMemcpyDeviceToHost);

  double **zvalues = new double*[image_count];
  cudaMemcpy(zvalues, data.zvalues, image_count * sizeof( double * ), cudaMemcpyDeviceToHost);
  for(int i = 0; i < image_count; i++) {
    double *zvals = new double[output_size[i]];
    cudaMemcpy(zvals, zvalues[i], output_size[i] * sizeof(double), cudaMemcpyDeviceToHost);
    zvalues[i] = zvals;
  }

  std::vector<FileSignatures> file_signatures;
  for(int i = 0; i < image_count; i++) 
  {
    FileSignatures file_signature;
    file_signature.file_name = images[i]->source_file;
    int x = 0;
    int y = 0;
    for(int j = 0; j < output_size[i]; j++) 
    {
      std::stringstream ss;
      ss << "ZernikeMoments Z_" << y << "_" << x;

      Signature signature;
      signature.signature_name = ss.str();
      signature.value = zvalues[i][j];

      file_signature.signatures.push_back(signature);

      if (x >= y)
        x = 1 - (y++ % 2);
      else
        x += 2;
    }
    file_signatures.push_back(file_signature);
  }

  for(int i = 0; i < image_count; i++) {
    delete [] zvalues[i];
  }
  delete [] zvalues;
  delete [] output_size;

  return file_signatures;
}



void cuda_delete_zernike_data(ZernikeData &data, int image_count)
{
  cudaFree(&data.D);
  cudaFree(&data.R);

  double **Y       = new double*[image_count];
  double **X       = new double*[image_count];
  double **P       = new double*[image_count];
  double **xcoords = new double*[image_count];
  double **zvalues = new double*[image_count];

  cudaMemcpy(Y,       data.Y,       image_count * sizeof(double *), cudaMemcpyDeviceToHost);
  cudaMemcpy(X,       data.X,       image_count * sizeof(double *), cudaMemcpyDeviceToHost);
  cudaMemcpy(P,       data.P,       image_count * sizeof(double *), cudaMemcpyDeviceToHost);
  cudaMemcpy(xcoords, data.xcoords, image_count * sizeof(double *), cudaMemcpyDeviceToHost);
  cudaMemcpy(zvalues, data.zvalues, image_count * sizeof(double *), cudaMemcpyDeviceToHost);

  for(int i = 0; i < image_count; i++)
  {
    cudaFree(&Y[i]);
    cudaFree(&X[i]);
    cudaFree(&P[i]);
    cudaFree(&xcoords[i]);
    cudaFree(&zvalues[i]);
  }

  delete [] Y;
  delete [] X;
  delete [] P;
  delete [] xcoords;
  delete [] zvalues;

  cudaFree(data.Y);
  cudaFree(data.X);
  cudaFree(data.P);
  cudaFree(data.xcoords);
  cudaFree(data.zvalues);
  cudaFree(data.output_size);

  memset(&data, 0, sizeof(ZernikeData));
}



#pragma package(smart_init)


