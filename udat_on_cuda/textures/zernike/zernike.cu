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
#include "../../utils/cuda_utils.h"
#include "cuda_complex.h"
#include "zernike.h"

#include <iostream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

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
__device__ double mb_imgmoments(pix_data *pixels, int width, int height, int x, int y)
{ 
  double xcoord;
  double sum = 0;
  /* Generate a matrix with the y coordinates of each pixel. */
  for (int col = 0; col < width; col++) 
  {
    for (int row = 0; row < height; row++)
    {
       xcoord = pow((double)(col+1), (double)x);
       if (y != 0)
       {  
         if (x == 0) 
           xcoord = pow(double(row + 1), double(y));
         else
           xcoord = pow(double(col + 1), double(y)) * xcoord;
       }
       sum += xcoord * get_pixel(pixels, width, height, col, row, 0).intensity;
    }
  }

  return sum;
}



__device__ void mb_Znl(long n, long l, double *X, double *Y, double *P, int size, double *out_r, double *out_i)
{
  double x, y, p ;   /* individual values of X, Y, P */
  int i,m;
  fcomplex sum;              /* Accumulator for complex moments */
  fcomplex Vnl;              /* Inner sum in Zernike calculations */

  sum = Complex (0.0, 0.0);

  for(i = 0 ; i < size ; i++) {
    x = X[i] ;
    y = Y[i] ;
    p = P[i] ;

    Vnl = Complex (0.0, 0.0);
    for( m = 0; m <= (n-l)/2; m++) {
      double tmp = (pow((double)-1.0,(double)m)) * ( factorial(n-m) ) /
				( factorial(m) * (factorial((n - 2.0*m + l) / 2.0)) *
	  			(factorial((n - 2.0*m - l) / 2.0)) ) *
				( pow( sqrt(x*x + y*y), (double)(n - 2*m)) );

	  Vnl = Cadd (Vnl, RCmul(tmp, Rpolar(1.0, l*atan2(y,x))) );
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
	sum = Cadd(sum, RCmul(p, Conjg(Vnl)));
  }

  /* sum *= (n+1)/3.14159265 ; */
  sum = RCmul((n+1)/3.14159265, sum);


  /* Assign the returned value */
  *out_r = sum.r ;
  *out_i = sum.i ;

}



__global__ void cuda_zernike(CudaImages images, ZernikeData data)
{  
  int th_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int rows, cols;
  if (data.D[th_idx] <= 0) 
    data.D[th_idx] = 15;

  if (data.R[th_idx] <= 0)
  {  
    rows = images.heights[th_idx];
    cols = images.widths[th_idx];
    data.R[th_idx] = rows / 2;
  }

  // Find all non-zero pixel coordinates and values 
  double psum = 0;

  int size = 0;
  for (int y=0; y < rows; y++) 
  {
    for (int x=0; x < cols; x++) 
    {
      pix_data pixel = get_pixel(images.pixels[th_idx], 
                                 images.widths[th_idx], 
                                 images.heights[th_idx], 
                                 x, y, 0);
      if (pixel.intensity != 0)
      {  
        data.Y[th_idx][size] = double(y+1);
        data.X[th_idx][size] = double(x+1);
        data.P[th_idx][size] = double(pixel.intensity);
        psum += double(pixel.intensity);
        size++;
      }
    }
  }

  // Normalize the coordinates to the center of mass and normalize
  // pixel distances using the maximum radius argument (R) 
  double moment10 = mb_imgmoments(images.pixels[th_idx], images.widths[th_idx], images.heights[th_idx], 1, 0);
  double moment00 = mb_imgmoments(images.pixels[th_idx], images.widths[th_idx], images.heights[th_idx], 0, 0);
  double moment01 = mb_imgmoments(images.pixels[th_idx], images.widths[th_idx], images.heights[th_idx], 0, 1);

  int size2 = 0;
  for (int a = 0; a < size; a++)
  { 
    data.X[th_idx][size2] = (data.X[th_idx][a] - moment10/moment00)/data.R[th_idx];
    data.Y[th_idx][size2] = (data.Y[th_idx][a] - moment01/moment00)/data.R[th_idx];
    data.P[th_idx][size2] = data.P[th_idx][a] / psum;

    double squareX = data.X[th_idx][size2] * data.X[th_idx][size2];
    double squareY = data.Y[th_idx][size2] * data.Y[th_idx][size2];
    double radius = sqrt( squareX + squareY );
    if (radius <= 1.0) 
      size2++;
  }

  int size3 = 0;
  for (int n = 0; n <= data.D[th_idx]; n++) 
  {
    for (int l = 0; l <= n; l++) 
    {
      if (((n - l) % 2) == 0)
      {  
        double preal, pimag;
        mb_Znl(n, l, data.X[th_idx], data.Y[th_idx], data.P[th_idx], size2, &preal, &pimag);
        data.zvalues[th_idx][size3++] = fabs(sqrt(preal*preal + pimag*pimag));
      }
    }
  }
  data.output_size[th_idx] = size3;
}



long cuda_zernike_mem_estimate(ImageMatrix *image)
{
  return /* D */        sizeof(double) +
         /* R */        sizeof(double) +
         /* Y */        sizeof(double*) + sizeof(double) * image->width * image->height +
         /* X */        sizeof(double*) + sizeof(double) * image->width * image->height +
         /* P */        sizeof(double*) + sizeof(double) * image->width * image->height +
         /* zvalues */  sizeof(double*) + sizeof(double) * MAX_OUTPUT_SIZE +
         /* out_size */ sizeof(long);
}



ZernikeData cuda_allocate_zernike_data(const std::vector<ImageMatrix *> &images)
{
  ZernikeData zdata;
  memset(&zdata, 0, sizeof(ZernikeData));

  cudaMalloc(&zdata.D,    images.size() * sizeof(double));
  cudaMemset(zdata.D,  0, images.size() * sizeof(double));

  cudaMalloc(&zdata.R,    images.size() * sizeof(double));
  cudaMemset(zdata.R,  0, images.size() * sizeof(double));

  cudaMalloc(&zdata.output_size, images.size() * sizeof(long));
  cudaMemset(zdata.output_size, 0, images.size() * sizeof(long));

  unsigned int *sizes = new unsigned int[images.size()];
  for(unsigned int i = 0; i < images.size(); i++)
    sizes[i] = images[i]->width * images[i]->height;

  cuda_alloc_multivar_array<double>(sizes,       images.size(), zdata.Y);
  cuda_alloc_multivar_array<double>(sizes,       images.size(), zdata.X);
  cuda_alloc_multivar_array<double>(sizes,       images.size(), zdata.P);
  cuda_alloc_cube_array<double>(MAX_OUTPUT_SIZE, images.size(), zdata.zvalues);

  delete [] sizes;

  return zdata;
}



std::vector<FileSignatures> cuda_get_zernike_signatures(const std::vector<ImageMatrix *> &images, 
                                                        const ZernikeData &data, int image_count)
{
  cudaError status;
  long *output_size = new long[image_count];
  status = cudaMemcpy(output_size, data.output_size, image_count * sizeof(long), cudaMemcpyDeviceToHost);

  double **zvalues = new double*[image_count];
  status = cudaMemcpy(zvalues, data.zvalues, image_count * sizeof( double * ), cudaMemcpyDeviceToHost);
  for(int i = 0; i < image_count; i++) {
    double *zvals = new double[output_size[i]];
    status = cudaMemcpy(zvals, zvalues[i], output_size[i] * sizeof(double), cudaMemcpyDeviceToHost);
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
  cudaError status;
  status = cudaFree(data.D);
  status = cudaFree(data.R);
  status = cudaFree(data.output_size);

  cuda_free_multidim_arr<double>(data.Y,       image_count);
  cuda_free_multidim_arr<double>(data.X,       image_count);
  cuda_free_multidim_arr<double>(data.P,       image_count);
  cuda_free_multidim_arr<double>(data.zvalues, image_count);

  memset(&data, 0, sizeof(ZernikeData));
}



//CudaZernike2D::CudaZernike2D(const std::vector<ImageMatrix *> *images,
//                             const CudaImages *cuda_images)
//: CudaAlgorithm(images, cuda_images)
//{
//  data = new ZernikeData();
//  memset(data, 0, sizeof(ZernikeData));
//
//  cudaMalloc(&data->D,    images->size() * sizeof(double));
//  cudaMemset(data->D,  0, images->size() * sizeof(double));
//
//  cudaMalloc(&data->R,    images->size() * sizeof(double));
//  cudaMemset(data->R,  0, images->size() * sizeof(double));
//
//  cudaMalloc(&data->output_size, images->size() * sizeof(long));
//  cudaMemset(data->output_size, 0, images->size() * sizeof(long));
//
//  unsigned int *sizes = new unsigned int[images->size()];
//  for(unsigned int i = 0; i < images->size(); i++)
//    sizes[i] = (*images)[i]->width * (*images)[i]->height;
//
//  cuda_alloc_multivar_array<double>(sizes,       images->size(), data->Y);
//  cuda_alloc_multivar_array<double>(sizes,       images->size(), data->X);
//  cuda_alloc_multivar_array<double>(sizes,       images->size(), data->P);
//  cuda_alloc_cube_array<double>(MAX_OUTPUT_SIZE, images->size(), data->zvalues);
//
//  delete [] sizes;
//}
//
//
//
//CudaZernike2D::~CudaZernike2D()
//{
//  cudaError status;
//  status = cudaFree(data->D);
//  status = cudaFree(data->R);
//  status = cudaFree(data->output_size);
//
//  cuda_free_multidim_arr<double>(data->Y,       images->size());
//  cuda_free_multidim_arr<double>(data->X,       images->size());
//  cuda_free_multidim_arr<double>(data->P,       images->size());
//  cuda_free_multidim_arr<double>(data->zvalues, images->size());
//
//  memset(&data, 0, sizeof(ZernikeData));
//
//  delete data;
//}
//
//
//
//void CudaZernike2D::print_message() const
//{
//  std::cout << "Calculating Zernike features." << std::endl;
//}
//
//
//
//void CudaZernike2D::compute()
//{
//  cuda_zernike<<< 1, images->size()>>>(*cuda_images, *data);
//}
//
//
//
//std::vector<FileSignatures> CudaZernike2D::get_signatures() const
//{
//  cudaError status;
//
//  long *output_size = new long[images->size()];
//  status = cudaMemcpy(output_size, data->output_size, images->size() * sizeof(long), cudaMemcpyDeviceToHost);
//
//  double **zvalues = new double*[images->size()];
//  status = cudaMemcpy(zvalues, data->zvalues, images->size() * sizeof( double * ), cudaMemcpyDeviceToHost);
//  for(int i = 0; i < images->size(); i++) {
//    double *zvals = new double[output_size[i]];
//    status = cudaMemcpy(zvals, zvalues[i], output_size[i] * sizeof(double), cudaMemcpyDeviceToHost);
//    zvalues[i] = zvals;
//  }
//
//  std::vector<FileSignatures> file_signatures;
//  for(int i = 0; i < images->size(); i++) 
//  {
//    FileSignatures file_signature;
//    file_signature.file_name = (*images)[i]->source_file;
//    int x = 0;
//    int y = 0;
//    for(int j = 0; j < output_size[i]; j++) 
//    {
//      std::stringstream ss;
//      ss << "ZernikeMoments Z_" << y << "_" << x;
//
//      Signature signature;
//      signature.signature_name = ss.str();
//      signature.value = zvalues[i][j];
//
//      file_signature.signatures.push_back(signature);
//
//      if (x >= y)
//        x = 1 - (y++ % 2);
//      else
//        x += 2;
//    }
//    file_signatures.push_back(file_signature);
//  }
//
//  for(int i = 0; i < images->size(); i++) {
//    delete [] zvalues[i];
//  }
//  delete [] zvalues;
//  delete [] output_size;
//
//  return file_signatures;
//}