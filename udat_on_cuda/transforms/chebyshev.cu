/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*                                                                               */
/*    This file is part of Cuda-Wndchrm.                                         */
/*    Copyright (C) 2017 Kamil Jarosz, Christopher K. Horton and Tyler Wiersing  */
/*                                                                               */
/*    This library is free software; you can redistribute it and/or              */
/*    modify it under the terms of the GNU Lesser General Public                 */
/*    License as published by the Free Software Foundation; either               */
/*    version 2.1 of the License, or (at your option) any later version.         */
/*                                                                               */
/*    This library is distributed in the hope that it will be useful,            */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of             */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU          */
/*    Lesser General Public License for more details.                            */
/*                                                                               */
/*    You should have received a copy of the GNU Lesser General Public           */
/*    License along with this library; if not, write to the Free Software        */
/*    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA  */
/*                                                                               */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <sstream>

#include "chebyshev.h"

#include "../utils/cuda_utils.h"

/* inputs:
IM - image
N - coefficient
*/
__global__ void cuda_chebyshev(CudaImages images, ChebyshevData data)
{
  unsigned long th_idx = blockIdx.x * blockDim.x + threadIdx.x;

  pix_data *pixels  = images.pixels[th_idx];
  int       width   = images.widths[th_idx];
  int       height  = images.heights[th_idx];

  int       N       = data.N[th_idx];
  double   *out     = data.out[th_idx];

  double   *x       = data.x[th_idx];
  double   *y       = data.y[th_idx];
  double   *in      = data.in[th_idx];

  double   *c_y     = data.c_y[th_idx];
  double   *c_y_out = data.c_y_out[th_idx];
  double   *Tj      = data.c_Tj[th_idx];
  double   *tj      = data.c_tj[th_idx];
  double   *tnx1    = data.tnx1[th_idx];
  double   *tnx2    = data.tnx2[th_idx];



  for (int a = 0; a < width; a++)
    x[a] = 2 * (double)(a + 1) / (double)width - 1;

  for (int a = 0; a < height; a++)
    y[a] = 2 * (double)(a + 1) / (double)height -1;

  for (int j = 0; j < height; j++)
    for (int i = 0; i < width; i++)
      in[j*width + i] = (double)get_pixel(pixels, width, height, i, j, 0).intensity;

  cuda_chebyshev_coefficients(in, out, x, N, width, height,
    c_y, c_y_out, Tj, tj, tnx1, tnx2);

  /* transpose the matrix "out" into "in" */
  for (int j = 0; j < N; j++)
    for (int i = 0; i < height; i++)
      in[j*height + i] = out[i*N + j];

  cuda_chebyshev_coefficients(in, out, y, N, height, N,
    c_y, c_y_out, Tj, tj, tnx1, tnx2);
}



__device__
void cuda_chebyshev_coefficients(
  double *intensities, double *out, double *x,
  int N, int width, int height, double *y, double *y_out,
  double *Tj, double *tj, double *tnx1, double *tnx2)
{
  for(int iy = 0; iy < height; iy++)
  {
    for(int ix = 0; ix < width; ix++)
      y[ix] = intensities[iy*width + ix];

    cuda_chebyshev_coefficients_1D(y, y_out, x, N, width, Tj, tj, tnx1, tnx2);

    for(int ix = 0; ix < N; ix++)
      out[iy*N + ix] = y_out[ix];
  }
}



__device__
void cuda_chebyshev_coefficients_1D(
  double *f, double *out, double *x,
  int N, int width, double *Tj, double *tj,
  double *tnx1, double *tnx2)
{
  cuda_TNx(x, Tj, N, width, tnx1, tnx2);
  for(int jj = 0; jj < N; jj++)
  {
    int jx = jj;
    for(int a = 0; a < width; a++)
      tj[a] = Tj[a*N + jj];

    if (!jx)
    {
      for(int a = 0; a < width; a++)
        tj[a] = tj[a] / (double)width;
    }
    else
    {
      for(int a = 0; a < width; a++)
        tj[a] = tj[a] * 2 / (double)width;
    }

    out[jj] = 0;
    for(int a = 0; a < width; a++)
      out[jj] += f[a] * tj[a] / 2;
  }
}



__device__
void cuda_TNx(double *x, double *out, int N, int height, double *temp, double *temp1)
{
     // x'*ones(1,N)
   for (int ix=0;ix<N;ix++)
     for (int iy=0;iy<height;iy++)
       temp[iy*N+ix]=x[iy];

   // acos
   for (int ix=0;ix<N;ix++)
     for (int iy=0;iy<height;iy++)
       if (fabs(temp[iy*N+ix])>1) temp[iy*N+ix]=0;   /* protect from acos domain error */
       else temp[iy*N+ix]=acos(temp[iy*N+ix]);
   // ones(size(x,2),1)*(0:(N-1))
   for (int ix=0;ix<N;ix++)
     for (int iy=0;iy<height;iy++)
       temp1[iy*N+ix]=ix;
   //.*
   for (int ix=0;ix<N;ix++)
     for (int iy=0;iy<height;iy++)
       out[iy*N+ix]=temp[iy*N+ix]*temp1[iy*N+ix];
   //cos
   for (int ix=0;ix<N;ix++)
     for (int iy=0;iy<height;iy++)
       out[iy*N+ix]=cos(out[iy*N+ix]);

   for (int iy=0;iy<height;iy++)
       out[iy*N+0]=1;
}



ChebyshevData cuda_allocate_chebyshev_data(const std::vector<ImageMatrix *> &images)
{
  ChebyshevData data;

  int *N = new int[images.size()];
  unsigned int *row_sizes = new unsigned int[images.size()];

  for(int i = 0; i < images.size(); i++)
    N[i] = min(images[i]->width, images[i]->height);
  move_host_to_cuda<int>(N, images.size(), data.N);

  for(int i = 0; i < images.size(); i++)
    row_sizes[i] = N[i] * images[i]->height;
  cuda_alloc_multivar_array<double>(row_sizes, images.size(), data.out);

  for(int i = 0; i < images.size(); i++)
    row_sizes[i] = images[i]->width;
  cuda_alloc_multivar_array<double>(row_sizes, images.size(), data.x);

  for(int i = 0; i < images.size(); i++)
    row_sizes[i] = images[i]->height;
  cuda_alloc_multivar_array<double>(row_sizes, images.size(), data.y);

  for(int i = 0; i < images.size(); i++)
    row_sizes[i] = images[i]->width * images[i]->height;
  cuda_alloc_multivar_array<double>(row_sizes, images.size(), data.in);

  cuda_alloc_cube_array<double>(CHEBYSHEV_COEFF_COUNT, images.size(), data.c_y_out);

  for(int i = 0; i < images.size(); i++)
    row_sizes[i] = max(images[i]->width, images[i]->height);
  cuda_alloc_multivar_array<double>(row_sizes, images.size(), data.c_y);
  cuda_alloc_multivar_array<double>(row_sizes, images.size(), data.c_tj);

  for(int i = 0; i < images.size(); i++)
    row_sizes[i] *= CHEBYSHEV_COEFF_COUNT;
  cuda_alloc_multivar_array<double>(row_sizes, images.size(), data.c_Tj);
  cuda_alloc_multivar_array<double>(row_sizes, images.size(), data.tnx1);
  cuda_alloc_multivar_array<double>(row_sizes, images.size(), data.tnx2);

  delete [] row_sizes;
  delete [] N;

  return data;
}



std::vector<FileSignatures> cuda_get_chebyshev_signatures(const std::vector<ImageMatrix *> &images, const ChebyshevData &data)
{
  int *N;
  move_cuda_to_host<int>(data.N, images.size(), N);

  double **out;
  move_cuda_to_host<double*>(data.out, images.size(), out);
  for(int i = 0; i < images.size(); i++)
  {
    double *o;
    move_cuda_to_host<double>(out[i], N[i], o);
    out[i] = o;
  }

  std::vector<FileSignatures> file_signatures;
  for(int i = 0; i < images.size(); i++)
  {
    FileSignatures f_signatures;
    f_signatures.file_name = images[i]->source_file;

    for(int j = 0; j < N[i]; j++)
    {
      std::stringstream ss;
      ss << "Chebyshev bin " << j;

      Signature signature;
      signature.signature_name = ss.str();
      signature.value = out[i][j];

      f_signatures.signatures.push_back(signature);
    }

    file_signatures.push_back(f_signatures);
  }

  delete [] N;
  for(int i = 0; i < images.size(); i++)
    delete [] out[i];
  delete [] out;

  return file_signatures;
}



void cuda_delete_chebyshev_data(const std::vector<ImageMatrix *> &images, ChebyshevData &data)
{
  cuda_free_multidim_arr<double>(data.x,       images.size());
  cuda_free_multidim_arr<double>(data.y,       images.size());
  cuda_free_multidim_arr<double>(data.in,      images.size());
  cuda_free_multidim_arr<double>(data.out,     images.size());
  cudaFree(data.N);

  cuda_free_multidim_arr<double>(data.c_y,     images.size());
  cuda_free_multidim_arr<double>(data.c_y_out, images.size());
  cuda_free_multidim_arr<double>(data.c_Tj,    images.size());
  cuda_free_multidim_arr<double>(data.c_tj,    images.size());
  cuda_free_multidim_arr<double>(data.tnx1,    images.size());
  cuda_free_multidim_arr<double>(data.tnx2,    images.size());
}
