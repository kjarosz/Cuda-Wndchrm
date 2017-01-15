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

#ifndef ZERNIKE_H
#define ZERNIKE_H



#include <vector>

#include "device_launch_parameters.h"

#include "../../cuda_signatures.h"
#include "../../image/image_matrix.h"



struct ZernikeData
{
  double *D;
  double *R;
  double **Y;
  double **X;
  double **P;

  double **zvalues;
  long   *output_size;
};



__global__ void cuda_zernike(CudaImages images, ZernikeData data);
__device__ void mb_Znl(long n, long l, double *X, double *Y, double *P, int size, double *out_r, double *out_i);
__device__ double mb_imgmoments(pix_data *pixels, int width, int height, int x, int y);
__device__ double factorial(double n);


long                        cuda_zernike_mem_estimate(ImageMatrix *image);
ZernikeData                 cuda_allocate_zernike_data(const std::vector<ImageMatrix *> &images);
std::vector<FileSignatures> cuda_get_zernike_signatures(const std::vector<ImageMatrix *> &images,
                                                        const ZernikeData &data, int image_count);
void                        cuda_delete_zernike_data(ZernikeData &data, int image_count);



#endif // ZERNIKE_H
