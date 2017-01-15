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

//---------------------------------------------------------------------------

#ifndef _HARALICK_H
#define _HARALICK_H
//---------------------------------------------------------------------------

#include "../../cuda_signatures.h"
#include "../../image/image_matrix.h"
#include "device_launch_parameters.h"

const unsigned int HARALICK_FEATURE_SIZE               = 14;
const unsigned int HARALICK_OUT_SIZE                   = 28;
const unsigned int HARALICK_TONE_MAX                   = 255;
const unsigned int HARALICK_BUF_VEC_COUNT              = 4;

struct HaralickData
{
  double          *distance;
  unsigned char ***gray;

  double         **min;
  double         **max;
  double         **sum;

  double         **out_buffer;
  double         **out;

  double        ***tone_matrix;

  double        ***buffer_matrix;
  double         **buffer_vector;
};

__global__ void cuda_haralick(CudaImages images, HaralickData data);
__device__ void get_intensity_range(pix_data *pixels, int pixel_count, double *min, double *max);
__device__ void normalize_to_8_bits(pix_data *image, int width, int height, int bits,
                                    double min, double max, unsigned char **gray);
__device__ inline void assign_feature(float feature, double *min, double *max, double *sum);

HaralickData                cuda_allocate_haralick_data(const std::vector<ImageMatrix *> &images);
std::vector<FileSignatures> cuda_get_haralick_signatures(const std::vector<ImageMatrix *> &images, HaralickData &data);
void                        cuda_delete_haralick_data(const std::vector<ImageMatrix *> &images, HaralickData &data);


#endif // _HARALICK_H
