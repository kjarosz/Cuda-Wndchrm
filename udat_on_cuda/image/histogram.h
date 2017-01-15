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

#ifndef histogramH
#define histogramH
//---------------------------------------------------------------------------

#include <vector>

#include "../cuda_signatures.h"
#include "image_matrix.h"
#include "device_launch_parameters.h"

const unsigned int HISTOGRAM_BIN_COUNT = 24;

struct HistogramData
{
  double **out;
};

__global__ void cuda_multiscalehistogram(CudaImages images, HistogramData data);
__device__ void histogram(pix_data *data, int width, int height, int depth,
                          int bits, double *bins, unsigned short bins_num, int imhist);

HistogramData               cuda_allocate_histogram_data(const std::vector<ImageMatrix *> &images);
std::vector<FileSignatures> cuda_get_histogram_signatures(const std::vector<ImageMatrix *> &images, HistogramData &data);
void                        cuda_delete_histogram_data(const std::vector<ImageMatrix *> &images, HistogramData &data);

#endif
