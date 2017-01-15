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

#include <math.h>
#include <stdio.h>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "image_matrix.h"
#include "histogram.h"
#include "../utils/cuda_utils.h"



__global__ void cuda_multiscalehistogram(CudaImages images, HistogramData data)
{
  const int th_idx = blockIdx.x * blockDim.x + threadIdx.x;

  pix_data *image  = images.pixels[th_idx];
  int       width  = images.widths[th_idx];
  int       height = images.heights[th_idx];
  int       depth  = images.depths[th_idx];
  int       bits   = images.bits[th_idx];
  double   *out    = data.out[th_idx];

	histogram(image, width, height, depth, bits, out, 3, 0);
	histogram(image, width, height, depth, bits, &(out[3]), 5, 0);
	histogram(image, width, height, depth, bits, &(out[8]), 7, 0);
	histogram(image, width, height, depth, bits, &(out[15]), 9, 0);

	double max = 0;
	for (int a = 0; a < HISTOGRAM_BIN_COUNT; a++)
    if (out[a]>max)
      max = out[a];

	for (int a = 0; a < HISTOGRAM_BIN_COUNT; a++)
		out[a] = out[a] / max;
}



__device__ void histogram(pix_data *data, int width, int height, int depth, int bits, double *bins, unsigned short bins_num, int imhist)
{
	double min = INF, max = -INF;

	/* find the minimum and maximum */
	if (imhist == 1)    /* similar to the Matlab imhist */
	{
		min = 0;
		max = pow(2.0, (double)bits) - 1;
	}
	else
	{
		for (long a = 0; a < width*height*depth; a++)
		{
			if (data[a].intensity>max)
				max = data[a].intensity;
			if (data[a].intensity<min)
				min = data[a].intensity;
		}
	}
	/* initialize the bins */
	for (long a = 0; a<bins_num; a++)
		bins[a] = 0;

	/* build the histogram */
  long pix_idx;
	for (long a = 0; a < width*height*depth; a++)
  {
	  if (data[a].intensity == max) {
      pix_idx = bins_num - 1;
    } else {
      pix_idx = (long)(((data[a].intensity - min) / (max - min))*bins_num);
    }
    bins[pix_idx] += 1;
  }
}



HistogramData cuda_allocate_histogram_data(const std::vector<ImageMatrix *> &images)
{
  HistogramData data;
  memset(&data, 0, sizeof(HistogramData));
  cuda_alloc_cube_array<double>(HISTOGRAM_BIN_COUNT, images.size(), data.out);
  return data;
}



std::vector<FileSignatures> cuda_get_histogram_signatures(const std::vector<ImageMatrix *> &images, HistogramData &data)
{
  double **outputs = new double*[images.size()];
  cudaError status = cudaMemcpy(outputs, data.out, images.size() * sizeof(double *), cudaMemcpyDeviceToHost);
  for(unsigned int i = 0; i < images.size(); i++)
  {
    double *out = new double[HISTOGRAM_BIN_COUNT];
    status = cudaMemcpy(out, outputs[i], HISTOGRAM_BIN_COUNT * sizeof(double), cudaMemcpyDeviceToHost);
    outputs[i] = out;
  }

  std::vector<FileSignatures> file_signatures;
  for(int i = 0; i < images.size(); i++)
  {
    FileSignatures file_signature;
    file_signature.file_name = images[i]->source_file;

    for(int j = 0; j < HISTOGRAM_BIN_COUNT; j++)
    {
      std::stringstream ss;
      ss << "MultiScale Histogram bin " << j;

      Signature signature;
      signature.signature_name = ss.str();
      signature.value = outputs[i][j];

      file_signature.signatures.push_back(signature);
    }
    file_signatures.push_back(file_signature);
  }

  for(unsigned int i = 0; i < images.size(); i++)
    delete [] outputs[i];
  delete [] outputs;

  return file_signatures;
}



void cuda_delete_histogram_data(const std::vector<ImageMatrix *> &images, HistogramData &data)
{
  cuda_free_multidim_arr<double>(data.out, images.size());
}
