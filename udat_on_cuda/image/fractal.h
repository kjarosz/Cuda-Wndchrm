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

#ifndef FRACTAL_H
#define FRACTAL_H

#include "../cuda_signatures.h"
#include "image_matrix.h"
#include "device_launch_parameters.h"

const unsigned int FRACTAL_BIN_COUNT = 20;

struct FractalData
{
  double **output;
  int     *bin_count;
};

__global__ void cuda_fractal(CudaImages images, FractalData data);

FractalData                 cuda_allocate_fractal_data(const std::vector<ImageMatrix *> &images);
std::vector<FileSignatures> cuda_get_fractal_signatures(const std::vector<ImageMatrix *> &images, FractalData &data);
void                        cuda_delete_fractal_data(const std::vector<ImageMatrix *> &images, FractalData &data);


#endif // FRACTAL_H
