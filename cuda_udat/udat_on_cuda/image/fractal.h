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