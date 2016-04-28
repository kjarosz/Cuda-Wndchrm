//---------------------------------------------------------------------------

#ifndef haarlickH
#define haarlickH
//---------------------------------------------------------------------------

#include "image_matrix.h"
#include "device_launch_parameters.h"

__global__ void CUDA_haarlick2d(ImageMatrix *Im, double distance, double *out);
void allocate_haarlick_memory(ImageMatrix *Im, double distance, double *out);
__device__ void BasicStatistics(pix_data *data, double *min, double *max, int bins, int num_pixels);
#endif
