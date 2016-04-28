//---------------------------------------------------------------------------

#ifndef haarlickH
#define haarlickH
//---------------------------------------------------------------------------

#include "image_matrix.h"
#include "device_launch_parameters.h"

__global__ void haarlick(pix_data *pixels, double *distance, double *out, int *height, int *width, int *depth, unsigned short int *bits);
void allocate_haarlick_memory(ImageMatrix *Im, double distance, double *out);
__device__ void BasicStatistics(pix_data *data, double *min, double *max, int bins, int num_pixels);
#endif
