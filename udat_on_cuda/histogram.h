//---------------------------------------------------------------------------

#ifndef histogramH
#define histogramH
//---------------------------------------------------------------------------

#include "image_matrix.h"
#include "device_launch_parameters.h"

__device__ void histogram(pix_data *data, double *bins, unsigned short bins_num, int imhist, int *width, int *height, int *depth, int *bits);
__device__ void multiscalehistogram(pix_data *data, double *out, int *width, int *height, int *depth, int *bits);

#endif