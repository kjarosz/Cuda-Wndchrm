//---------------------------------------------------------------------------

#ifndef cuda_zernikeH
#define cuda_zernikeH
//---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "../../cmatrix.h"

__global__ static void cuda_mb_zernike2D(ImageMatrix *I, double D, double R, double *zvalues, long *output_size);

#endif // cuda_zernikeH
