//---------------------------------------------------------------------------

#ifndef haarlickH
#define haarlickH
//---------------------------------------------------------------------------

#include "image_matrix.h"

void CUDA_haarlick2d(ImageMatrix *Im, double distance, double *out);
void allocate_haarlick_memory(ImageMatrix *Im, double distance, double *out);
#endif
