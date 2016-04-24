//---------------------------------------------------------------------------

#ifndef haarlickH
#define haarlickH
//---------------------------------------------------------------------------

#include "cmatrix.h"

void haarlick2D(ImageMatrix *Im, double distance, double *out);
void CUDA_haarlick2d(ImageMatrix *Im, double distance, double *out);

#endif
