//---------------------------------------------------------------------------

#ifndef zernikeH
#define zernikeH
//---------------------------------------------------------------------------

#include "../../cuda_signatures.h"

struct ZernikeData
{
  double *D;
  double *R;
  double **Y;
  double **X;
  double **P;
  double **xcoords;

  double **zvalues;
  double *output_size;
};



__global__ void mb_zernike2D(CudaImages images, ZernikeData data);



ZernikeData                 cuda_allocate_zernike_data(const CudaImages &images);
std::vector<FileSignatures> cuda_get_zernike_signatures(const ZernikeData &data, int image_count);
void                        cuda_delete_zernike_data(ZernikeData &data, int image_count);



#endif
