#ifndef ZERNIKE_H
#define ZERNIKE_H



#include <vector>

#include "device_launch_parameters.h"

#include "../../cuda_signatures.h"
#include "../../image/image_matrix.h"



struct ZernikeData
{
  double *D;
  double *R;
  double **Y;
  double **X;
  double **P;

  double **zvalues;
  long   *output_size;
};



__global__ void cuda_zernike(CudaImages images, ZernikeData data);
__device__ void mb_Znl(long n, long l, double *X, double *Y, double *P, int size, double *out_r, double *out_i);
__device__ double mb_imgmoments(pix_data *pixels, int width, int height, int x, int y);
__device__ double factorial(double n);


long                        cuda_zernike_mem_estimate(ImageMatrix *image);
ZernikeData                 cuda_allocate_zernike_data(const std::vector<ImageMatrix *> &images);
std::vector<FileSignatures> cuda_get_zernike_signatures(const std::vector<ImageMatrix *> &images,
                                                        const ZernikeData &data, int image_count);
void                        cuda_delete_zernike_data(ZernikeData &data, int image_count);



#endif // ZERNIKE_H
