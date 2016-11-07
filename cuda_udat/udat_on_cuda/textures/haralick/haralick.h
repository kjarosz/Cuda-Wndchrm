//---------------------------------------------------------------------------

#ifndef _HARALICK_H
#define _HARALICK_H
//---------------------------------------------------------------------------

#include "../../cuda_signatures.h"
#include "../../image/image_matrix.h"
#include "device_launch_parameters.h"

const unsigned int HARALICK_FEATURE_SIZE               = 14;
const unsigned int HARALICK_OUT_SIZE                   = 28;
const unsigned int HARALICK_TONE_MAX                   = 255;
const unsigned int HARALICK_BUF_VEC_COUNT              = 4;

struct HaralickData
{
  double          *distance;
  unsigned char ***gray;

  double         **min;
  double         **max;
  double         **sum;

  double         **out_buffer;
  double         **out;

  double        ***tone_matrix;

  double        ***buffer_matrix;
  double         **buffer_vector;
};

__global__ void cuda_haralick(CudaImages images, HaralickData data);
__device__ void get_intensity_range(pix_data *pixels, int pixel_count, double *min, double *max);
__device__ void normalize_to_8_bits(pix_data *image, int width, int height, int bits, 
                                    double min, double max, unsigned char **gray);
__device__ inline void assign_feature(float feature, double *min, double *max, double *sum);

HaralickData                cuda_allocate_haralick_data(const std::vector<ImageMatrix *> &images);
std::vector<FileSignatures> cuda_get_haralick_signatures(const std::vector<ImageMatrix *> &images, HaralickData &data);
void                        cuda_delete_haralick_data(const std::vector<ImageMatrix *> &images, HaralickData &data);


#endif // _HARALICK_H
