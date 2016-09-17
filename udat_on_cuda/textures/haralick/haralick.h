//---------------------------------------------------------------------------

#ifndef _HARALICK_H
#define _HARALICK_H
//---------------------------------------------------------------------------

#include "../../cuda_signatures.h"
#include "../../image_matrix.h"
#include "device_launch_parameters.h"

const unsigned int HARALICK_FEATURE_SIZE               = 14;
const unsigned int HARALICK_OUT_SIZE                   = 28;

/* Order in which the features go into the output array. */
const unsigned int HARALICK_OUT_MAP[HARALICK_OUT_SIZE] = {
  0,  14, //  (1) Angular Second Moment
  1,  15, //  (2) Contrast
  2,  16, //  (3) Correlation
  9,  23, // (10) Difference Variance
  10, 24, // (11) Difference Entropy
  8,  22, //  (9) Entropy
  11, 25, // (12) Measure of Correlation 1
  4,  18, //  (5) Inverse Difference Moment
  13, 27, // (14) Maximal Correlation Coefficient
  12, 26, // (13) Measure of Correlation 2
  5,  19, //  (6) Sum Average
  7,  21, //  (8) Sum Entropy
  6,  20, //  (7) Sum Variance
  3,  17  //  (4) Variance
};

struct HaralickData
{
  double          *distance;
  unsigned char ***gray;

  double         **min;
  double         **max;
  double         **sum;

  double         **out_buffer;
  double         **out;
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
