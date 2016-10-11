//---------------------------------------------------------------------------

#ifndef histogramH
#define histogramH
//---------------------------------------------------------------------------

#include <vector>

#include "cuda_signatures.h"
#include "image_matrix.h"
#include "device_launch_parameters.h"

const unsigned int HISTOGRAM_BIN_COUNT = 24;

struct HistogramData 
{
  double **out;
};

__global__ void cuda_multiscalehistogram(CudaImages images, HistogramData data); 
__device__ void histogram(pix_data *data, int width, int height, int depth, 
                          int bits, double *bins, unsigned short bins_num, int imhist);

HistogramData               cuda_allocate_histogram_data(const std::vector<ImageMatrix *> &images);
std::vector<FileSignatures> cuda_get_histogram_signatures(const std::vector<ImageMatrix *> &images, HistogramData &data);
void                        cuda_delete_histogram_data(const std::vector<ImageMatrix *> &images, HistogramData &data);

#endif