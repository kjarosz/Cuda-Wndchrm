#ifndef CHEBYSHEV_H
#define CHEBYSHEV_H


#include <vector>

#include "../cuda_signatures.h"
#include "../image/image_matrix.h"
#include "device_launch_parameters.h"

const int CHEBYSHEV_COEFF_COUNT = 20; // TODO CHAAANGE!!!!

struct ChebyshevData
{
  double **x;
  double **y;
  double **in;
  double **out;
  int     *N;

  // Buffers
  double **c_y;     // buffer for coeff y
  double **c_y_out; // buffer for coeff y_out
  double **c_Tj;    // buffer for coeff_1D Tj
  double **c_tj;    // buffer for coeff_1D tj
  double **tnx1;    // buffer for TNx temp
  double **tnx2;    // buffer for TNx temp1
};



__global__ void cuda_chebyshev(CudaImages, ChebyshevData);
__device__ void cuda_chebyshev_coefficients(double *, double *, double *, int, int, int, double *, double *, double *, double *, double *, double *);
__device__ void cuda_chebyshev_coefficients_1D(double *, double *, double *, int, int, double *, double *, double *, double *);
__device__ void cuda_TNx(double *, double *, int, int, double *, double *);



//long                        cuda_chebyshev_mem_estimate(ImageMatrix *image);
ChebyshevData               cuda_allocate_chebyshev_data(const std::vector<ImageMatrix *> &images);
std::vector<FileSignatures> cuda_get_chebyshev_signatures(const std::vector<ImageMatrix *> &images,
                                                        const ChebyshevData &data);
void                        cuda_delete_chebyshev_data(const std::vector<ImageMatrix *> &images,
                                                       ChebyshevData &data);



#endif // CHEBYSHEV_H