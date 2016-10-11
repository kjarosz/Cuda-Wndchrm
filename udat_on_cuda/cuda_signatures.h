#ifndef CUDASIGNATURES_H
#define CUDASIGNATURES_H


#include <string>
#include <vector>

#include "image/image_matrix.h"

#include "signatures.h"
#include "constants.h"
#include "utils/DirectoryListing.h"


// 256 MB Batch Limit
const int BATCH_SIZE = 268435456;

const int MAXIMUM_CUDA_THREADS = 356;


// SECTION A - Main Interface
//------------------------------------------------------------------------------
std::vector<ClassSignatures> compute_signatures(char *root_dir, char **class_dirs, int class_count);
ClassSignatures              compute_class_signatures(std::string class_dir);
std::vector<FileSignatures>  compute_signatures_on_cuda(std::vector<ImageMatrix *> &matrices);

void                         save_signatures(std::vector<ClassSignatures> &class_signatures, char *root_dir);


// SECTION B.a - Auxiliary Types
//------------------------------------------------------------------------------
struct CudaImages
{
  int        count;

  pix_data **pixels;
  int       *widths;
  int       *heights;
  int       *depths;
  int       *bits;
};


// SECTION B.b - Auxiliary Functions
//------------------------------------------------------------------------------
bool          get_next_batch(DirectoryListing *listing, std::vector<ImageMatrix *> &images);
bool          batch_is_full(std::vector<ImageMatrix *> &images);

bool          supported_format(const char *filename);
ImageMatrix*  load_image_matrix(const char *filename);

void          move_images_to_cuda(std::vector<ImageMatrix *> &images, CudaImages &cuda_images);
void          delete_cuda_images(CudaImages &cuda_images);

std::vector<FileSignatures> &merge_signatures(std::vector<FileSignatures> &dst, std::vector<FileSignatures> &src);
int                          find_in_vector(std::vector<std::string> &vector, std::string value);

// SECTION C - Cuda Algorithms
//------------------------------------------------------------------------------
std::vector<FileSignatures> compute_zernike_on_cuda(const std::vector<ImageMatrix *> &images, CudaImages &cuda_images);
std::vector<FileSignatures> compute_haralick_on_cuda(const std::vector<ImageMatrix *> &images, CudaImages &cuda_images);
std::vector<FileSignatures> compute_histogram_on_cuda(const std::vector<ImageMatrix *> &images, CudaImages &cuda_images);
std::vector<FileSignatures> compute_fractals_on_cuda(const std::vector<ImageMatrix *> &images, CudaImages &cuda_images);
std::vector<FileSignatures> compute_chebyshev_on_cuda(const std::vector<ImageMatrix *> &images, CudaImages &cuda_images);



//// SECTION D.a - Cuda Algorithms Class
////------------------------------------------------------------------------------
//class CudaAlgorithm
//{
//public:
//  CudaAlgorithm(const std::vector<ImageMatrix *> *, const CudaImages *);
//  virtual ~CudaAlgorithm() {};
//
//  virtual void                        print_message()  const = 0;
//  virtual void                        compute()              = 0;
//  virtual std::vector<FileSignatures> get_signatures() const = 0;
//
//private:
//  CudaAlgorithm(const CudaAlgorithm &);
//  CudaAlgorithm &operator=(const CudaAlgorithm &);
//
//protected:
//  const std::vector<ImageMatrix *> *images;
//  const CudaImages                 *cuda_images;
//};
//
//// SECTION D.b - Cuda Algorithms template
////------------------------------------------------------------------------------
//template <class T>
//std::vector<FileSignatures> compute_features_on_cuda(const std::vector<ImageMatrix *> &images, const CudaImages &cuda_images)
//{
//  T computer(&images, &cuda_images);
//  computer.print_message();
//  computer.compute();
//
//  cudaError sync_status = cudaGetLastError();
//  cudaError async_status = cudaDeviceSynchronize();
//
//  if(sync_status == cudaSuccess && async_status == cudaSuccess) 
//  {
//    return computer.get_signatures();
//  } 
//  else 
//  {
//    if (sync_status != cudaSuccess)
//      print_cuda_error(sync_status, "Synchronous CUDA error occurred");
//
//    if (async_status != cudaSuccess)
//      print_cuda_error(async_status, "Asynchronous CUDA error occurred");
//
//    return std::vector<FileSignatures>();
//  }
//}

#endif // CUDASIGNATURES_H
