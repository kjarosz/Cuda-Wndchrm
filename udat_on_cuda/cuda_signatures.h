#ifndef CUDASIGNATURES_H
#define CUDASIGNATURES_H


#include <string>
#include <vector>

#include "image_matrix.h"
#include "signatures.h"
#include "constants.h"
#include "utils/DirectoryListing.h"

// 1 GB Batch limit
#define  BATCH_SIZE  1073741824



std::vector<ClassSignatures> compute(char *root_dir, char **class_dirs, int class_count);
std::vector<Signature>       compute_class_signatures(std::string class_dir);
std::vector<Signature>       compute_signatures_on_cuda(std::vector<ImageMatrix *> &matrices);

void                         save_signatures(std::vector<ClassSignatures> &class_signatures, char *root_dir);


bool                         get_next_batch(DirectoryListing *listing, std::vector<ImageMatrix *> &images);
bool                         batch_is_full(std::vector<ImageMatrix *> &images);

bool                         supported_format(const char *filename);
ImageMatrix*                 load_image_matrix(const char *filename);


class CUDASignatures
{
  // All CUDA functions.
  void compute_zernike_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long *sizes);
  void compute_haralick_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long *sizes, int *bits);
  void compute_histogram_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long *sizes, int *bits);
};



#endif // CUDASIGNATURES_H
