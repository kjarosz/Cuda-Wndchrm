#ifndef SIGNATURES_H
#define SIGNATURES_H



#include "image_matrix.h"
#include "file_manip.h"



const int INIT_MATRIX_CONTAINER_SIZE = 16;
const int MAX_OUTPUT_SIZE            = 72;



struct DirectoryTracker
{
  char **directories;
  int count;
  int current_dir;
  char *directory;
  DIR *opened_dir;
};



class CUDASignatures
{
public:
  CUDASignatures();
  ~CUDASignatures();

  void compute(char **directories, int count);

  void save_in(char *directory);



private:
  CUDASignatures(const CUDASignatures &source);
  CUDASignatures &operator=(const CUDASignatures &other);



private:
  bool supported_format(char *filename);

  void reset_directory_tracker(char **directories, int count);
  bool read_next_batch();
  bool batch_capacity_reached();
  bool batch_ready_to_compute();
  dirent *read_next_entry();

  void load_image_matrix(char *filename);
  void double_matrix_container();
  void empty_matrix_container();
  
  void compute_signatures_on_cuda();

  // All CUDA functions.
  void compute_zernike_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long *sizes);
  void compute_haarlick_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long*sizes);
  
private:
  DirectoryTracker directory_tracker;
  
  ImageMatrix **image_matrices;
  int matrix_container_size;
  int image_matrix_count;
};



#endif // SIGNATURES_H