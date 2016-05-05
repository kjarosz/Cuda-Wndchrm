#ifndef SIGNATURES_H
#define SIGNATURES_H



#include <string>
#include <vector>
#include <map>



#include "image_matrix.h"
#include "file_manip.h"



const int INIT_MATRIX_CONTAINER_SIZE = 16;
const int INIT_SIG_CONTAINER_SIZE    = 64;

const int SIGNATURE_NAME_LENGTH      = 80;

const int MAX_OUTPUT_SIZE            = 72;



struct DirectoryTracker
{
  char **directories;
  int count;
  int current_dir;
  char *directory;
  DIR *opened_dir;
};



class Signatures
{
public:
  Signatures();
  ~Signatures();

  void   add_signature(const char *sig_name, const char *filename, double value);

  double get_signature(const char *sig_name, const char *filename) const;
  double get_signature(const char *sig_name, int         row     ) const;
  double get_signature(int         col,      const char *filename) const;
  double get_signature(int         col,      int         row     ) const;

  void   get_sig_name(int col, char *output);
  void   get_col_name(int row, char *output);

  int    get_signature_index(const char *name) const;
  int    get_filename_index (const char *name) const;

  int    get_row_count() const;
  int    get_col_count() const;

  void   clear();

  std::vector<std::string> get_sig_names() const;
  std::vector<std::string> get_filenames() const;

private:
  Signatures(const Signatures &other);
  Signatures &operator=(const Signatures &other);

  int find_in_array(char **arr, int len, const char *element) const;

  int insert_new_signature(const char *name);
  int insert_new_filename (const char *name);

  void expand_signature_container();
  void expand_filename_container();
  void expand_value_array(const int new_col_len, const int new_row_len);

  inline std::vector<std::string> get_array_copy(char **arr, int len) const;

private:

  // Size of the containers.
  int col_len;
  int row_len;

  // Number of actual values in the containers.
  int col_n;
  int row_n;

  char **sigs;    // Columns
  char **files;   // Rows
  double *values; // Data matrix
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
  dirent *read_next_entry();

  void load_image_matrix(char *filename);
  void double_matrix_container();
  void empty_matrix_container();
  
  void compute_signatures_on_cuda();

  // All CUDA functions.
  void compute_zernike_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long *sizes);
  
  void allocate_signature_buffers();
  void add_signatures(double *output, long size);
  void deallocate_signature_buffers();

private:
  DirectoryTracker directory_tracker;
  
  ImageMatrix **image_matrices;
  int matrix_container_size;
  int image_matrix_count;

  Signatures signatures;
};



#endif // SIGNATURES_H