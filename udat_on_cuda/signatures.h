#ifndef SIGNATURES_H
#define SIGNATURES_H



#include "image_matrix.h"
#include "file_manip.h"



struct DirectoryTracker
{
  char **directories;
  int count;
  int current_dir;
  DIR *opened_dir;
};



class CUDASignatures
{
public:
  CUDASignatures();

  void compute(char **directories, int count);

  void save_in(char *directory);



private:
  bool supported_format(char *filename);

  void reset_directory_tracker(char **directories, int count);
  bool read_next_batch();
  dirent *read_next_entry();

  void compute_images_on_cuda();
  
  
private:
  DirectoryTracker directory_tracker;
  


};



#endif // SIGNATURES_H