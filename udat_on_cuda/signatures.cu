#include <cstring>
#include <cstdio>



#include "file_manip.h"
#include "signatures.h"
#include "cuda_runtime.h"



CUDASignatures::CUDASignatures()
  :image_matrices(0), 
   matrix_container_size(INIT_MATRIX_CONTAINER_SIZE),
   image_matrix_count(0)
{ 
  image_matrices = new ImageMatrix*[matrix_container_size];
}



CUDASignatures::~CUDASignatures()
{
  empty_matrix_container();
  delete [] image_matrices;
}



void CUDASignatures::compute(char **directories, int count)
{
  reset_directory_tracker(directories, count);
  while(read_next_batch())
    compute_images_on_cuda();
}



bool CUDASignatures::supported_format(char *filename)
{
  if (strstr(filename, ".tif") || strstr(filename, ".TIF"))
    return true;
  return false;
}



void CUDASignatures::reset_directory_tracker(char **directories, int count)
{
  directory_tracker.directories = directories;
  directory_tracker.current_dir = 0;
  directory_tracker.count       = count;
  directory_tracker.opened_dir  = 0;
}



bool CUDASignatures::read_next_batch()
{
  dirent *entry;
  char filename_buffer[FILENAME_MAX];
  while((entry = read_next_entry()) && !batch_capacity_reached())
  {
    if (entry->d_name[0] == '.')
      continue;

    if (!supported_format(entry->d_name))
      continue;

    join_paths(filename_buffer, directory_tracker.directory, entry->d_name);

    load_image_matrix(filename_buffer);
  }

  return batch_ready_to_compute();
}



bool CUDASignatures::batch_capacity_reached()
{
}



bool CUDASignatures::batch_ready_to_compute()
{
}



dirent * CUDASignatures::read_next_entry()
{
  dirent *entry = 0;

  bool done = false;
  while(!done && directory_tracker.current_dir < directory_tracker.count)
  { 
    if(!directory_tracker.opened_dir)
    {
      directory_tracker.opened_dir = opendir(
        directory_tracker.directories[directory_tracker.current_dir]);
    }

    entry = readdir(directory_tracker.opened_dir);
    if (!entry)
    {
      done = true;
    } 
    else
    {
      closedir(directory_tracker.opened_dir);
      directory_tracker.current_dir++;
    }
  }

  return entry;
}



void CUDASignatures::load_image_matrix(char *filename)
{
  if(image_matrix_count >= matrix_container_size)
    double_matrix_container();

  ImageMatrix *matrix;

  // TODO: Load Image matrix;

  image_matrices[image_matrix_count++] = matrix;
}



void CUDASignatures::double_matrix_container()
{
  ImageMatrix **new_container = new ImageMatrix*[matrix_container_size * 2];
  memcpy(new_container, image_matrices, matrix_container_size * sizeof(ImageMatrix*));
  delete [] image_matrices;
  image_matrices = new_container;
  matrix_container_size *= 2;
}



void CUDASignatures::empty_matrix_container()
{
  for(int i = 0; i < image_matrix_count; i++) 
    delete image_matrices[i];
  image_matrix_count = 0;
}



void CUDASignatures::compute_images_on_cuda()
{

}


