#include <cstring>



#include "file_manip.h"
#include "signatures.h"
#include "cuda_runtime.h"



CUDASignatures::CUDASignatures()
{ }



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
  while(entry = read_next_entry())
  {
    if (entry->d_name[0] == '.')
      continue;

    if (!supported_format(ent->d_name))
      continue;
  }
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