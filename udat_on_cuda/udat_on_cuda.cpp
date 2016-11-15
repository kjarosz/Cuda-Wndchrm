#include "udat_on_cuda.h"

#include "utils/file_manip.h"
#include "cuda_signatures.h"

int load_subdirectories(char *directory, char **&filenames, int &buffer_size)
{
  buffer_size = 64;
  filenames = alloc_filename_buffers(buffer_size);
  int file_idx = 0;

  char buffer[FILENAME_MAX];
  DIR * root_dir;
  if(root_dir = opendir(directory))
  {
    dirent *ent;
    while (ent = readdir(root_dir))
    {
      // Ignoring '.' and '..'.
      if (ent->d_name[0] == '.')
        continue;

      join_paths(buffer, directory, ent->d_name);

      // Ignore non-directories.
      DIR *sub_dir = opendir(buffer);
      if (!sub_dir)
        continue;
      closedir(sub_dir);

      strcpy_s(filenames[file_idx], FILENAME_MAX, ent->d_name);
      file_idx++;

      if (file_idx >= buffer_size)
      {
        extend_filename_buffers(filenames, buffer_size, buffer_size*2);
        buffer_size *= 2;
      }
    }
    closedir(root_dir);
  }

  return file_idx;
}

std::vector<ClassSignatures> compute(char *root)
{
  printf("Getting directories\n");

  char *directory = root;
  char **filenames;
  int filename_buffer_size;
  int dir_count = load_subdirectories(directory, filenames, filename_buffer_size);

  printf("Found %i directories\n", dir_count);
  
  printf("Computing signatures\n");
  std::vector<ClassSignatures> signatures = compute_signatures(directory, filenames, dir_count);

  printf("Saving signatures to file\n");
  save_signatures(signatures, directory); 

  printf("Done\n");

  free_filename_buffers(filenames, filename_buffer_size);

  return signatures;
}