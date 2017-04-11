/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*                                                                               */
/*    This file is part of Cuda-Wndchrm.                                         */
/*    Copyright (C) 2017 Kamil Jarosz, Christopher K. Horton and Tyler Wiersing  */
/*                                                                               */
/*    This library is free software; you can redistribute it and/or              */
/*    modify it under the terms of the GNU Lesser General Public                 */
/*    License as published by the Free Software Foundation; either               */
/*    version 2.1 of the License, or (at your option) any later version.         */
/*                                                                               */
/*    This library is distributed in the hope that it will be useful,            */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of             */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU          */
/*    Lesser General Public License for more details.                            */
/*                                                                               */
/*    You should have received a copy of the GNU Lesser General Public           */
/*    License along with this library; if not, write to the Free Software        */
/*    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA  */
/*                                                                               */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

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
