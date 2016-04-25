#include "file_manip.h"



#include <cstdio>
#include <cstring>



char **alloc_filename_buffers(int size)
{
  char **buffers = new char*[size];
  for(int i = 0; i < size; i++)
    buffers[i] = new char[FILENAME_MAX];
  return buffers;
}



bool extend_filename_buffers(char **&old_buffers, int old_size, int new_size)
{
  if (old_size > new_size)
  {
    printf("extend_filename_buffers:\n");
    printf("New filename buffer length has to be longer than the old.\n");
    return false;
  }

  char **buffers = alloc_filename_buffers(new_size);
  for(int i = 0; i < old_size; i++)
  {
    strcpy(buffers[i], old_buffers[i]);
  }
  free_filename_buffers(old_buffers, old_size);
  old_buffers = buffers;
  return true;
}



void free_filename_buffers(char **&filenames, int count)
{
  for(int i = 0; i < count; i++)
  {
    delete [] filenames[i];
  }
  delete [] filenames;
  filenames = 0;
}


