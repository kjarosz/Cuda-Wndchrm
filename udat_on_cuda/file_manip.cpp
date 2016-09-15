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



void join_paths(char *output, char *p1, char *p2)
{
  sprintf(output, "%s\\%s", p1, p2);
}



dirent dirent_data;



DIR * opendir(char *path)
{
  WIN32_FIND_DATA file_data;
  WCHAR w_path[MAX_PATH];
  HANDLE hDir;
  unsigned int char_index;
  for (char_index=0;char_index<strlen(path);char_index++)
   w_path[char_index]=(WCHAR)(path[char_index]);
  w_path[char_index]=(WCHAR)'\0';
  hDir=FindFirstFile(w_path,&file_data);
  FindClose(hDir);
  if ((file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != FILE_ATTRIBUTE_DIRECTORY) return(0);
  w_path[char_index]=(WCHAR)'\\';
  w_path[char_index+1]=(WCHAR)'*';
  w_path[char_index+2]=(WCHAR)'\0';
  hDir=FindFirstFile(w_path,&file_data);
  return((DIR *)hDir);
}



dirent *readdir(DIR *class_dir)
{
  WIN32_FIND_DATA file_data;
  int char_index=0;
  if (!FindNextFile((HANDLE)class_dir,&file_data)) return(NULL);
  while (file_data.cFileName[char_index])
  {  
    dirent_data.d_name[char_index] = (char)(file_data.cFileName[char_index]);
    char_index++;
  }
  dirent_data.d_name[char_index]='\0';  /* end the string */
  return(&dirent_data);
}



int closedir(DIR *dir)
{
  return FindClose((HANDLE)dir);
}


