#ifndef FILE_MANIP_H
#define FILE_MANIP_H



#include <Windows.h>



#define DIR HANDLE
struct dirent
{
  long           d_ino;
  long           d_off;
  unsigned short d_reclen;
  char           d_name[256];
};



char  **alloc_filename_buffers (int size);
bool    extend_filename_buffers(char **&old_buffers, int old_size, int new_size);
void    free_filename_buffers  (char **&filenames, int count);



void    join_paths(char *output, char *p1, char *p2);



DIR    *opendir (char *path);
dirent *readdir (DIR *class_dir);
int     closedir(DIR *dir);



#endif // FILE_MANIP_H