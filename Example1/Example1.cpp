#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <vector>

#include "udat_on_cuda.h"



void print_help()
{
  printf("Usage:\n");
  printf("<program> <directory>\n");
}



int main(int argc, char *argv[])
{
  if (argc < 2) {
    print_help();
    system("pause");
    return -1;
  }

  compute(argv[1]);
  system("pause");

  return 0;
}