#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <cuda_runtime.h>

std::string print_windows_error();
void        print_cuda_error(cudaError error, std::string message);

#endif // UTILS_H