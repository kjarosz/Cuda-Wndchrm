#include <cstring>
#include <cstdio>



#include "file_manip.h"
#include "textures/zernike/zernike.h"
#include "haarlick.h"
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
    compute_signatures_on_cuda();
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
  return true;
}



bool CUDASignatures::batch_ready_to_compute()
{
  return true;
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

  ImageMatrix *matrix = new ImageMatrix();
  matrix->OpenImage(filename);
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



void CUDASignatures::compute_signatures_on_cuda()
{
  // Arrange data in RAM
  pix_data **pixels  = new pix_data*[image_matrix_count];
  int *widths  = new int[image_matrix_count];
  int *heights = new int[image_matrix_count];
  int *depths  = new int[image_matrix_count];

  for(int i = 0; i < image_matrix_count; i++)
  {
    widths[i]  = image_matrices[i]->width;
    heights[i] = image_matrices[i]->height;
    depths[i]  = image_matrices[i]->depth;

    int size = widths[i] * heights[i] * depths[i];
    pix_data *pixel_array;
    cudaMalloc(&pixel_array, size * sizeof(pix_data));
    cudaMemcpy(pixel_array, image_matrices[i]->pixel, size * sizeof(pix_data), cudaMemcpyHostToDevice);
    pixels[i] = pixel_array;
  }

  // Move data from RAM to VRAM
  pix_data **cPixels = 0; 
  int *cWidths = 0, *cHeights = 0, *cDepths = 0;

  cudaMalloc(&cPixels,  image_matrix_count * sizeof(pix_data*));
  cudaMalloc(&cWidths,  image_matrix_count * sizeof(int));
  cudaMalloc(&cHeights, image_matrix_count * sizeof(int));
  cudaMalloc(&cDepths,  image_matrix_count * sizeof(int));

  cudaMemcpy(cWidths,  widths,  image_matrix_count * sizeof(int),       cudaMemcpyHostToDevice);
  cudaMemcpy(cHeights, heights, image_matrix_count * sizeof(int),       cudaMemcpyHostToDevice);
  cudaMemcpy(cDepths,  depths,  image_matrix_count * sizeof(int),       cudaMemcpyHostToDevice);
  cudaMemcpy(cPixels,  pixels,  image_matrix_count * sizeof(pix_data*), cudaMemcpyHostToDevice);

  double *cOutputs = 0;
  cudaMalloc(&cOutputs, MAX_OUTPUT_SIZE * image_matrix_count * sizeof(double));

  long *cSizes = 0;
  cudaMalloc(&cSizes, image_matrix_count * sizeof(long));

  // Execute the features.
  compute_zernike_on_cuda(cPixels, cWidths, cHeights, cDepths, cOutputs, cSizes);
  compute_haarlick_on_cuda(cPixels, cWidths, cHeights, cDepths, cOutputs, cSizes);

  cudaFree(cSizes);
  cudaFree(cOutputs);

  for(int i = 0; i < image_matrix_count; i++)
  {
    cudaFree(pixels[i]);
  }
  cudaFree(cPixels);
  cudaFree(cDepths);
  cudaFree(cHeights);
  cudaFree(cWidths);

  delete [] pixels;
  delete [] depths;
  delete [] heights;
  delete [] widths;
}



void CUDASignatures::compute_zernike_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long *sizes)
{

  double *d;
  double *r;

  cudaMalloc(&d, image_matrix_count * sizeof(double));
  cudaMalloc(&r, image_matrix_count * sizeof(double));

  cudaMemset(d, 0, image_matrix_count * sizeof(double));
  cudaMemset(r, 0, image_matrix_count * sizeof(double));

  zernike<<< 1, image_matrix_count >>>(images, widths, heights, depths, image_matrix_count, 
                                       d, r, outputs, sizes);

  cudaFree(r);
  cudaFree(d);
}

void CUDASignatures::compute_haarlick_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long*sizes)
{
	__device__ const int cDistances = 0;

	haarlick<<< 1, image_matrix_count >>>(images, cDistances, outputs, heights, widths, depths, bits);

	cudaFree(cDistances);
}

