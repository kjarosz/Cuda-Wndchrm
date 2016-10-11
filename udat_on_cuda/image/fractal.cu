#include "fractal.h"
#include "../utils/cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>


__global__ void cuda_fractal(CudaImages images, FractalData data)
{  
  int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int bins = data.bin_count[th_idx];

  pix_data *pixels = images.pixels[th_idx];
  int width        = images.widths[th_idx];
  int height       = images.heights[th_idx];

  int K = min(width, height) / 5;

  int step = (long)floor((double)(K/bins));
  if (step<1) 
    step=1;   /* avoid an infinite loop if the image is small */

  int bin = 0;
  for (int k = 1; k < K; k += step)
  {  
    double sum=0.0;
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height - k; y++) {
        pix_data pixel  = get_pixel(pixels, width, height, x, y, 0);
        pix_data pixel2 = get_pixel(pixels, width, height, x, y+k, 0);
        sum+=fabs(pixel.intensity - pixel2.intensity);
      }
    }

    for (int x = 0; x < width - k; x++) {
      for (int y = 0; y < height; y++) {
        pix_data pixel  = get_pixel(pixels, width, height, x, y, 0);
        pix_data pixel2 = get_pixel(pixels, width, height, x+k, y, 0);
        sum+=fabs(pixel.intensity - pixel2.intensity);
      }
    }

    if (bin < bins) 
      data.output[th_idx][bin++] = sum / (width*(width-k)+height*(height-k));	  
  }
}



FractalData cuda_allocate_fractal_data(const std::vector<ImageMatrix *> &images)
{
  FractalData data;
  memset(&data, 0, sizeof( FractalData ));

  int *bins = new int[images.size()];
  for(int i = 0; i < images.size(); i++)
    bins[i] = FRACTAL_BIN_COUNT;

  cudaMalloc(&data.bin_count, images.size() * sizeof(int));
  cudaMemcpy(data.bin_count, bins, images.size() * sizeof(int), cudaMemcpyHostToDevice);
  delete [] bins;

  cuda_alloc_cube_array(FRACTAL_BIN_COUNT, images.size(), data.output);

  return data;
}



std::vector<FileSignatures> cuda_get_fractal_signatures(const std::vector<ImageMatrix *> &images, FractalData &data)
{
  double **outputs = new double*[images.size()];
  cudaError status = cudaMemcpy(outputs, data.output, images.size() * sizeof(double *), cudaMemcpyDeviceToHost);
  for(unsigned int i = 0; i < images.size(); i++)
  {
    double *out = new double[FRACTAL_BIN_COUNT];
    status = cudaMemcpy(out, outputs[i], FRACTAL_BIN_COUNT * sizeof(double), cudaMemcpyDeviceToHost);
    outputs[i] = out;
  }

  std::vector<FileSignatures> file_signatures;
  for(int i = 0; i < images.size(); i++)
  {
    FileSignatures file_signature;
    file_signature.file_name = images[i]->source_file;

    for(int j = 0; j < FRACTAL_BIN_COUNT; j++)
    {
      std::stringstream ss;
      ss << "Fractal " << j;

      Signature signature;
      signature.signature_name = ss.str();
      signature.value = outputs[i][j];

      file_signature.signatures.push_back(signature);
    }
    file_signatures.push_back(file_signature);
  }

  for(unsigned int i = 0; i < images.size(); i++)
    delete [] outputs[i];
  delete [] outputs;

  return file_signatures;
}



void cuda_delete_fractal_data(const std::vector<ImageMatrix *> &images, FractalData &data)
{
  cuda_free_multidim_arr<double>(data.output, images.size());
}
