//---------------------------------------------------------------------------

#pragma hdrstop

#include "haralick.h"
#include "../../CVIPtexture.h"
#include "../../image_matrix.h"
#include "../../utils/cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"



__global__ void haralick(CudaImages images, HaralickData data) 
// pix_data *pixels, double *distance, double *out, int *height, int *width, int *depth, unsigned short int *bits) 
{
	const int th_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (data.distance[th_idx] <= 0) 
    data.distance[th_idx] = 1;

  unsigned int pixel_count = images.heights[th_idx] * images.widths[th_idx] * images.depths[th_idx];

	double min_value = INF;
  double max_value = -INF; 
	get_intensity_range(images.pixels[th_idx], pixel_count, &min_value, &max_value);
  normalize_to_8_bits(images.pixels[th_idx], images.widths[th_idx], images.heights[th_idx],
                      images.bits[th_idx], min_value, max_value, data.gray[th_idx]);

	for (int a = 0; a < HARALICK_FEATURE_SIZE; a++)
	{
		data.min[th_idx][a] = INF;
		data.max[th_idx][a] = -INF;
		data.sum[th_idx][a] = 0;
	}

	for (long angle = 0; angle <= 135; angle = angle + 45)
	{
    TEXTURE features;
		Extract_Texture_Features(&features, (int)data.distance[th_idx], angle, data.gray[th_idx], 
                             images.heights[th_idx], images.widths[th_idx], (int)max_value);

		/*  (1) Angular Second Moment */
    assign_feature(features.ASM,           &data.min[th_idx][0], &data.max[th_idx][0], &data.sum[th_idx][0]);

		/*  (2) Contrast */
    assign_feature(features.contrast,      &data.min[th_idx][1], &data.max[th_idx][1], &data.sum[th_idx][1]);

		/*  (3) Correlation */
    assign_feature(features.correlation,   &data.min[th_idx][2], &data.max[th_idx][2], &data.sum[th_idx][2]);

		/*  (4) Variance */
    assign_feature(features.variance,      &data.min[th_idx][3], &data.max[th_idx][3], &data.sum[th_idx][3]);

		/*  (5) Inverse Diffenence Moment */
    assign_feature(features.IDM,           &data.min[th_idx][4], &data.max[th_idx][4], &data.sum[th_idx][4]);

		/*  (6) Sum Average */
    assign_feature(features.sum_avg,       &data.min[th_idx][5], &data.max[th_idx][5], &data.sum[th_idx][5]);

		/*  (7) Sum Variance */
    assign_feature(features.sum_var,       &data.min[th_idx][6], &data.max[th_idx][6], &data.sum[th_idx][6]);

		/*  (8) Sum Entropy */
    assign_feature(features.sum_entropy,   &data.min[th_idx][7], &data.max[th_idx][7], &data.sum[th_idx][7]);

		/*  (9) Entropy */
    assign_feature(features.entropy,       &data.min[th_idx][8], &data.max[th_idx][8], &data.sum[th_idx][8]);

		/* (10) Difference Variance */
    assign_feature(features.diff_var,      &data.min[th_idx][9], &data.max[th_idx][9], &data.sum[th_idx][9]);

		/* (11) Diffenence Entropy */
    assign_feature(features.diff_entropy,  &data.min[th_idx][10], &data.max[th_idx][10], &data.sum[th_idx][10]);

		/* (12) Measure of Correlation 1 */
    assign_feature(features.meas_corr1,    &data.min[th_idx][11], &data.max[th_idx][11], &data.sum[th_idx][11]);

		/* (13) Measure of Correlation 2 */
    assign_feature(features.meas_corr2,    &data.min[th_idx][12], &data.max[th_idx][12], &data.sum[th_idx][12]);

		/* (14) Maximal Correlation Coefficient */
    assign_feature(features.max_corr_coef, &data.min[th_idx][13], &data.max[th_idx][13], &data.sum[th_idx][13]);
	}

	/* copy the values to the output array in the right output order */
	for (unsigned int a = 0; a < HARALICK_FEATURE_SIZE; a++)
	{
		data.out_buffer[th_idx][a]                         = data.sum[th_idx][a] / 4;
		data.out_buffer[th_idx][a + HARALICK_FEATURE_SIZE] = data.max[th_idx][a] - data.min[th_idx][a];
	}

  for (unsigned int a = 0; a < HARALICK_OUT_SIZE; a++) 
    data.out[th_idx][a] = data.out_buffer[th_idx][HARALICK_OUT_MAP[a]];
}



__device__ void get_intensity_range(pix_data *pixels, int pixel_count, double *min, double *max)
{
	double min1     = INF;
  double max1     = -INF;

	/* compute min and max */
	for (long pixel_index = 0; pixel_index < pixel_count; pixel_index++)
	{
		if (pixels[pixel_index].intensity > max1)
			max1 = pixels[pixel_index].intensity;

		if (pixels[pixel_index].intensity < min1)
			min1 = pixels[pixel_index].intensity;
	}

	if (max) *max = max1;
	if (min) *min = min1;
}



__device__ void normalize_to_8_bits(pix_data *image, int width, int height, int bits, 
                                    double min, double max, unsigned char **gray)
{
	// for more than 8 bits - normalize the image to (0,255) range 
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) { 
      pix_data pixel = get_pixel(image, x, y, 0, width, height);

			if (bits > 8) 
				gray[y][x] = unsigned char((pixel.intensity - min)*(255.0 / (max - min)));
			else 
				gray[y][x] = unsigned char(pixel.intensity);
    }
  }
}



__device__ inline void assign_feature(float feature, double *min, double *max, double *sum)
{
  *sum += feature;

  if (feature < (*min)) 
    *min = feature;

  if (feature > (*max)) 
    *max = feature;
}



HaralickData cuda_allocate_haralick_data(const std::vector<ImageMatrix *> &images)
{
  HaralickData data;
  memset(&data, 0, sizeof(data));

  cudaMalloc(&data.distance, sizeof(double) * images.size());
  cudaMemset(data.distance, 0 , sizeof(double) * images.size());

  unsigned char ***th_gray = new unsigned char**[images.size()];
  for(int i = 0; i < images.size(); i++)
  {
    unsigned char **gray = new unsigned char*[images[i]->height];
    for(int j = 0; j < images[i]->height; j++) 
      cudaMalloc(&gray[j], sizeof(unsigned char) * images[i]->width);

    cudaMalloc(&th_gray[i], sizeof(unsigned char *) * images[i]->height);
    cudaMemcpy(th_gray[i], gray, sizeof(unsigned char*) * images[i]->height, cudaMemcpyHostToDevice);

    delete [] gray;
  }

  cudaMalloc(&data.gray, sizeof(unsigned char **) * images.size());
  cudaMemcpy(data.gray, th_gray, sizeof(unsigned char **) * images.size(), cudaMemcpyHostToDevice);
  delete [] th_gray;

  cuda_alloc_cube_array<double>(HARALICK_FEATURE_SIZE, images.size(), data.min);
  cuda_alloc_cube_array<double>(HARALICK_FEATURE_SIZE, images.size(), data.max);
  cuda_alloc_cube_array<double>(HARALICK_FEATURE_SIZE, images.size(), data.sum);

  cuda_alloc_cube_array<double>(HARALICK_OUT_SIZE, images.size(), data.out_buffer);
  cuda_alloc_cube_array<double>(HARALICK_OUT_SIZE, images.size(), data.out);
}



std::vector<FileSignatures> cuda_get_haralick_signatures(const std::vector<ImageMatrix *> &images, HaralickData &data)
{
	int outs_size = MAX_OUTPUT_SIZE * image_matrix_count;
  double *outs = new double[MAX_OUTPUT_SIZE * image_matrix_count];

  int   sizes_size = image_matrix_count;
  long *lSizes     = new long[image_matrix_count];

  cudaMemcpy(outs, outputs, outs_size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(lSizes, sizes, sizes_size * sizeof(long), cudaMemcpyDeviceToHost);

  char buffer[64];
  for(int i = 0; i < image_matrix_count; i++)
  {
    for(int j = 0; j < lSizes[i]; j++)
    {
      sprintf(buffer, "Haarlick bin %i", j);
      double value = outs[i * MAX_OUTPUT_SIZE + j];
      signatures.add_signature(buffer, image_matrices[i]->source_file, value);
    }
  }

  delete [] outs;
  delete [] lSizes;
  std::vector<FileSignatures> signatures;
  return signatures;
}



void cuda_delete_haralick_data(const std::vector<ImageMatrix *> &images, HaralickData &data)
{
  cudaFree(data.distance);

  // Gray stuff
  unsigned char ***th_gray = new unsigned char**[images.size()];
  cudaMemcpy(th_gray, data.gray, sizeof(unsigned char **) * images.size(), cudaMemcpyDeviceToHost);
  for(int i = 0; i < images.size(); i++)
  {
    unsigned char ** gray = new unsigned char*[images[i]->width];
    cudaMemcpy(gray, th_gray[i], sizeof(unsigned char *) * images[i]->width, cudaMemcpyDeviceToHost);
    for(int j = 0; j < images[i]->width; j++)
      cudaFree(gray[j]);
    cudaFree(th_gray[i]);
    delete [] gray;
  }
  delete [] th_gray;
  cudaFree(data.gray);

  cuda_free_multidim_arr<double>(data.min,        images.size());
  cuda_free_multidim_arr<double>(data.max,        images.size());
  cuda_free_multidim_arr<double>(data.sum,        images.size());

  cuda_free_multidim_arr<double>(data.out_buffer, images.size());
  cuda_free_multidim_arr<double>(data.out,        images.size());
}

#pragma package(smart_init)