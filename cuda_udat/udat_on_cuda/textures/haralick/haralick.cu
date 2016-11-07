//---------------------------------------------------------------------------

#pragma hdrstop

#include "haralick.h"
#include "CVIPtexture.h"
#include "../../image/image_matrix.h"
#include "../../utils/cuda_utils.h"

#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"



/* Order in which the features go into the output array. */
__device__ const unsigned int HARALICK_OUT_MAP[HARALICK_OUT_SIZE] = {
  0,  14, //  (1) Angular Second Moment
  1,  15, //  (2) Contrast
  2,  16, //  (3) Correlation
  9,  23, // (10) Difference Variance
  10, 24, // (11) Difference Entropy
  8,  22, //  (9) Entropy
  11, 25, // (12) Measure of Correlation 1
  4,  18, //  (5) Inverse Difference Moment
  13, 27, // (14) Maximal Correlation Coefficient
  12, 26, // (13) Measure of Correlation 2
  5,  19, //  (6) Sum Average
  7,  21, //  (8) Sum Entropy
  6,  20, //  (7) Sum Variance
  3,  17  //  (4) Variance
};



__global__ void cuda_haralick(CudaImages images, HaralickData data) 
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
		Extract_Texture_Features(&features, data.tone_matrix[th_idx], data.buffer_matrix[th_idx],
                             data.buffer_vector[th_idx], (int)data.distance[th_idx], angle, 
                             data.gray[th_idx], images.heights[th_idx], images.widths[th_idx], 
                             (int)max_value);

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

   data.out[th_idx][0]  = data.out_buffer[th_idx][0];
   data.out[th_idx][1]  = data.out_buffer[th_idx][14];
   data.out[th_idx][2]  = data.out_buffer[th_idx][1];
   data.out[th_idx][3]  = data.out_buffer[th_idx][15];
   data.out[th_idx][4]  = data.out_buffer[th_idx][2];
   data.out[th_idx][5]  = data.out_buffer[th_idx][16];
   data.out[th_idx][6]  = data.out_buffer[th_idx][9];
   data.out[th_idx][7]  = data.out_buffer[th_idx][23];
   data.out[th_idx][8]  = data.out_buffer[th_idx][10];
   data.out[th_idx][9]  = data.out_buffer[th_idx][24];
   data.out[th_idx][10] = data.out_buffer[th_idx][8];
   data.out[th_idx][11] = data.out_buffer[th_idx][22];
   data.out[th_idx][12] = data.out_buffer[th_idx][11];
   data.out[th_idx][13] = data.out_buffer[th_idx][25];
   data.out[th_idx][14] = data.out_buffer[th_idx][4];
   data.out[th_idx][15] = data.out_buffer[th_idx][18];
   data.out[th_idx][16] = data.out_buffer[th_idx][13];
   data.out[th_idx][17] = data.out_buffer[th_idx][27];
   data.out[th_idx][18] = data.out_buffer[th_idx][12];
   data.out[th_idx][19] = data.out_buffer[th_idx][26];
   data.out[th_idx][20] = data.out_buffer[th_idx][5];
   data.out[th_idx][21] = data.out_buffer[th_idx][19];
   data.out[th_idx][22] = data.out_buffer[th_idx][7];
   data.out[th_idx][23] = data.out_buffer[th_idx][21];
   data.out[th_idx][24] = data.out_buffer[th_idx][6];
   data.out[th_idx][25] = data.out_buffer[th_idx][20];
   data.out[th_idx][26] = data.out_buffer[th_idx][3];
   data.out[th_idx][27] = data.out_buffer[th_idx][17];
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
      pix_data pixel = get_pixel(image, width, height, x, y, 0);

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

  cudaError status;
  status = cudaMalloc(&data.distance, sizeof(double) * images.size());
  status = cudaMemset(data.distance, 0 , sizeof(double) * images.size());

  unsigned char ***th_gray = new unsigned char**[images.size()];
  for(int i = 0; i < images.size(); i++)
  {
    unsigned char **gray = new unsigned char*[images[i]->height];
    for(int j = 0; j < images[i]->height; j++) 
      status = cudaMalloc(&gray[j], sizeof(unsigned char) * images[i]->width);

    status = cudaMalloc(&th_gray[i], sizeof(unsigned char *) * images[i]->height);
    status = cudaMemcpy(th_gray[i], gray, sizeof(unsigned char*) * images[i]->height, cudaMemcpyHostToDevice);

    delete [] gray;
  }

  status = cudaMalloc(&data.gray, sizeof(unsigned char **) * images.size());
  status = cudaMemcpy(data.gray, th_gray, sizeof(unsigned char **) * images.size(), cudaMemcpyHostToDevice);
  delete [] th_gray;

  status = cuda_alloc_cube_array<double>(HARALICK_FEATURE_SIZE, images.size(), data.min);
  status = cuda_alloc_cube_array<double>(HARALICK_FEATURE_SIZE, images.size(), data.max);
  status = cuda_alloc_cube_array<double>(HARALICK_FEATURE_SIZE, images.size(), data.sum);

  status = cuda_alloc_cube_array<double>(HARALICK_OUT_SIZE, images.size(), data.out_buffer);
  status = cuda_alloc_cube_array<double>(HARALICK_OUT_SIZE, images.size(), data.out);

  double ***tone_matrices = new double**[images.size()];
  for(int i = 0; i < images.size(); i++) {
    status = cuda_alloc_cube_array<double>((HARALICK_TONE_MAX+1), (HARALICK_TONE_MAX+1), tone_matrices[i]);
  }
  move_host_to_cuda<double **>(tone_matrices, images.size(), data.tone_matrix);
  delete [] tone_matrices;

  double ***buffer_matrix = new double**[images.size()];
  for(int i = 0; i < images.size(); i++) {
    status = cuda_alloc_cube_array<double>((HARALICK_TONE_MAX+1), (HARALICK_TONE_MAX+1), buffer_matrix[i]);
  }
  move_host_to_cuda<double **>(buffer_matrix, images.size(), data.buffer_matrix);
  delete [] buffer_matrix;

  status = cuda_alloc_cube_array<double>(HARALICK_BUF_VEC_COUNT * (HARALICK_TONE_MAX+1), images.size(), data.buffer_vector);

  return data;
}



std::vector<FileSignatures> cuda_get_haralick_signatures(const std::vector<ImageMatrix *> &images, HaralickData &data)
{
  cudaError status;

  double **output = new double*[images.size()];
  status = cudaMemcpy(output, data.out, sizeof(double *) * images.size(), cudaMemcpyDeviceToHost);
  for(int i = 0; i < images.size(); i++) 
  {
    double *out = new double[HARALICK_OUT_SIZE];
    cudaMemcpy(out, output[i], sizeof(double) * HARALICK_OUT_SIZE, cudaMemcpyDeviceToHost);
    output[i] = out;
  }

  std::vector<FileSignatures> file_signatures;
  for(int i = 0; i < images.size(); i++)
  {
    FileSignatures file_signature;
    file_signature.file_name = images[i]->source_file;

    for(int j = 0; j < HARALICK_OUT_SIZE; j++)
    {
      std::stringstream ss;
      ss << "Haralick Texture " << j;

      Signature sig;
      sig.signature_name = ss.str();
      sig.value = output[i][j];
      file_signature.signatures.push_back(sig);
    }

    file_signatures.push_back(file_signature);
  }

  for(int i = 0; i < images.size(); i++) {
    delete [] output[i];
  }
  delete [] output;

  return file_signatures;
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

  double ***tone_matrices;
  move_cuda_to_host<double **>(data.tone_matrix, images.size(), tone_matrices);
  for(int i = 0; i < images.size(); i++)
    cuda_free_multidim_arr<double>(tone_matrices[i], HARALICK_TONE_MAX);
  cudaFree(data.tone_matrix);
  delete [] tone_matrices;

  double ***buffer_matrix;
  move_cuda_to_host<double **>(data.buffer_matrix, images.size(), buffer_matrix);
  for(int i = 0; i < images.size(); i++)
    cuda_free_multidim_arr<double>(buffer_matrix[i], HARALICK_TONE_MAX);
  cudaFree(data.buffer_matrix);
  delete [] buffer_matrix;

  cuda_free_multidim_arr<double>(data.buffer_vector, images.size());
}

#pragma package(smart_init)