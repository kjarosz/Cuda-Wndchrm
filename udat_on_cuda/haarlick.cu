//---------------------------------------------------------------------------

#pragma hdrstop

#ifndef BORLAND_C
#include <stdlib.h>
#include <stdio.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include "haarlick.h"
#include "CVIPtexture.h"
#include "image_matrix.h"
#include "device_launch_parameters.h"
//---------------------------------------------------------------------------
/* haarlick
output -array of double- a pre-allocated array of 28 doubles
*/

__global__ void CUDA_haarlick2d(ImageMatrix *Im, double distance, double *out) {
	const int i = threadIdx.x * blockDim.x + threadIdx.x;
	int a, x, y;
	unsigned char **p_gray;
	TEXTURE *features;
	long angle;
	double min[14], max[14], sum[14];
	double min_value = INF, max_value = -INF;//max_value=pow(2,Im->bits)-1;

	if (distance <= 0) distance = 1;

	p_gray = new unsigned char *[Im->height];
	for (y = 0; y<Im[i].height; y++)
		p_gray[y] = new unsigned char[Im[i].width];
	/* for more than 8 bits - normalize the image to (0,255) range */

	Im[i].BasicStatistics(NULL, NULL, NULL, &min_value, &max_value, NULL, 0);
	for (y = 0; y<Im[i].height; y++)
		for (x = 0; x<Im[i].width; x++)
			if (Im[i].bits>8) 
				p_gray[y][x] = (unsigned char)((Im[i].pixel(x, y, 0).intensity - min_value)*(255.0 / (max_value - min_value)));
			else 
				p_gray[y][x] = (unsigned char)(Im[i].pixel(x, y, 0).intensity);

	for (a = 0; a<14; a++)
	{
		min[a] = INF;
		max[a] = -INF;
		sum[a] = 0;
	}

	for (angle = 0; angle <= 135; angle = angle + 45)
	{
		features = Extract_Texture_Features((int)distance, angle, p_gray, Im[i].height, Im[i].width, (int)max_value);
		/*  (1) Angular Second Moment */
		sum[0] += features->ASM;
		if (features->ASM<min[0]) min[0] = features->ASM;
		if (features->ASM>max[0]) max[0] = features->ASM;
		/*  (2) Contrast */
		sum[1] += features->contrast;
		if (features->contrast<min[1]) min[1] = features->contrast;
		if (features->contrast>max[1]) max[1] = features->contrast;
		/*  (3) Correlation */
		sum[2] += features->correlation;
		if (features->correlation<min[2]) min[2] = features->correlation;
		if (features->correlation>max[2]) max[2] = features->correlation;
		/*  (4) Variance */
		sum[3] += features->variance;
		if (features->variance<min[3]) min[3] = features->variance;
		if (features->variance>max[3]) max[3] = features->variance;
		/*  (5) Inverse Diffenence Moment */
		sum[4] += features->IDM;
		if (features->IDM<min[4]) min[4] = features->IDM;
		if (features->IDM>max[4]) max[4] = features->IDM;
		/*  (6) Sum Average */
		sum[5] += features->sum_avg;
		if (features->sum_avg<min[5]) min[5] = features->sum_avg;
		if (features->sum_avg>max[5]) max[5] = features->sum_avg;
		/*  (7) Sum Variance */
		sum[6] += features->sum_var;
		if (features->sum_var<min[6]) min[6] = features->sum_var;
		if (features->sum_var>max[6]) max[6] = features->sum_var;
		/*  (8) Sum Entropy */
		sum[7] += features->sum_entropy;
		if (features->sum_entropy<min[7]) min[7] = features->sum_entropy;
		if (features->sum_entropy>max[7]) max[7] = features->sum_entropy;
		/*  (9) Entropy */
		sum[8] += features->entropy;
		if (features->entropy<min[8]) min[8] = features->entropy;
		if (features->entropy>max[8]) max[8] = features->entropy;
		/* (10) Difference Variance */
		sum[9] += features->diff_var;
		if (features->diff_var<min[9]) min[9] = features->diff_var;
		if (features->diff_var>max[9]) max[9] = features->diff_var;
		/* (11) Diffenence Entropy */
		sum[10] += features->diff_entropy;
		if (features->diff_entropy<min[10]) min[10] = features->diff_entropy;
		if (features->diff_entropy>max[10]) max[10] = features->diff_entropy;
		/* (12) Measure of Correlation 1 */
		sum[11] += features->meas_corr1;
		if (features->meas_corr1<min[11]) min[11] = features->meas_corr1;
		if (features->meas_corr1>max[11]) max[11] = features->meas_corr1;
		/* (13) Measure of Correlation 2 */
		sum[12] += features->meas_corr2;
		if (features->meas_corr2<min[12]) min[12] = features->meas_corr2;
		if (features->meas_corr2>max[12]) max[12] = features->meas_corr2;
		/* (14) Maximal Correlation Coefficient */
		sum[13] += features->max_corr_coef;
		if (features->max_corr_coef<min[13]) min[13] = features->max_corr_coef;
		if (features->max_corr_coef>max[13]) max[13] = features->max_corr_coef;
		free(features);
	}

	for (y = 0; y<Im[i].height; y++)
		delete p_gray[y];
	delete p_gray;

	/* copy the values to the output array in the right output order */
	double temp[28];
	for (a = 0; a<14; a++)
	{
		temp[a] = sum[a] / 4;
		temp[a + 14] = max[a] - min[a];
	}

	out[0] = temp[0];
	out[1] = temp[14];
	out[2] = temp[1];
	out[3] = temp[15];
	out[4] = temp[2];
	out[5] = temp[16];
	out[6] = temp[9];
	out[7] = temp[23];
	out[8] = temp[10];
	out[9] = temp[24];
	out[10] = temp[8];
	out[11] = temp[22];
	out[12] = temp[11];
	out[13] = temp[25];
	out[14] = temp[4];
	out[15] = temp[18];
	out[16] = temp[13];
	out[17] = temp[27];
	out[18] = temp[12];
	out[19] = temp[26];
	out[20] = temp[5];
	out[21] = temp[19];
	out[22] = temp[7];
	out[23] = temp[21];
	out[24] = temp[6];
	out[25] = temp[20];
	out[26] = temp[3];
	out[27] = temp[17];
}

void allocate_haarlick_memory(ImageMatrix *matrix, double distance, double *out) {
	// haarlick computation
	double *d_distance, *d_out;
	ImageMatrix *d_matrix;
	/* removed currently because I believe the device is able to allocate its own memory for variables declared within a kernel function
	TEXTURE *d_features;
	int d_a, d_x, d_y;
	unsigned char **d_p_gray;
	TEXTURE *d_features;
	long d_angle;
	double d_min[14], d_max[14], d_sum[14];
	double d_min_value = INF, d_max_value = -INF;//max_value=pow(2,Im->bits)-1;
	size_t pitch;

	cudaMallocPitch((void**)&d_p_gray, &pitch, matrix->width * sizeof(unsigned char), matrix->height);
	cudaMalloc((void**)d_a, sizeof(int));
	cudaMalloc((void**)d_x, sizeof(int));
	cudaMalloc((void**)d_y, sizeof(int));
	cudaMalloc((void**)d_angle, sizeof(long));
	cudaMalloc((void**)&d_min, sizeof(double));
	cudaMalloc((void**)&d_max, sizeof(double));
	cudaMalloc((void**)&d_sum, sizeof(double));
	cudaMalloc((void**)&d_min_value, sizeof(double));
	cudaMalloc((void**)&d_max_value, sizeof(double));
	cudaMalloc((void**)&d_features, sizeof(TEXTURE));
	*/

	// Allocate memory for variables on device
	cudaMalloc((void**)&d_matrix, sizeof(ImageMatrix));
	cudaMalloc((void**)&d_out, sizeof(double));
	cudaMalloc((void**)&d_distance, sizeof(double));


	cudaMemcpy(d_matrix, matrix, sizeof(ImageMatrix), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_distance, &distance, sizeof(double), cudaMemcpyHostToDevice);
	CUDA_haarlick2d<<<1, 1>>>(matrix, distance, out);
	cudaMemcpy(out, d_out, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_matrix);
	cudaFree(d_out);
	cudaFree(d_distance);
}


#pragma package(smart_init)