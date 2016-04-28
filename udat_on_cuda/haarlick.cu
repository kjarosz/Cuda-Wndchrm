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

__global__ void haarlick(pix_data *pixels, double *distance, double *out, int *height, int *width, int *depth, unsigned short int *bits) {
	const int i = threadIdx.x * blockDim.x + threadIdx.x;
	int a, x, y;
	unsigned char **p_gray;
	TEXTURE *features;
	long angle;
	double min[14], max[14], sum[14];
	double min_value = INF, max_value = -INF;//max_value=pow(2,Im->bits)-1;

	if (distance <= 0) distance[i] = 1;

	p_gray = new unsigned char *[height[i]];
	for (y = 0; y<height[i]; y++)
		p_gray[y] = new unsigned char[width[i]];
	/* for more than 8 bits - normalize the image to (0,255) range */

	BasicStatistics(pixels, &min_value, &max_value, 0, height[i]*width[i]*depth[i]);
	for (y = 0; y<height[i]; y++)
		for (x = 0; x<width[i]; x++)
			if (bits[i]>8) 
				p_gray[y][x] = (unsigned char)((get_pixel(pixels, x, y, 0, width[i], height[i]).intensity - min_value)*(255.0 / (max_value - min_value)));
			else 
				p_gray[y][x] = (unsigned char)(get_pixel(pixels, x, y, 0, width[i], height[i]).intensity);

	for (a = 0; a<14; a++)
	{
		min[a] = INF;
		max[a] = -INF;
		sum[a] = 0;
	}

	for (angle = 0; angle <= 135; angle = angle + 45)
	{
		features = Extract_Texture_Features((int)distance, angle, p_gray, height[i], width[i], (int)max_value);
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

	for (y = 0; y<height[i]; y++)
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


__device__ void BasicStatistics(pix_data *color_data, double *min, double *max, int bins, int num_pixels)
{
	long pixel_index;
	double *pixels;
	double min1 = INF, max1 = -INF, mean_sum = 0;

	pixels = new double[num_pixels];

	/* compute the average, min and max */
	for (pixel_index = 0; pixel_index<num_pixels; pixel_index++)
	{
		mean_sum += color_data[pixel_index].intensity;
		if (color_data[pixel_index].intensity>max1)
			max1 = color_data[pixel_index].intensity;
		if (color_data[pixel_index].intensity<min1)
			min1 = color_data[pixel_index].intensity;
		pixels[pixel_index] = color_data[pixel_index].intensity;
	}
	if (max) *max = max1;
	if (min) *min = min1;

	delete pixels;
}

#pragma package(smart_init)