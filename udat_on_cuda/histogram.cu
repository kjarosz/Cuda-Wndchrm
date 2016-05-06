#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "image_matrix.h"


__device__ void histogram(pix_data *data, double *bins, unsigned short bins_num, int imhist, int *width, int *height, int *depth, int *bits)
{
	const int index = threadIdx.x * blockDim.x + threadIdx.x;
	long a;
	double min = INF, max = -INF;
	/* find the minimum and maximum */
	if (imhist == 1)    /* similar to the Matlab imhist */
	{
		min = 0;
		max = pow(2.0, (double)bits[index]) - 1;
	}
	else
	{
		for (a = 0; a<width[index]*height[index]*depth[index]; a++)
		{
			if (data[a].intensity>max)
				max = data[a].intensity;
			if (data[a].intensity<min)
				min = data[a].intensity;
		}
	}
	/* initialize the bins */
	for (a = 0; a<bins_num; a++)
		bins[a] = 0;

	/* build the histogram */
	for (a = 0; a<width[index]*height[index]*depth[index]; a++)
	if (data[a].intensity == max) bins[bins_num - 1] += 1;
	else bins[(int)(((data[a].intensity - min) / (max - min))*bins_num)] += 1;

	return;
}

__device__ void multiscalehistogram(pix_data *data, double *out, int *width, int *height, int *depth, int *bits)
{
	int a;
	double max = 0;
	histogram(data, out, 3, 0, width, height, depth, bits);
	histogram(data, &(out[3]), 5, 0, width, height, depth, bits);
	histogram(data, &(out[8]), 7, 0, width, height, depth, bits);
	histogram(data, &(out[15]), 9, 0, width, height, depth, bits);
	for (a = 0; a<24; a++)
	if (out[a]>max) max = out[a];
	for (a = 0; a<24; a++)
		out[a] = out[a] / max;
}