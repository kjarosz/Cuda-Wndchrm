/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*                                                                               */
/*                                                                               */
/*                   Universal Data Analysis Tool (UDAT)                         */
/*                                                                               */
/*                                                                               */
/*    This library is free software; you can redistribute it and/or              */
/*    modify it under the terms of the GNU Lesser General Public                 */
/*    License as published by the Free Software Foundation; either               */
/*    version 2.1 of the License, or (at your option) any later version.         */
/*                                                                               */
/*    This library is distributed in the hope that it will be useful,            */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of             */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU          */
/*    Lesser General Public License for more details.                            */
/*                                                                               */
/*    You should have received a copy of the GNU Lesser General Public           */
/*    License along with this library; if not, write to the Free Software        */
/*    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA  */
/*                                                                               */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*                                                                               */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*    Written by:  Lior Shamir <lshamir [at] mtu [dot] edu >                     */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/



#pragma hdrstop

#include <math.h>
#include <stdio.h>
#include "image_matrix.h"
#include "textures/gabor.h"
#include "textures/tamura.h"
#include "haarlick.h"
#include "zernike.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef BORLAND_C
#ifndef VISUAL_C
#include <stdlib.h>
#include <string.h>
#endif
#endif

#ifdef BORLAND_C
#include "libtiff32/tiffio.h"
#include <jpeg.hpp>
#endif


#ifdef VISUAL_C
#include "libtiff32/tiffio.h"
#include <stdlib.h>
#endif

#define SOUND_FILES
#ifndef WIN32
#ifndef VISUAL_C
#define __int64 uint64_t
#endif
#endif
#include "sndfile.h"

#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)

RGBcolor HSV2RGB(HSVcolor hsv)
{
	RGBcolor rgb;
	float R, G, B;
	float H, S, V;
	float i, f, p, q, t;

	H = hsv.hue;
	S = (float)(hsv.saturation) / 240;
	V = (float)(hsv.value) / 240;
	if (S == 0 && H == 0) { R = G = B = V; }  /*if S=0 and H is undefined*/
	H = H*(float)(360.0 / 240.0);
	if (H == 360) { H = 0; }
	H = H / 60;
	i = floor(H);
	f = H - i;
	p = V*(1 - S);
	q = V*(1 - (S*f));
	t = V*(1 - (S*(1 - f)));

	if (i == 0) { R = V;  G = t;  B = p; }
	if (i == 1) { R = q;  G = V;  B = p; }
	if (i == 2) { R = p;  G = V;  B = t; }
	if (i == 3) { R = p;  G = q;  B = V; }
	if (i == 4) { R = t;  G = p;  B = V; }
	if (i == 5) { R = V;  G = p;  B = q; }

	rgb.red = (byte)(R * 255);
	rgb.green = (byte)(G * 255);
	rgb.blue = (byte)(B * 255);
	return rgb;
}
//-----------------------------------------------------------------------
HSVcolor RGB2HSV(RGBcolor rgb)
{
	float r, g, b, h, max, min, delta;
	HSVcolor hsv;

	r = (float)(rgb.red) / 255;
	g = (float)(rgb.green) / 255;
	b = (float)(rgb.blue) / 255;

	max = MAX(r, MAX(g, b)), min = MIN(r, MIN(g, b));
	delta = max - min;

	hsv.value = (byte)(max*240.0);
	if (max != 0.0)
		hsv.saturation = (byte)((delta / max)*240.0);
	else
		hsv.saturation = 0;
	if (hsv.saturation == 0) hsv.hue = 0; //-1;
	else {
		if (r == max)
			h = (g - b) / delta;
		else if (g == max)
			h = 2 + (b - r) / delta;
		else if (b == max)
			h = 4 + (r - g) / delta;
		h *= 60.0;
		if (h < 0.0) h += 360.0;
		hsv.hue = (byte)(h *(240.0 / 360.0));
	}
	return(hsv);
}


//--------------------------------------------------------------------------
TColor RGB2COLOR(RGBcolor rgb)
{
	return((TColor)(rgb.blue * 65536 + rgb.green * 256 + rgb.red));
}

double COLOR2GRAY(TColor color1)
{
	double r, g, b;

	r = (byte)(color1 & 0xFF);
	g = (byte)((color1 & 0xFF00) >> 8);
	b = (byte)((color1 & 0xFF0000) >> 16);

	return((0.3*r + 0.59*g + 0.11*b));
}


#ifdef BORLAND_C

//--------------------------------------------------------------------------
int ImageMatrix::LoadImage(TPicture *picture, int ColorMode)
{
	int a, b, x, y;
	pix_data pix;
	width = picture->Width;
	height = picture->Height;
	depth = 1;   /* TPicture is a two-dimentional structure */
	bits = 8;
	this->ColorMode = ColorMode;
	/* allocate memory for the image's pixels */
	data = new pix_data[width*height*depth];
	if (!data) return(0); /* memory allocation failed */
	/* load the picture */
	for (y = 0; y<height; y++)
	for (x = 0; x<width; x++)
	{
		pix.clr.RGB.red = (byte)(picture->Bitmap->Canvas->Pixels[x][y] & 0xFF);               /* red value */
		pix.clr.RGB.green = (byte)((picture->Bitmap->Canvas->Pixels[x][y] & 0xFF00) >> 8);    /* green value */
		pix.clr.RGB.blue = (byte)((picture->Bitmap->Canvas->Pixels[x][y] & 0xFF0000) >> 16);  /* blue value */
		if (ColorMode == cmHSV) pix.clr.HSV = RGB2HSV(pix.clr.RGB);
		pix.intensity = COLOR2GRAY(picture->Bitmap->Canvas->Pixels[x][y]);
		set(x, y, 0, pix);
	}
	return(1);
}

int ImageMatrix::LoadBMP(char *filename, int ColorMode)
{
	TPicture *picture;
	int ret_val = 0;
	picture = new TPicture;
	if (FileExists(filename))
	{
		picture->LoadFromFile(filename);
		ret_val = LoadImage(picture, ColorMode);
	}
	delete picture;
	return(ret_val);
}

int ImageMatrix::LoadJPG(char *filename, int ColorMode)
{
	TJPEGImage *i;
	TPicture *picture;
	int ret_val;
	i = new TJPEGImage();
	try
	{
		i->LoadFromFile(filename);
	}
	catch (...) { return(0); }
	picture = new TPicture();
	picture->Bitmap->Assign(i);
	ret_val = LoadImage(picture, ColorMode);
	delete i;
	delete picture;
	return(ret_val);
}

#endif


/*
OpenImage - opens an image file and loads its content into the matrix
image_file_name -char *- full path to the image file
downsample -int- a scale to downsample the image by
bounding_rect -rect *- a rect within the image. If specified only that recangle is copied to the matrix
mean -double- a mean value to normalize for.
stddev -double- standard deviation to normalize for.
DynamicRange -long- change to a new dynamic range. Ignore if 0.
*/
int ImageMatrix::OpenImage(char *image_file_name, int downsample, rect *bounding_rect, double mean, double stddev, long DynamicRange, double otsu_mask)
{
	int res = 0;
#ifdef BORLAND_C
	if (strstr(image_file_name, ".bmp") || strstr(image_file_name, ".BMP"))
		res = LoadBMP(image_file_name, cmHSV);
	else
	if (strstr(image_file_name, ".jpg") || strstr(image_file_name, ".jpeg") || strstr(image_file_name, ".JPG"))
		res = LoadJPG(image_file_name, cmHSV);
	else
#endif
	if (strstr(image_file_name, ".tif") || strstr(image_file_name, ".TIF"))
	{
		res = LoadTIFF(image_file_name);
	}
	//#endif
	else
	if (strstr(image_file_name, ".ppm") || strstr(image_file_name, ".PPM"))
		res = LoadPPM(image_file_name, cmHSV);
	else /* dicom, jpeg, any other file format */
	if (strstr(image_file_name, ".dcm") || strstr(image_file_name, ".DCM") || strstr(image_file_name, ".jpg") || strstr(image_file_name, ".JPG"))
	{
		char buffer[512], temp_filename[64];
		sprintf(temp_filename, "tempimage%d.tif", rand() % 30000);  /* the getpid is to allow several processes to run from the same folder */
		sprintf(buffer, "convert %s %s", image_file_name, temp_filename);
		system(buffer);
		res = LoadTIFF(temp_filename);
		if (res <= 0) printf("Could not convert '%s' to tiff\n", image_file_name);
#if defined BORLAND_C || defined VISUAL_C
		sprintf(buffer, "del %s", temp_filename);
#else
		sprintf(buffer, "rm %s", temp_filename);
#endif
		system(buffer);
	}
	else if (strstr(image_file_name, ".wav") || strstr(image_file_name, ".WAV")) res = LoadWav(image_file_name);

	if (res)  /* add the image only if it was loaded properly */
	{
		if (DynamicRange) SetDynamicRange(DynamicRange);    /* change the dynamic range of the image (if needed)    */
		if (bounding_rect && bounding_rect->x >= 0)                    /* compute features only from an area of the image     */
		{
			ImageMatrix *temp;
			temp = new ImageMatrix(this, bounding_rect->x, bounding_rect->y, bounding_rect->x + bounding_rect->w - 1, bounding_rect->y + bounding_rect->h - 1, 0, depth - 1);
			delete data;
			width = temp->width; height = temp->height;
			if (!(data = new pix_data[width*height*depth])) return(0);  /* allocate new memory */
			memcpy(data, temp->data, width*height*depth*sizeof(pix_data));
			//         for (int a=0;a<width*height*depth;a++)
			//		   data[a]=temp->data[a];
			delete temp;
		}
		if (downsample>0 && downsample<100)  /* downsample by a given factor */
			Downsample(((double)downsample) / 100.0, ((double)downsample) / 100.0);   /* downsample the image */
		if (mean>0)  /* normalize to a given mean and standard deviation */
			normalize(-1, -1, -1, mean, stddev);
		if (otsu_mask>0) Mask(Otsu()*otsu_mask);    /* mask using the otsu threshold */
	}
	/*
	{   ImageMatrix *color_mask;
	pix_data blank_pix;
	blank_pix.intensity=0.0;
	blank_pix.clr.RGB.red=0;
	blank_pix.clr.RGB.green=0;
	blank_pix.clr.RGB.blue=0;
	color_mask=this->duplicate();
	color_mask->ColorTransform(NULL,0);
	for (int z=0;z<depth;z++)
	for (int y=0;y<height;y++)
	for (int x=0;x<width;x++)
	if ((color_mask->pixel(x,y,z).intensity!=(int)((255*COLOR_DARK_BROWN)/COLORS_NUM)) && (color_mask->pixel(x,y,z).intensity!=(int)((255*COLOR_LIGHT_BROWN)/COLORS_NUM))) set(x,y,z,blank_pix);
	//   SaveTiff("temp.tif");
	delete color_mask;
	}
	*/
	return(res);
}

/* simple constructors */

ImageMatrix::ImageMatrix()
{
	data = NULL;
	width = 0;
	height = 0;
	depth = 1;
	ColorMode = cmHSV;    /* set a diffult color mode */
}

ImageMatrix::ImageMatrix(int width, int height, int depth)
{
	bits = 8; /* set some default value */
	if (depth<1) depth = 1;    /* make sure the image is at least two dimensional */
	ColorMode = cmHSV;
	this->width = width;
	this->height = height;
	this->depth = depth;
	data = new pix_data[width*height*depth];
	Initialize();
	//   memset(data,0,width*height*depth*sizeof(pix_data));  /* initialize */
}

/* create an image which is part of the image
(x1,y1) - top left
(x2,y2) - bottom right
*/
ImageMatrix::ImageMatrix(ImageMatrix *matrix, int x1, int y1, int x2, int y2, int z1, int z2)
{
	int x, y, z;
	bits = matrix->bits;
	ColorMode = matrix->ColorMode;
	/* verify that the image size is OK */
	if (x1<0) x1 = 0;
	if (y1<0) y1 = 0;
	if (z1<0) z1 = 0;
	if (x2 >= matrix->width) x2 = matrix->width - 1;
	if (y2 >= matrix->height) y2 = matrix->height - 1;
	if (z2 >= matrix->depth) z2 = matrix->depth - 1;

	width = x2 - x1 + 1;
	height = y2 - y1 + 1;
	depth = z2 - z1 + 1;
	data = new pix_data[width*height*depth];

	for (z = z1; z<z1 + depth; z++)
	for (y = y1; y<y1 + height; y++)
	for (x = x1; x<x1 + width; x++)
		set(x - x1, y - y1, z - z1, matrix->pixel(x, y, z));
}

/* free the memory allocated in "ImageMatrix::LoadImage" */
ImageMatrix::~ImageMatrix()
{
	if (data) delete data;
	data = NULL;
}

/* get a pixel value */
pix_data ImageMatrix::pixel(int x, int y, int z)
{
	return(data[z*width*height + y*width + x]);
}

/* assigne a pixel value */
void ImageMatrix::set(int x, int y, int z, pix_data val)
{
	data[z*width*height + y*width + x] = val;
}

/* assigne a pixel intensity only */
void ImageMatrix::SetInt(int x, int y, int z, double val)
{
	data[z*width*height + y*width + x].intensity = val;
}

/* initialize the image */
void ImageMatrix::Initialize()
{
	int x, y, z;
	pix_data zero_pix;
	zero_pix.clr.RGB.red = zero_pix.clr.RGB.green = zero_pix.clr.RGB.blue = 0;
	zero_pix.intensity = 0.0;
	for (z = 0; z<depth; z++)
	for (y = 0; y<height; y++)
	for (x = 0; x<width; x++)
		set(x, y, z, zero_pix);
}

/* compute the difference from another image */
void ImageMatrix::diff(ImageMatrix *matrix)
{
	int x, y, z;
	for (z = 0; z<depth; z++)
	for (y = 0; y<height; y++)
	for (x = 0; x<width; x++)
	{
		pix_data pix1, pix2;
		pix1 = pixel(x, y, z);
		pix2 = matrix->pixel(x, y, z);
		pix1.intensity = fabs(pix1.intensity - pix2.intensity);
		pix1.clr.RGB.red = (byte)abs(pix1.clr.RGB.red - pix2.clr.RGB.red);
		pix1.clr.RGB.green = (byte)abs(pix1.clr.RGB.green - pix2.clr.RGB.green);
		pix1.clr.RGB.blue = (byte)abs(pix1.clr.RGB.blue - pix2.clr.RGB.blue);
		set(x, y, z, pix1);
	}
}


/* duplicate
create another matrix the same as the first
*/
ImageMatrix *ImageMatrix::duplicate()
{
	ImageMatrix *new_matrix;
	new_matrix = new ImageMatrix;
	new_matrix->data = new pix_data[width*height*depth];
	if (!(new_matrix->data)) return(NULL); /* memory allocation failed */
	new_matrix->width = width;
	new_matrix->height = height;
	new_matrix->depth = depth;
	new_matrix->bits = bits;
	new_matrix->ColorMode = ColorMode;
	memcpy(new_matrix->data, data, width*height*depth*sizeof(pix_data));
	return(new_matrix);
}

/* SetDynamicRange
change the dynamic range of an image
parameters:
dr -long- the new dynamic range (bits). if the value of dr is the new dynamic range is +1000, then the histogram is also normalized to the new dynamic range
*/
void ImageMatrix::SetDynamicRange(long dr)
{
	double max_val, min_val, max_dr_val;
	int normalize_histogram = (dr>1000);
	if (normalize_histogram) dr -= 1000;
	if (bits == dr && normalize_histogram == 0) return;   /* no dynamic range to change */
	if (normalize_histogram == 0)
	{
		max_val = pow(2.0, bits) - 1;
		min_val = 0;
	}
	else BasicStatistics(NULL, NULL, NULL, &min_val, &max_val, NULL, 0);
	max_dr_val = pow(2.0, dr) - 1;
	bits = (unsigned short)dr;                /* set the new dynamic range  */
	for (int a = 0; a<width*height*depth; a++)
		data[a].intensity = max_dr_val*((data[a].intensity - min_val) / (max_val - min_val));
}

/* flip
flip an image horizontaly
*/
void ImageMatrix::flip()
{
	int x, y, z;
	pix_data temp;
	for (z = 0; z<depth; z++)
	for (y = 0; y<height; y++)
	for (x = 0; x<width / 2; x++)
	{
		temp = pixel(x, y, z);
		set(x, y, z, pixel(width - x - 1, y, z));
		set(width - x - 1, y, z, temp);
	}
}

/* rotate an image
angle -double- angle of rotation (degrees)
center_x, center_y -int- the center of the rotation (0 - the center of the image)
The rotated image is of different size as the original depends on the degree of rotation
http://www.codeguru.com/cpp/g-m/bitmap/specialeffects/article.php/c1743/Rotate-a-bitmap-image.htm
*/
void ImageMatrix::rotate(double angle)
{
	int new_height, new_width, x, y, z, x1, x2, x3, y1, y2, y3, minx, miny, maxx, maxy;
	pix_data pix, *new_pixels;
	double rad_angle = angle*3.14159265 / 180;

	/* Compute dimensions of the resulting image */
	x1 = (int)(-height * sin(rad_angle));
	y1 = (int)(height * cos(rad_angle));
	x2 = (int)(width * cos(rad_angle) - height * sin(rad_angle));
	y2 = (int)(height * cos(rad_angle) + width * sin(rad_angle));
	x3 = (int)(width * cos(rad_angle));
	y3 = (int)(width * sin(rad_angle));
	minx = min(0, min(x1, min(x2, x3)));
	miny = min(0, min(y1, min(y2, y3)));
	maxx = max(x1, max(x2, x3));
	maxy = max(y1, max(y2, y3));
	new_width = maxx - minx;
	new_height = maxy - miny;

	/* allocate memory for the new image and initialize the pixels */
	new_pixels = new pix_data[new_width*new_height*depth];
	pix.intensity = 0.0;
	pix.clr.RGB.red = pix.clr.RGB.blue = pix.clr.RGB.green = 0;
	for (z = 0; z<depth; z++)
	for (y = 0; y<new_height; y++)
	for (x = 0; x<new_width; x++)
		new_pixels[z*new_width*new_height + y*new_width + x] = pix;

	for (z = 0; z<depth; z++)
	for (y = miny; y < maxy; y++)
	for (x = minx; x < maxx; x++)
	{
		int sourcex = (int)(x*cos(rad_angle) + y*sin(rad_angle));
		int sourcey = (int)(y*cos(rad_angle) - x*sin(rad_angle));
		if (sourcex >= 0 && sourcex < width && sourcey >= 0 && sourcey < height)
			new_pixels[z*new_width*new_height + y*new_width + x] = data[z*width*height + sourcey*width + sourcex];
	}

	delete data;
	data = new pix_data[depth*new_width*new_height];  /* set the data with the new dimensions */

	width = new_width;
	height = new_height;
	for (z = 0; z<depth; z++)
	for (y = 0; y<height; y++)
	for (x = 0; x<width; x++)
		set(x, y, z, new_pixels[z*width*height + y*width + x]);
	delete new_pixels;
}

/* find basic intensity statistics */

int compare_doubles(const void *a, const void *b)
{
	if (*((double *)a) > *((double*)b)) return(1);
	if (*((double*)a) == *((double*)b)) return(0);
	return(-1);
}

/* BasicStatistics
get basic statistical properties of the intensity of the image
mean -double *- pre-allocated one double for the mean intensity of the image
median -double *- pre-allocated one double for the median intensity of the image
std -double *- pre-allocated one double for the standard deviation of the intensity of the image
min -double *- pre-allocated one double for the minimum intensity of the image
max -double *- pre-allocated one double for the maximal intensity of the image
histogram -double *- a pre-allocated vector for the histogram. If NULL then histogram is not calculated
nbins -int- the number of bins for the histogram

if one of the pointers is NULL, the corresponding value is not computed.
*/
__device__ void ImageMatrix::BasicStatistics(double *mean, double *median, double *std, double *min, double *max, double *hist, int bins)
{
	long pixel_index, num_pixels;
	double *pixels;
	double min1 = INF, max1 = -INF, mean_sum = 0;

	num_pixels = height*width*depth;
	pixels = new double[num_pixels];

	/* compute the average, min and max */
	for (pixel_index = 0; pixel_index<num_pixels; pixel_index++)
	{
		mean_sum += data[pixel_index].intensity;
		if (data[pixel_index].intensity>max1)
			max1 = data[pixel_index].intensity;
		if (data[pixel_index].intensity<min1)
			min1 = data[pixel_index].intensity;
		pixels[pixel_index] = data[pixel_index].intensity;
	}
	if (max) *max = max1;
	if (min) *min = min1;
	if (mean || std) *mean = mean_sum / num_pixels;

	/* calculate the standard deviation */
	if (std)
	{
		*std = 0;
		for (pixel_index = 0; pixel_index<num_pixels; pixel_index++)
			*std = *std + pow(data[pixel_index].intensity - *mean, 2);
		*std = sqrt(*std / (num_pixels - 1));
	}

	if (hist)  /* do the histogram only if needed */
		histogram(hist, bins, 0);

	/* find the median */
	if (median)
	{
		qsort(pixels, num_pixels, sizeof(double), compare_doubles);
		*median = pixels[num_pixels / 2];
	}
	delete pixels;
}

/* normalize the pixel values into a given range
min -double- the min pixel value (ignored if <0)
max -double- the max pixel value (ignored if <0)
range -long- nominal dynamic range (ignored if <0)
mean -double- the mean of the normalized image (ignored if <0)
stddev -double- the stddev of the normalized image (ignored if <0)
*/
void ImageMatrix::normalize(double min, double max, long range, double mean, double stddev)
{
	long x;
	/* normalized to min and max */
	if (min >= 0 && max>0 && range>0)
	for (x = 0; x<width*height*depth; x++)
	{
		if (data[x].intensity<min) data[x].intensity = 0;
		else if (data[x].intensity>max) data[x].intensity = range;
		else data[x].intensity = ((data[x].intensity - min) / (max - min))*range;
	}

	/* normalize to mean and stddev */
	if (mean>0)
	{
		double original_mean, original_stddev;
		BasicStatistics(&original_mean, NULL, &original_stddev, NULL, NULL, NULL, 0);
		for (x = 0; x<width*height*depth; x++)
		{
			data[x].intensity -= (original_mean - mean);
			if (stddev>0)
				data[x].intensity = mean + (data[x].intensity - mean)*(stddev / original_stddev);
			if (data[x].intensity<0) data[x].intensity = 0;
			if (data[x].intensity>pow(2.0, bits) - 1) data[x].intensity = pow(2.0, bits) - 1;
		}
		//BasicStatistics(&original_mean, NULL, &original_stddev, NULL, NULL, NULL, 0);		  
		//printf("%f %f\n",original_mean,original_stddev);
	}
}



/* get image histogram */
void ImageMatrix::histogram(double *bins, unsigned short bins_num, int imhist)
{
	long a;
	double min = INF, max = -INF;
	/* find the minimum and maximum */
	if (imhist == 1)    /* similar to the Matlab imhist */
	{
		min = 0;
		max = pow(2.0, bits) - 1;
	}
	else
	{
		for (a = 0; a<width*height*depth; a++)
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
	for (a = 0; a<width*height*depth; a++)
	if (data[a].intensity == max) bins[bins_num - 1] += 1;
	else bins[(int)(((data[a].intensity - min) / (max - min))*bins_num)] += 1;

	return;
}



/*
FeatureStatistics
Find feature statistics. Before calling this function the image should be transformed into a binary
image using "OtsuBinaryMaskTransform".

count -int *- the number of objects detected in the binary image
Euler -int *- the euler number (number of objects - number of holes
centroid_x -int *- the x coordinate of the centroid of the binary image
centroid_y -int *- the y coordinate of the centroid of the binary image
AreaMin -int *- the smallest area
AreaMax -int *- the largest area
AreaMean -int *- the mean of the areas
AreaMedian -int *- the median of the areas
AreaVar -int *- the variance of the areas
DistMin -int *- the smallest distance
DistMax -int *- the largest distance
DistMean -int *- the mean of the distance
DistMedian -int *- the median of the distances
DistVar -int *- the variance of the distances

*/

int compare_ints(const void *a, const void *b)
{
	if (*((int *)a) > *((int *)b)) return(1);
	if (*((int *)a) == *((int *)b)) return(0);
	return(-1);
}

void ImageMatrix::FeatureStatistics(int *count, int *Euler, double *centroid_x, double *centroid_y, double *centroid_z, int *AreaMin, int *AreaMax,
	double *AreaMean, int *AreaMedian, double *AreaVar, int *area_histogram, double *DistMin, double *DistMax,
	double *DistMean, double *DistMedian, double *DistVar, int *dist_histogram, int num_bins)
{
	int object_index, inv_count;
	double sum_areas, sum_dists;
	ImageMatrix *BWImage, *BWInvert, *temp;
	int *object_areas;
	double *centroid_dists, sum_dist;

	BWInvert = duplicate();   /* check if the background is brighter or dimmer */
	BWInvert->invert();
	BWInvert->OtsuBinaryMaskTransform();
	inv_count = BWInvert->BWlabel(8);

	BWImage = duplicate();
	BWImage->OtsuBinaryMaskTransform();
	BWImage->centroid(centroid_x, centroid_y, centroid_z);
	*count = BWImage->BWlabel(8);
	if (inv_count>*count)
	{
		temp = BWImage;
		BWImage = BWInvert;
		BWInvert = temp;
		*count = inv_count;
		BWImage->centroid(centroid_x, centroid_y, centroid_z);
	}
	delete BWInvert;
	*Euler = EulerNumber(BWImage, *count) + 1;

	/* calculate the areas */
	sum_areas = 0;
	sum_dists = 0;
	object_areas = new int[*count];
	centroid_dists = new double[*count];
	for (object_index = 1; object_index <= *count; object_index++)
	{
		double x_centroid, y_centroid, z_centroid;
		object_areas[object_index - 1] = FeatureCentroid(BWImage, object_index, &x_centroid, &y_centroid, &z_centroid);
		centroid_dists[object_index - 1] = sqrt(pow(x_centroid - (*centroid_x), 2) + pow(y_centroid - (*centroid_y), 2));
		sum_areas += object_areas[object_index - 1];
		sum_dists += centroid_dists[object_index - 1];
	}
	/* compute area statistics */
	qsort(object_areas, *count, sizeof(int), compare_ints);
	*AreaMin = object_areas[0];
	*AreaMax = object_areas[*count - 1];
	if (*count>0) *AreaMean = sum_areas / (*count);
	else *AreaMean = 0;
	*AreaMedian = object_areas[(*count) / 2];
	for (object_index = 0; object_index<num_bins; object_index++)
		area_histogram[object_index] = 0;
	/* compute the variance and the histogram */
	sum_areas = 0;
	if (*AreaMax - *AreaMin>0)
	for (object_index = 1; object_index <= *count; object_index++)
	{
		sum_areas += pow(object_areas[object_index - 1] - *AreaMean, 2);
		if (object_areas[object_index - 1] == *AreaMax) area_histogram[num_bins - 1] += 1;
		else area_histogram[((object_areas[object_index - 1] - *AreaMin) / (*AreaMax - *AreaMin))*num_bins] += 1;
	}
	if (*count>1) *AreaVar = sum_areas / ((*count) - 1);
	else *AreaVar = sum_areas;

	/* compute distance statistics */
	qsort(centroid_dists, *count, sizeof(double), compare_doubles);
	*DistMin = centroid_dists[0];
	*DistMax = centroid_dists[*count - 1];
	if (*count>0) *DistMean = sum_dists / (*count);
	else *DistMean = 0;
	*DistMedian = centroid_dists[(*count) / 2];
	for (object_index = 0; object_index<num_bins; object_index++)
		dist_histogram[object_index] = 0;

	/* compute the variance and the histogram */
	sum_dist = 0;
	for (object_index = 1; object_index <= *count; object_index++)
	{
		sum_dist += pow(centroid_dists[object_index - 1] - *DistMean, 2);
		if (centroid_dists[object_index - 1] == *DistMax) dist_histogram[num_bins - 1] += 1;
		else dist_histogram[(int)(((centroid_dists[object_index - 1] - *DistMin) / (*DistMax - *DistMin))*num_bins)] += 1;
	}
	if (*count>1) *DistVar = sum_dist / ((*count) - 1);
	else *DistVar = sum_dist;

	delete BWImage;
	delete object_areas;
	delete centroid_dists;
}

/* GaborFilters */
/* ratios -array of double- a pre-allocated array of double[7]
*/
void ImageMatrix::GaborFilters2D(double *ratios)
{
	GaborTextureFilters2D(this, ratios);
}


/* haarlick
output -array of double- a pre-allocated array of 28 doubles
*/
void ImageMatrix::HaarlickTexture2D(double distance, double *out)
{
	if (distance <= 0) distance = 1;
	CUDA_haarlick2d<<<1, 1>>>(this, distance, out);
}

/* MultiScaleHistogram
histograms into 3,5,7,9 bins
Function computes signatures based on "multiscale histograms" idea.
Idea of multiscale histogram came from the belief of a unique representativity of an
image through infinite series of histograms with sequentially increasing number of bins.
Here we used 4 histograms with number of bins being 3,5,7,9.
out -array of double- a pre-allocated array of 24 bins
*/
void ImageMatrix::MultiScaleHistogram(double *out)
{
	int a;
	double max = 0;
	histogram(out, 3, 0);
	histogram(&(out[3]), 5, 0);
	histogram(&(out[8]), 7, 0);
	histogram(&(out[15]), 9, 0);
	for (a = 0; a<24; a++)
	if (out[a]>max) max = out[a];
	for (a = 0; a<24; a++)
		out[a] = out[a] / max;
}

/* TamuraTexture
Tamura texture signatures: coarseness, directionality, contrast
vec -array of double- a pre-allocated array of 6 doubles
*/
void ImageMatrix::TamuraTexture2D(double *vec)
{
	Tamura3Sigs2D(this, vec);
}

/* zernike
zvalue -array of double- a pre-allocated array of double of a suficient size
(the actual size is returned by "output_size))
output_size -* long- the number of enteries in the array "zvalues" (normally 72)
*/
void ImageMatrix::zernike2D(double *zvalues, long *output_size)
{
	mb_zernike2D(this, 0, 0, zvalues, output_size);
}

/* fractal
brownian fractal analysis
bins - the maximal order of the fractal
output - array of the size k
the code is based on: CM Wu, YC Chen and KS Hsieh, Texture features for classification of ultrasonic liver images, IEEE Trans Med Imag 11 (1992) (2), pp. 141Ð152.
method of approaximation of CC Chen, JS Daponte and MD Fox, Fractal feature analysis and classification in medical imaging, IEEE Trans Med Imag 8 (1989) (2), pp. 133Ð142.
*/

void ImageMatrix::fractal2D(int bins, double *output)
{
	int x, y, k, bin = 0;
	int K = min(width, height) / 5;
	int step = (long)floor((double)(K / bins));
	if (step<1) step = 1;   /* avoid an infinite loop if the image is small */
	for (k = 1; k<K; k = k + step)
	{
		double sum = 0.0;
		for (x = 0; x<width; x++)
			for (y = 0; y<height - k; y++)
				sum += fabs(pixel(x, y, 0).intensity - pixel(x, y + k, 0).intensity);
		for (x = 0; x<width - k; x++)
			for (y = 0; y<height; y++)
				sum += fabs(pixel(x, y, 0).intensity - pixel(x + k, y, 0).intensity);
		if (bin<bins) 
			output[bin++] = sum / (width*(width - k) + height*(height - k));
	}
}


#pragma package(smart_init)




