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
#include <algorithm>

#include "image_matrix.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#ifndef BORLAND_C
#ifndef VISUAL_C
#include <stdlib.h>
#include <string.h>
#endif
#endif

#include "../libs/libtiff/tiffio.h"

#ifdef BORLAND_C
#include "../libs/libtiff/tiffio.h"
#include <jpeg.hpp>
#endif


#ifdef VISUAL_C
#include "../libs/libtiff/tiffio.h"
#include <stdlib.h>
#endif

#define SOUND_FILES
#ifndef WIN32
#ifndef VISUAL_C
#define __int64 uint64_t
#endif
#endif

#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)

/* ***************************************************************************** */
    __host__ __device__ pix_data get_pixel(pix_data *pixels, 
                                          int width, int height,
                                          int x, int y, int z )
/* ***************************************************************************** */
{
  return pixels[z * width * height + y * width + x];
}



/* ***************************************************************************** */
    __host__ __device__ void set_pixel(pix_data *pixels, 
                                       int width, int height, 
                                       int x, int y, int z, 
                                       pix_data &new_pixel)
/* ***************************************************************************** */
{
  pixels[z * width * height + y * width + x] = new_pixel;
}



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


TColor RGB2COLOR(RGBcolor rgb)
{  return((TColor)(rgb.blue*65536+rgb.green*256+rgb.red));
}

double COLOR2GRAY(TColor color1)
{  double r,g,b;

   r=(byte)(color1 & 0xFF);
   g=(byte)((color1 & 0xFF00)>>8);
   b=(byte)((color1 & 0xFF0000)>>16);

   return((0.3*r+0.59*g+0.11*b));
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



/* LoadTIFF
   filename -char *- full path to the image file
*/
int ImageMatrix::LoadTIFF(const char *filename)
{
//#ifndef BORLAND_C
   unsigned long h, w, x, y, z;   /* (originally it's tdir_t) */
   unsigned short int spp,bps;
   TIFF *tif = NULL;
   //tdata_t buf;
   unsigned char *buf8=NULL;
   unsigned short *buf16=NULL;
   float *buf_float=NULL;
   double max_val;
   pix_data pix;
   if (tif = TIFFOpen(filename, "r"))
   {
     TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
     width = w;
     TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
     height = h;
     TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
     bits=bps;
	 if (bits != 8 && bits != 16 && bits != 32) { printf("Unsupported bits per pixel (%d) in file '%s'\n", bits, filename); TIFFClose(tif);  return(0); }  /* unsupported numbers of bits per pixel */
     TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
     if (!spp) spp=1;  /* assume one sample per pixel if nothing is specified */
	 if ((depth = TIFFNumberOfDirectories(tif)) < 0) {TIFFClose(tif); return(0); }   /* get the number of slices (Zs) */
     
     /* allocate the data */
     data=new pix_data[width*height*depth];
	 if (!data) { TIFFClose(tif); return(0); } /* memory allocation failed */

     max_val=pow(2.0,(double)bits)-1;
     if (bps==32 && spp==1) max_val=1.0;  /* max value of a floaiting point image */
     /* read TIFF header and determine image size */
     if (bits==8) buf8 = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(tif)*spp);
     if (bits==16) buf16 = (unsigned short *)_TIFFmalloc(TIFFScanlineSize(tif)*sizeof(unsigned short)*spp);
     if (bits==32) buf_float = (float *)_TIFFmalloc(TIFFScanlineSize(tif)*sizeof(float)*spp);	 
     for (z=0;z<(unsigned long)depth;z++)
	 {  TIFFSetDirectory(tif,z);
        for (y = 0; y < (unsigned long)height; y++)
        {   int col;
            if (bits==8) TIFFReadScanline(tif, buf8, y);
            if (bits==16) TIFFReadScanline(tif, buf16, y);
            if (bits==32) TIFFReadScanline(tif, buf_float, y);			
            x=0;col=0;
            while (x<(unsigned long)width)
            { unsigned char byte_data;
              unsigned short short_data;
              float float_data;
              double val;
              int sample_index;
              for (sample_index=0;sample_index<spp;sample_index++)
              {  if (bits==8)
                 {  byte_data=buf8[col+sample_index];
                    val=(double)byte_data;
                 }
                 if (bits==16)
                 {  short_data=buf16[col+sample_index];
                    val=(double)(short_data);
                 }
                 if (bits==32)
                 {  float_data=buf_float[col+sample_index];
                    val=(double)(float_data);
                    if (max_val==1.0) val*=255;   /* change to 0,255 for compatability with algorithms that use integers */
                 }
                 if (spp==3 || spp==4)  /* RGB image or RGB+alpha */
                 {  if (sample_index==0) pix.clr.RGB.red=(unsigned char)(255*(val/max_val));
                    if (sample_index==1) pix.clr.RGB.green=(unsigned char)(255*(val/max_val));
                    if (sample_index==2) pix.clr.RGB.blue=(unsigned char)(255*(val/max_val));
                 }
              }
              if (spp==3 || spp==4) 
              {  pix.intensity=COLOR2GRAY(RGB2COLOR(pix.clr.RGB));     // pix.clr.RGB.red*0.3+pix.clr.RGB.green*0.59+pix.clr.RGB.blue*0.11;
                 if (ColorMode==cmHSV) pix.clr.HSV=RGB2HSV(pix.clr.RGB);			  
              }
              if (spp==1)
              {  pix.clr.RGB.red=(unsigned char)(255*(val/max_val));
                 pix.clr.RGB.green=(unsigned char)(255*(val/max_val));
                 pix.clr.RGB.blue=(unsigned char)(255*(val/max_val));
                 pix.intensity=val;
              }
              set(x,y,z,pix);		   
              x++;
              col+=spp;
            }
        }
     }
     if (spp==3 || spp==4) bits=8;   /* set color images to 8-bits */
     if (buf8) _TIFFfree(buf8);
     if (buf16) _TIFFfree(buf16);
     if (buf_float) _TIFFfree(buf_float);
     TIFFClose(tif);
   }
   else return(0);
//#endif
   return(1);
}



/*
OpenImage - opens an image file and loads its content into the matrix
image_file_name -char *- full path to the image file
downsample -int- a scale to downsample the image by
bounding_rect -rect *- a rect within the image. If specified only that recangle is copied to the matrix
mean -double- a mean value to normalize for.
stddev -double- standard deviation to normalize for.
DynamicRange -long- change to a new dynamic range. Ignore if 0.
*/
int ImageMatrix::OpenImage(const char *image_file_name) //, int downsample, rect *bounding_rect, double mean, double stddev, long DynamicRange, double otsu_mask)
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

  if (res)
  {
    strcpy(source_file, image_file_name);
  }

  /*
	if (res)  // add the image only if it was loaded properly 
	{
		if (DynamicRange) SetDynamicRange(DynamicRange);    // change the dynamic range of the image (if needed)    
		if (bounding_rect && bounding_rect->x >= 0)                    // compute features only from an area of the image     
		{
			ImageMatrix *temp;
			temp = new ImageMatrix(this, bounding_rect->x, bounding_rect->y, bounding_rect->x + bounding_rect->w - 1, bounding_rect->y + bounding_rect->h - 1, 0, depth - 1);
			delete data;
			width = temp->width; height = temp->height;
			if (!(data = new pix_data[width*height*depth])) return(0);  // allocate new memory 
			memcpy(data, temp->data, width*height*depth*sizeof(pix_data));
			//         for (int a=0;a<width*height*depth;a++)
			//		   data[a]=temp->data[a];
			delete temp;
		}
		if (downsample>0 && downsample<100)  // downsample by a given factor 
			Downsample(((double)downsample) / 100.0, ((double)downsample) / 100.0);   // downsample the image 
		if (mean>0)  // normalize to a given mean and standard deviation 
			normalize(-1, -1, -1, mean, stddev);
		if (otsu_mask>0) 
      Mask(Otsu()*otsu_mask);    // mask using the otsu threshold 
	}
  */
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
	minx = std::min<int>(0, std::min<int>(x1, std::min<int>(x2, x3)));
	miny = std::min<int>(0, std::min<int>(y1, std::min<int>(y2, y3)));
	maxx = std::max<int>(x1, std::max<int>(x2, x3));
	maxy = std::max<int>(y1, std::max<int>(y2, y3));
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
void ImageMatrix::BasicStatistics(double *mean, double *median, double *std, double *min, double *max, double *hist, int bins)
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



/* get image histogram */
void ImageMatrix::histogram(double *bins,unsigned short bins_num, int imhist)
{  long a;
   double min=INF,max=-INF;
   /* find the minimum and maximum */
   if (imhist==1)    /* similar to the Matlab imhist */
   {  min=0;
      max=pow(2.0,bits)-1;
   }
   else
   {  for (a=0;a<width*height*depth;a++)
      {  if (data[a].intensity>max)
           max=data[a].intensity;
         if (data[a].intensity<min)
           min=data[a].intensity;
      }
   }
   /* initialize the bins */
   for (a=0;a<bins_num;a++)
     bins[a]=0;

   /* build the histogram */
   for (a=0;a<width*height*depth;a++)   
     if (data[a].intensity==max) bins[bins_num-1]+=1;
     else bins[(int)(((data[a].intensity-min)/(max-min))*bins_num)]+=1;
	
   return;
}



//-----------------------------------------------------------------------------------
/* Otsu
   Find otsu threshold
*/
double ImageMatrix::Otsu()
{  long a;
   double hist[256],omega[256],mu[256],sigma_b2[256],maxval=-INF,sum,count;
   double max=pow(2.0,bits)-1;
   histogram(hist,256,1);
   omega[0]=hist[0]/(width*height);
   mu[0]=1*hist[0]/(width*height);
   for (a=1;a<256;a++)
   {  omega[a]=omega[a-1]+hist[a]/(width*height);
      mu[a]=mu[a-1]+(a+1)*hist[a]/(width*height);
   }
   for (a=0;a<256;a++)
   {  if (omega[a]==0 || 1-omega[a]==0) sigma_b2[a]=0;
      else sigma_b2[a]=pow(mu[255]*omega[a]-mu[a],2)/(omega[a]*(1-omega[a]));
      if (sigma_b2[a]>maxval) maxval=sigma_b2[a];
   }
   sum=0.0;
   count=0.0;
   for (a=0;a<256;a++)
     if (sigma_b2[a]==maxval)
     {  sum+=a;
        count++;
     }	 
   return((pow(2.0,bits)/256.0)*((sum/count)/max));
}



//-----------------------------------------------------------------------------------
/*
  Mask
  Set to zero all pixels that are lower than the threshold. 
  The threshold is a value within the interval (0,1) and is a fraction to the dynamic range
*/
void ImageMatrix::Mask(double threshold)
{  double max=pow(2.0,bits)-1;
   for (long a=0;a<width*height*depth;a++)
     if (data[a].intensity<=threshold*max) 
	 {  data[a].intensity=0.0;
        data[a].clr.RGB.red=data[a].clr.RGB.green=data[a].clr.RGB.blue=0;  
     }
//	 else data[a].intensity-=threshold*max;
}



#pragma package(smart_init)
