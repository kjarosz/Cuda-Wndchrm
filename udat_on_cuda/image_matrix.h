/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*                                                                               */
/* Copyright (C) 2007 Open Microscopy Environment                                */
/*       Massachusetts Institue of Technology,                                   */
/*       National Institutes of Health,                                          */
/*       University of Dundee                                                    */
/*                                                                               */
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
/* Written by:  Lior Shamir <shamirl [at] mail [dot] nih [dot] gov>              */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//---------------------------------------------------------------------------
#ifndef IMAGE_MATRIX_H
#define IMAGE_MATRIX_H
//---------------------------------------------------------------------------

#ifdef BORLAND_C
#include <vcl.h>
#else  
#include "colors/FuzzyCalc.h"
#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) < (b)) ? (b) : (a))
#endif

#define cmRGB 1
#define cmHSV 2

#define INF 10E200


typedef unsigned char byte;

typedef struct RGBCOLOR
{
	byte red, green, blue;
}RGBcolor;

typedef struct HSVCOLOR
{
	byte hue, saturation, value;
}HSVcolor;

typedef union
{
	RGBcolor RGB;
	HSVcolor HSV;
}color;


typedef struct PIX_DATA
{
	color clr;
	double intensity;  /* normailized to (0,255) interval */
} pix_data;

typedef struct
{
	int x, y, w, h;
}
rect;

class ImageMatrix
{
private:
	pix_data *data;                                 /* data of the colors                   */
public:
	int ColorMode;                                  /* can be cmRGB or cmHSV                */
	unsigned short bits;                            /* the number of intensity bits (8,16, etc) */
	int width, height, depth;                         /* width and height of the picture      */
  int LoadTIFF(char *filename);
  /* load an image of any supported format */
	int OpenImage(char *image_file_name, int downsample, 
                rect *bounding_rect, double mean, 
                double stddev, long DynamicRange, 
                double otsu_mask); 

	ImageMatrix();                                  /* basic constructor                    */
	ImageMatrix(int width, int height, int depth);    /* construct a new empty matrix         */
	ImageMatrix(ImageMatrix *matrix, int x1, int y1, int x2, int y2, int z1, int z2);  /* create a new matrix which is part of the original one */
	~ImageMatrix();                                 /* destructor */
	ImageMatrix *duplicate();                       /* create a new identical matrix        */
	pix_data pixel(int x, int y, int z);              /* get a pixel value                    */

	void set(int x, int y, int z, pix_data val);      /* assign a pixel value                 */
	void SetInt(int x, int y, int z, double val);     /* set only the intensity of the pixel  */
	void Initialize();                              /* initialize the image                 */
	void diff(ImageMatrix *matrix);                 /* compute the difference from another image */
	void normalize(double min, double max, long range, double mean, double stddev); /* normalized an image to either min/max or mean/stddev */
	void SetDynamicRange(long dr);                  /* change the dynamic range of an image */
	void paste(ImageMatrix *matrix, long x, long y, long z); /* paste an image into an existing image */
	void flip();                                    /* flip an image horizonatally          */
	void invert();                                  /* invert the intensity of an image     */
	void Downsample(double x_ratio, double y_ratio);/* downsample an image                  */
	void rotate(double angle);                      /* rotate and image                     */
	void convolve(ImageMatrix *filter);
	void BasicStatistics(double *mean, double *median, double *std, double *min, double *max, double *histogram, int bins);
	
  double Otsu();
	void Mask(double threshold);   
  void histogram(double *bins,unsigned short bins_num, int imhist);


};



/* global functions */
HSVcolor RGB2HSV(RGBcolor rgb);
RGBcolor HSV2RGB(HSVcolor hsv);
TColor RGB2COLOR(RGBcolor rgb);
double COLOR2GRAY(TColor color);


#endif // IMAGE_MATRIX_H