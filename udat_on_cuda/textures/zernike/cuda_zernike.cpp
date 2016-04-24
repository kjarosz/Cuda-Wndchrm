/****************************************************************************/
/*                                                                          */
/*                                                                          */
/*                              mb_Znl.c                                    */
/*                                                                          */
/*                                                                          */
/*                           Michael Boland                                 */
/*                            09 Dec 1998                                   */
/*                                                                          */     
/*  Revisions:                                                              */
/*  9-1-04 Tom Macura <tmacura@nih.gov> modified to make the code ANSI C    */
/*         and work with included complex arithmetic library from           */
/*         Numerical Recepies in C instead of using the system's C++ STL    */
/*         Libraries.                                                       */
/*                                                                          */
/*  1-29-06 Lior Shamir <shamirl (-at-) mail.nih.gov> modified "factorial"  */
/*  to a loop, replaced input structure with ImageMatrix class.             */
/****************************************************************************/


//---------------------------------------------------------------------------

#pragma hdrstop

#include <math.h>

#include "../../cmatrix.h"
#include "complex.h"

#include "cuda_zernike.h"

//---------------------------------------------------------------------------






void mb_zernike2D(ImageMatrix *I, double D, double R, double *zvalues, long *output_size)
{  
   int rows,cols,n,l;
   double *Y,*X,*P,psum;
   double moment10,moment00,moment01;
   int x,y,size,size2,size3,a;

   if (D<=0) D=15;
   if (R<=0)
   {  rows=I->height;
      cols=I->width;
      R=rows/2;
   }
   Y=new double[rows*cols];
   X=new double[rows*cols];
   P=new double[rows*cols];

   /* Find all non-zero pixel coordinates and values */
   size=0;
   psum=0;
   for (y=0;y<rows;y++)
     for (x=0;x<cols;x++)
     if (I->pixel(x,y,0).intensity!=0)
     {  Y[size]=y+1;
        X[size]=x+1;
        P[size]=I->pixel(x,y,0).intensity;
        psum+=I->pixel(x,y,0).intensity;
        size++;
     }

   /* Normalize the coordinates to the center of mass and normalize
      pixel distances using the maximum radius argument (R) */
   moment10=mb_imgmoments(I,1,0);
   moment00=mb_imgmoments(I,0,0);
   moment01=mb_imgmoments(I,0,1);
   size2=0;
   for (a=0;a<size;a++)
   {  X[size2]=(X[a]-moment10/moment00)/R;
      Y[size2]=(Y[a]-moment01/moment00)/R;
      P[size2]=P[a]/psum;
      if (sqrt(X[size2]*X[size2]+Y[size2]*Y[size2])<=1) size2=size2++;
   }

   size3=0;
   for (n=0;n<=D;n++)
     for (l=0;l<=n;l++)
       if (((n-l) % 2) ==0)
       {  double preal,pimag;
          mb_Znl(n,l,X,Y,P,size2,&preal,&pimag);
          zvalues[size3++]=fabs(sqrt(preal*preal+pimag*pimag));
       }
   *output_size=size3;

   delete Y;
   delete X;
   delete P;

}



#pragma package(smart_init)


