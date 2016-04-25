#ifndef SIGNATURES_H
#define SIGNATURES_H



#include "image_matrix.h"



class CUDASignatures
{
public:
  CUDASignatures();

  void compute(ImageMatrix *images, int size);

private:
  
  
private:
  void move_images_to_gpu(ImageMatrix *images, int size);
  
};



#endif // SIGNATURES_H