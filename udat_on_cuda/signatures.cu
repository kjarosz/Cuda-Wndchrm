#include "signatures.h"
#include "cuda_runtime.h"

CUDASignatures::CUDASignatures()
{ }

void CUDASignatures::compute(ImageMatrix *images, int size)
{
}

void CUDASignatures::move_images_to_gpu(ImageMatrix *images, int size);