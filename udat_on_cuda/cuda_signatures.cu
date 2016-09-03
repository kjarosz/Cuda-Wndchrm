#include <fstream>
#include <sstream>



#include "textures/zernike/zernike.h"
#include "textures/haralick/haralick.h"
#include "utils/DirectoryListing.h"
#include "histogram.h"
#include "cuda_signatures.h"



std::vector<ClassSignatures> compute_signatures(char *root_dir, char **directories, int count)
{
  std::vector<ClassSignatures> class_signatures;
  for(int i = 0; i < count; i++)
  {
    std::stringstream class_dir;
    class_dir << root_dir << "\\" << directories[i];

    ClassSignatures signatures;
    signatures.class_name = std::string(directories[i]);
    signatures.signatures = compute_class_signatures(class_dir.str());
    class_signatures.push_back(signatures);
  }
  return class_signatures;
}



std::vector<Signature> compute_class_signatures(std::string class_dir)
{
  DirectoryListing *directory_listing = new DirectoryListing(class_dir);

  std::vector<ImageMatrix *> images;
  std::vector<Signature> signatures;
  while(get_next_batch(directory_listing, images))
    for(Signature signature: compute_signatures_on_cuda(images))
      signatures.push_back(signature);

  delete directory_listing;

  return signatures;
}



bool get_next_batch(DirectoryListing *listing, std::vector<ImageMatrix *> &images)
{
  for (ImageMatrix *image: images)
    delete image;
  images.clear();

  try 
  {
    while(!batch_is_full(images))
    {
      std::string filename = listing->next_file();
      if (supported_format(filename.c_str()))
      {
        ImageMatrix *image = load_image_matrix(filename.c_str());
        if (image)
          images.push_back(image);
      }
    }
  } 
  catch( OutOfFilesException &exc) 
  {
    if (images.size() == 0)
      return false;
  }

  return true;
}



bool batch_is_full(std::vector<ImageMatrix *> &images)
{
  long bytes_taken = 0;
  for(ImageMatrix *image: images)
    bytes_taken += image->width * image->height * sizeof(pix_data);
  return bytes_taken >= BATCH_SIZE;
}



bool supported_format(char *filename)
{
  int len, period, i;
  len = strlen(filename);
  for(i = len - 1; i > 0; i--) {
    if (filename[i] == '.') {
      period = i;
      break;
    }
  }

  if (period <= 0) 
    return false;

  // TODO Check if this compares the extension correctly.
  if (strstr(filename + period, ".tif") || strstr(filename + period, ".TIF"))
    return true;

  return false;
}



ImageMatrix *load_image_matrix(char *filename)
{
  ImageMatrix *matrix = new ImageMatrix();
  if(!matrix->OpenImage(filename)) {
    delete matrix;
    matrix = 0;
  }
  return matrix;
}



void CUDASignatures::compute_signatures_on_cuda()
{
  printf("Computing signatures for %i images on CUDA:\n", image_matrix_count);

  // Arrange data in RAM
  pix_data **pixels  = new pix_data*[image_matrix_count];
  int *widths  = new int[image_matrix_count];
  int *heights = new int[image_matrix_count];
  int *depths  = new int[image_matrix_count];
  int *bits = new int[image_matrix_count];

  for(int i = 0; i < image_matrix_count; i++)
  {
    widths[i]  = image_matrices[i]->width;
    heights[i] = image_matrices[i]->height;
    depths[i]  = image_matrices[i]->depth;
    bits[i]    = image_matrices[i]->bits;

    int size = widths[i] * heights[i] * depths[i];
    pix_data *pixel_array;
    cudaMalloc(&pixel_array, size * sizeof(pix_data));
    cudaMemcpy(pixel_array, image_matrices[i]->data, size * sizeof(pix_data), cudaMemcpyHostToDevice);
    pixels[i] = pixel_array;
  }

  // Move data from RAM to VRAM
  pix_data **cPixels = 0; 
  int *cWidths = 0, *cHeights = 0, *cDepths = 0, *cBits = 0;

  cudaMalloc(&cPixels,  image_matrix_count * sizeof(pix_data*));
  cudaMalloc(&cWidths,  image_matrix_count * sizeof(int));
  cudaMalloc(&cHeights, image_matrix_count * sizeof(int));
  cudaMalloc(&cDepths,  image_matrix_count * sizeof(int));
  cudaMalloc(&cBits,	image_matrix_count * sizeof(int));

  cudaMemcpy(cWidths,  widths,  image_matrix_count * sizeof(int),       cudaMemcpyHostToDevice);
  cudaMemcpy(cHeights, heights, image_matrix_count * sizeof(int),       cudaMemcpyHostToDevice);
  cudaMemcpy(cDepths,  depths,  image_matrix_count * sizeof(int),       cudaMemcpyHostToDevice);
  cudaMemcpy(cPixels,  pixels,  image_matrix_count * sizeof(pix_data*), cudaMemcpyHostToDevice);
  cudaMemcpy(cBits,	   bits,	image_matrix_count * sizeof(int),		cudaMemcpyHostToDevice);

  signatures.clear();

  double *cOutputs = 0;
  cudaMalloc(&cOutputs, MAX_OUTPUT_SIZE * image_matrix_count * sizeof(double));

  long *cSizes = 0;
  cudaMalloc(&cSizes, image_matrix_count * sizeof(long));

  // Execute the features.
  compute_zernike_on_cuda(cPixels, cWidths, cHeights, cDepths, cOutputs, cSizes);
//  compute_haralick_on_cuda(cPixels, cWidths, cHeights, cDepths, cOutputs, cSizes, cBits);
//  compute_histogram_on_cuda(cPixels, cWidths, cHeights, cDepths, cOutputs, cSizes, cBits);

  cudaFree(cSizes);
  cudaFree(cOutputs);

  for(int i = 0; i < image_matrix_count; i++)
  {
    cudaFree(pixels[i]);
  }
  cudaFree(cPixels);
  cudaFree(cDepths);
  cudaFree(cHeights);
  cudaFree(cWidths);
  cudaFree(cBits);

  delete [] pixels;
  delete [] depths;
  delete [] heights;
  delete [] widths;
  delete [] bits;

  printf("Signatures computed\n");
  printf("============================================================\n");

  empty_matrix_container();
}



void CUDASignatures::compute_zernike_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long *sizes)
{
  printf("Performing Zernike texture analysis\n");
  double *d;
  double *r;

  cudaMalloc(&d, image_matrix_count * sizeof(double));
  cudaMalloc(&r, image_matrix_count * sizeof(double));

  cudaMemset(d, 0, image_matrix_count * sizeof(double));
  cudaMemset(r, 0, image_matrix_count * sizeof(double));

  zernike<<< 1, image_matrix_count >>>(images, widths, heights, depths, 
                                       d, r, outputs, sizes);

  int outs_size = MAX_OUTPUT_SIZE * image_matrix_count;
  double *outs = new double[MAX_OUTPUT_SIZE * image_matrix_count];

  int   sizes_size = image_matrix_count;
  long *lSizes     = new long[image_matrix_count];

  cudaMemcpy(outs, outputs, outs_size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(lSizes, sizes, sizes_size * sizeof(long), cudaMemcpyDeviceToHost);

  char buffer[64];
  for(int i = 0; i < image_matrix_count; i++)
  {
    for(int j = 0; j < lSizes[i]; j++)
    {
      sprintf(buffer, "Zernike bin %i", j);
      double value = outs[i * MAX_OUTPUT_SIZE + j];
      signatures.add_signature(buffer, image_matrices[i]->source_file, value);
    }
  }

  delete [] outs;
  delete [] lSizes;

  cudaFree(r);
  cudaFree(d);
}


//void CUDASignatures::compute_haralick_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long *sizes, int *bits)
//{
//  printf("Performing Haralick texture analysis\n");
//
//	const int cDistances = 0;
//
//	haralick<<< 1, image_matrix_count >>>(images, cDistances, outputs, heights, widths, depths, bits);
//	int outs_size = MAX_OUTPUT_SIZE * image_matrix_count;
//  double *outs = new double[MAX_OUTPUT_SIZE * image_matrix_count];
//
//  int   sizes_size = image_matrix_count;
//  long *lSizes     = new long[image_matrix_count];
//
//  cudaMemcpy(outs, outputs, outs_size * sizeof(double), cudaMemcpyDeviceToHost);
//  cudaMemcpy(lSizes, sizes, sizes_size * sizeof(long), cudaMemcpyDeviceToHost);
//
//  char buffer[64];
//  for(int i = 0; i < image_matrix_count; i++)
//  {
//    for(int j = 0; j < lSizes[i]; j++)
//    {
//      sprintf(buffer, "Haarlick bin %i", j);
//      double value = outs[i * MAX_OUTPUT_SIZE + j];
//      signatures.add_signature(buffer, image_matrices[i]->source_file, value);
//    }
//  }
//
//  delete [] outs;
//  delete [] lSizes;
//}

//void CUDASignatures::compute_histogram_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long *sizes, int *bits)
//{
//  printf("Performing Multiscale Histogram analysis\n");
//
//	multiscalehistogram<<< 1, image_matrix_count >>>(images, outputs, widths, heights, depths, bits);
//
//  int outs_size = MAX_OUTPUT_SIZE * image_matrix_count;
//  double *outs = new double[MAX_OUTPUT_SIZE * image_matrix_count];
//
//  int   sizes_size = image_matrix_count;
//  long *lSizes     = new long[image_matrix_count];
//
//  cudaMemcpy(outs, outputs, outs_size * sizeof(double), cudaMemcpyDeviceToHost);
//  cudaMemcpy(lSizes, sizes, sizes_size * sizeof(long), cudaMemcpyDeviceToHost);
//
//  char buffer[64];
//  for(int i = 0; i < image_matrix_count; i++)
//  {
//    for(int j = 0; j < lSizes[i]; j++)
//    {
//      sprintf(buffer, "Multiscale Histogram bin %i", j);
//      double value = outs[i * MAX_OUTPUT_SIZE + j];
//      signatures.add_signature(buffer, image_matrices[i]->source_file, value);
//    }
//  }
//
//  delete [] outs;
//  delete [] lSizes;
//}



void CUDASignatures::save_in(char *directory)
{
  char buffer[FILENAME_MAX];
  join_paths(buffer, directory, "output.csv");

  std::ofstream output(buffer);
  if (!output.good())
  {
    printf("Failed to open file \"%s\"", buffer);
    return;
  }

  std::vector<std::string> signature_names = signatures.get_sig_names();
  std::vector<std::string> filenames       = signatures.get_filenames();

  output << "filename";
  for(std::string signature: signature_names)
    output << ',' << signature; 
  output << std::endl;

  for(int i = 0; i < filenames.size(); i++)
  {
    output << filenames[i];

    for(int j = 0; j < signature_names.size(); j++) 
      output << ',' << signatures.get_signature(j, i);

    output << std::endl;
  }

  output.flush();
  output.close();
}