#include <fstream>
#include <sstream>
#include <iostream>



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

    ClassSignatures signatures = compute_class_signatures(class_dir.str());
    class_signatures.push_back(signatures);
  }
  return class_signatures;
}



ClassSignatures compute_class_signatures(std::string class_dir)
{
  DirectoryListing *directory_listing = new DirectoryListing(class_dir);

  std::vector<ImageMatrix *> images;
  ClassSignatures class_signatures;
  class_signatures.class_name = class_dir;
  while(get_next_batch(directory_listing, images))
    for(FileSignatures signature: compute_signatures_on_cuda(images))
      class_signatures.signatures.push_back(signature);

  delete directory_listing;

  return class_signatures;
}



std::vector<FileSignatures> compute_signatures_on_cuda(std::vector<ImageMatrix *> &images)
{
  std::cout << "Computing signatures for " << images.size() << " images on CUDA:" << std::endl;

  std::vector<FileSignatures> signatures;

  CudaImages cuda_images;
  move_images_to_cuda(images, cuda_images);

  // Execute the features.
  merge_signatures(signatures, compute_zernike_on_cuda(images, cuda_images));
//  compute_haralick_on_cuda(cPixels, cWidths, cHeights, cDepths, cOutputs, cSizes, cBits);
//  compute_histogram_on_cuda(cPixels, cWidths, cHeights, cDepths, cOutputs, cSizes, cBits);


  std::cout << "Signatures computed" << std::endl;
  std::cout << "============================================================" << std::endl;

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



void move_images_to_cuda(std::vector<ImageMatrix *> &images, CudaImages &cuda_images)
{
  cuda_images.count = images.size();

  // Arrange data in RAM
  int *widths        = new int[cuda_images.count];
  int *heights       = new int[cuda_images.count];
  int *depths        = new int[cuda_images.count];
  int *bits          = new int[cuda_images.count];
  pix_data **pixels  = new pix_data*[cuda_images.count];

  for(int i = 0; i < cuda_images.count; i++)
  {
    widths[i]  = images[i]->width;
    heights[i] = images[i]->height;
    depths[i]  = images[i]->depth;
    bits[i]    = images[i]->bits;
    pixels[i]  = move_data_to_cuda<pix_data>(images[i]->data, widths[i] * heights[i] * depths[i]);

  }

  // Move data from RAM to VRAM
  cuda_images.pixels  = move_data_to_cuda<pix_data*>(pixels,  cuda_images.count);
  cuda_images.widths  = move_data_to_cuda<int>      (widths,  cuda_images.count);
  cuda_images.heights = move_data_to_cuda<int>      (heights, cuda_images.count);
  cuda_images.depths  = move_data_to_cuda<int>      (depths,  cuda_images.count);
  cuda_images.bits    = move_data_to_cuda<int>      (bits,    cuda_images.count);

  delete [] pixels;
  delete [] depths;
  delete [] heights;
  delete [] widths;
  delete [] bits;
}



void delete_cuda_images(CudaImages &cuda_images)
{
  for(int i = 0; i < cuda_images.count; i++)
  {
    cudaFree(cuda_images.pixels[i]);
  }
  cudaFree(cuda_images.pixels);
  cudaFree(cuda_images.depths);
  cudaFree(cuda_images.heights);
  cudaFree(cuda_images.widths);
  cudaFree(cuda_images.bits);

  memset(&cuda_images, 0, sizeof(CudaImages));
}



template <class T>
T* move_data_to_cuda(T *data, int size)
{
  T *cuda_data;
  cudaMalloc(&cuda_data, size * sizeof(T));
  cudaMemcpy(cuda_data, data, size * sizeof(T), cudaMemcpyHostToDevice);
  return cuda_data;
}



std::vector<FileSignatures> &merge_signatures(std::vector<FileSignatures> &dst,
                                              std::vector<FileSignatures> &src)
{
  for(FileSignatures src_signatures: src)
  {
    bool found = false;
    for(int i = 0; i < dst.size() && !found; i++)
    {
      if (src_signatures.file_name == dst[i].file_name) {
        found = true;
        for (Signature sig: src_signatures.signatures)
          dst[i].signatures.push_back(sig);
      }
    }
    if(!found)
      dst.push_back(src_signatures);
  }
  return dst;
}



std::vector<FileSignatures> compute_zernike_on_cuda(const std::vector<ImageMatrix *> &images, CudaImages &cuda_images)
{
  std::cout << "Performing Zernike texture analysis" << std::endl;

  ZernikeData zernike_data = cuda_allocate_zernike_data(cuda_images);
  mb_zernike2D<<< 1, cuda_images.count >>>(cuda_images, zernike_data);
  std::vector<FileSignatures> signatures = cuda_get_zernike_signatures(zernike_data, cuda_images.count);
  cuda_delete_zernike_data(zernike_data, cuda_images.count);
  return signatures;
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



//void CUDASignatures::save_in(char *directory)
//{
//  char buffer[FILENAME_MAX];
//  join_paths(buffer, directory, "output.csv");
//
//  std::ofstream output(buffer);
//  if (!output.good())
//  {
//    printf("Failed to open file \"%s\"", buffer);
//    return;
//  }
//
//  std::vector<std::string> signature_names = signatures.get_sig_names();
//  std::vector<std::string> filenames       = signatures.get_filenames();
//
//  output << "filename";
//  for(std::string signature: signature_names)
//    output << ',' << signature; 
//  output << std::endl;
//
//  for(int i = 0; i < filenames.size(); i++)
//  {
//    output << filenames[i];
//
//    for(int j = 0; j < signature_names.size(); j++) 
//      output << ',' << signatures.get_signature(j, i);
//
//    output << std::endl;
//  }
//
//  output.flush();
//  output.close();
//}