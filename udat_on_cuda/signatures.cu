#include <cstring>
#include <cstdio>
#include <fstream>



#include "file_manip.h"
#include "textures/zernike/zernike.h"
#include "haarlick.h"
#include "signatures.h"
#include "cuda_runtime.h"



Signatures::~Signatures()
{ 
  clear();
}



void Signatures::add_signature(const char *filename, const char *sig_name, double value)
{
  Signature *sig = 0;
  for(int i = 0; i < signatures.size(); i++)
  {
    if(strcmp(signatures[i]->name, sig_name) == 0)
    {
      sig = signatures[i];
      break;
    }
  }

  if(!sig)
  {
    int sig_name_len = strlen(sig_name);
    sig = new Signature();
    sig->name = new char[sig_name_len + 1];
    strcpy(sig->name, sig_name);
    signatures.push_back(sig);
  }

  bool found = false;
  for(int i = 0; i < sig->filenames.size(); i++)
  {
    if(strcmp(sig->filenames[i], filename) == 0)
    {
      found = true;
      sig->values[i] = value;
    }
  }

  if(!found)
  {
    char *fname = new char[FILENAME_MAX];
    sig->filenames.push_back(fname);
    sig->values.push_back(value);
  }
}



void Signatures::clear()
{
  for (Signature *sig: signatures)
  {
    delete [] sig->name;
    for(char * fname: sig->filenames)
      delete [] fname;
    delete sig;
  }
  signatures.clear();
}



std::vector<std::string> Signatures::get_sig_names()
{
  std::vector<std::string> sig_names;
  for(Signature *sig: signatures)
    sig_names.push_back(std::string(sig->name));
  return sig_names;
}



std::vector<std::string> Signatures::get_filenames()
{
  std::vector<std::string> filenames;
  for(Signature *sig: signatures)
  {
    for(char *fname: sig->filenames)
    {
      bool found = false;
      for(std::string filename: filenames)
      {
        if(filename.compare(fname) == 0)
        {
          found = true;
          break;
        }
      }

      if(!found)
      {
        filenames.push_back(std::string(fname));
      }
    }
  }
  return filenames;
}



double Signatures::get_signature(const char *filename, const char *sig_name)
{
  Signature *sig;
  for(Signature *signature: signatures)
  {
    if(strcmp(signature->name, sig_name) == 0)
    {
      sig = signature;
      break;
    }
  }

  double value = NAN;
  if(sig)
  {
    for(int i = 0; i < sig->filenames.size(); i++) 
    {
      if(strcmp(sig->filenames[i], filename) == 0)
      {
        value = sig->values[i];
        break;
      }
    }
  }
  return value;
}



CUDASignatures::CUDASignatures()
:image_matrices(0), 
 matrix_container_size(INIT_MATRIX_CONTAINER_SIZE),
 image_matrix_count(0)
{ 
  image_matrices = new ImageMatrix*[matrix_container_size];
}



CUDASignatures::~CUDASignatures()
{
  empty_matrix_container();
  delete [] image_matrices;
}



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
  for(int i = 0; i < signature_names.size(); i++)
    output << ',' << signature_names[i]; 
  output << std::endl;

  for(std::string filename: filenames)
  {
    output << filename;

    for(std::string signature_name: signature_names)
    {
      output << ',' << signatures.get_signature(filename.c_str(), signature_name.c_str());
    }

    output << std::endl;
  }

  output.flush();
  output.close();
}



void CUDASignatures::compute(char **directories, int count)
{
  reset_directory_tracker(directories, count);
  while(read_next_batch())
    compute_signatures_on_cuda();
}



bool CUDASignatures::supported_format(char *filename)
{
  if (strstr(filename, ".tif") || strstr(filename, ".TIF"))
    return true;
  return false;
}



void CUDASignatures::reset_directory_tracker(char **directories, int count)
{
  directory_tracker.directories = directories;
  directory_tracker.current_dir = 0;
  directory_tracker.count       = count;
  directory_tracker.opened_dir  = 0;
}



bool CUDASignatures::read_next_batch()
{
  dirent *entry;
  char filename_buffer[FILENAME_MAX];
  while((entry = read_next_entry()) && !batch_capacity_reached())
  {
    if (entry->d_name[0] == '.')
      continue;

    if (!supported_format(entry->d_name))
      continue;

    join_paths(filename_buffer, directory_tracker.directory, entry->d_name);

    printf("Loading image \"%s\"\n", filename_buffer);
    load_image_matrix(filename_buffer);
  }

  return (image_matrix_count > 0);
}


#define MAX_MATRIX_SIZE 1073741824

bool CUDASignatures::batch_capacity_reached()
{
  long bytes_taken = 0;
  for (int i = 0; i < image_matrix_count; i++)
  {
    bytes_taken += image_matrices[i]->width * image_matrices[i]->height * sizeof(pix_data);
  }
  return (bytes_taken < MAX_MATRIX_SIZE);
}



dirent * CUDASignatures::read_next_entry()
{
  dirent *entry = 0;

  bool done = false;
  while(!done && directory_tracker.current_dir < directory_tracker.count)
  { 
    if(!directory_tracker.opened_dir)
    {
      directory_tracker.opened_dir = opendir(
        directory_tracker.directories[directory_tracker.current_dir]);
    }

    entry = readdir(directory_tracker.opened_dir);
    if (!entry)
    {
      done = true;
    } 
    else
    {
      closedir(directory_tracker.opened_dir);
      directory_tracker.current_dir++;
    }
  }

  return entry;
}



void CUDASignatures::load_image_matrix(char *filename)
{
  if(image_matrix_count >= matrix_container_size)
    double_matrix_container();

  ImageMatrix *matrix = new ImageMatrix();
  matrix->OpenImage(filename);
  image_matrices[image_matrix_count++] = matrix;
}



void CUDASignatures::double_matrix_container()
{
  ImageMatrix **new_container = new ImageMatrix*[matrix_container_size * 2];
  memcpy(new_container, image_matrices, matrix_container_size * sizeof(ImageMatrix*));
  delete [] image_matrices;
  image_matrices = new_container;
  matrix_container_size *= 2;
}



void CUDASignatures::empty_matrix_container()
{
  for(int i = 0; i < image_matrix_count; i++) 
    delete image_matrices[i];
  image_matrix_count = 0;
}



void CUDASignatures::compute_signatures_on_cuda()
{
  printf("Computing signatures for %i images on CUDA:\n", image_matrix_count);

  // Arrange data in RAM
  pix_data **pixels  = new pix_data*[image_matrix_count];
  int *widths  = new int[image_matrix_count];
  int *heights = new int[image_matrix_count];
  int *depths  = new int[image_matrix_count];

  for(int i = 0; i < image_matrix_count; i++)
  {
    widths[i]  = image_matrices[i]->width;
    heights[i] = image_matrices[i]->height;
    depths[i]  = image_matrices[i]->depth;

    int size = widths[i] * heights[i] * depths[i];
    pix_data *pixel_array;
    cudaMalloc(&pixel_array, size * sizeof(pix_data));
    cudaMemcpy(pixel_array, image_matrices[i]->data, size * sizeof(pix_data), cudaMemcpyHostToDevice);
    pixels[i] = pixel_array;
  }

  // Move data from RAM to VRAM
  pix_data **cPixels = 0; 
  int *cWidths = 0, *cHeights = 0, *cDepths = 0;

  cudaMalloc(&cPixels,  image_matrix_count * sizeof(pix_data*));
  cudaMalloc(&cWidths,  image_matrix_count * sizeof(int));
  cudaMalloc(&cHeights, image_matrix_count * sizeof(int));
  cudaMalloc(&cDepths,  image_matrix_count * sizeof(int));

  cudaMemcpy(cWidths,  widths,  image_matrix_count * sizeof(int),       cudaMemcpyHostToDevice);
  cudaMemcpy(cHeights, heights, image_matrix_count * sizeof(int),       cudaMemcpyHostToDevice);
  cudaMemcpy(cDepths,  depths,  image_matrix_count * sizeof(int),       cudaMemcpyHostToDevice);
  cudaMemcpy(cPixels,  pixels,  image_matrix_count * sizeof(pix_data*), cudaMemcpyHostToDevice);

  signatures.clear();

  double *cOutputs = 0;
  cudaMalloc(&cOutputs, MAX_OUTPUT_SIZE * image_matrix_count * sizeof(double));

  long *cSizes = 0;
  cudaMalloc(&cSizes, image_matrix_count * sizeof(long));

  // Execute the features.
  printf("Performing Zernike texture analysis\n");
  compute_zernike_on_cuda(cPixels, cWidths, cHeights, cDepths, cOutputs, cSizes);

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

  delete [] pixels;
  delete [] depths;
  delete [] heights;
  delete [] widths;

  printf("Signatures computed\n");
  printf("============================================================\n");

  empty_matrix_container();
}



void CUDASignatures::compute_zernike_on_cuda(pix_data **images, int *widths, int *heights, int *depths, double *outputs, long *sizes)
{
  double *d;
  double *r;

  cudaMalloc(&d, image_matrix_count * sizeof(double));
  cudaMalloc(&r, image_matrix_count * sizeof(double));

  cudaMemset(d, 0, image_matrix_count * sizeof(double));
  cudaMemset(r, 0, image_matrix_count * sizeof(double));

  zernike<<< 1, image_matrix_count >>>(images, widths, heights, depths, 
                                       d, r, outputs, sizes);

  char buffer[64];
  for(int i = 0; i < image_matrix_count; i++)
  {
    for(int j = 0; j < sizes[i]; j++)
    {
      sprintf(buffer, "Zernike bin %i", j);
      signatures.add_signature(image_matrices[i]->source_file, buffer, outputs[MAX_OUTPUT_SIZE * i + j]);
    }
  }

  cudaFree(r);
  cudaFree(d);
}
