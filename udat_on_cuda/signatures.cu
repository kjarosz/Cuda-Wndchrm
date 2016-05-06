#include <cstring>
#include <cstdio>
#include <fstream>



#include "file_manip.h"
#include "textures/zernike/zernike.h"
#include "haarlick.h"
#include "signatures.h"
#include "cuda_runtime.h"



Signatures::Signatures()
: col_len(INIT_SIG_CONTAINER_SIZE), 
  row_len(INIT_MATRIX_CONTAINER_SIZE), 
  col_n(0), row_n(0),
  sigs(new char*[col_len]),
  files(new char*[row_len]),
  values(new double[row_len * col_len])
{
  memset(sigs,  0, col_len * sizeof(char *));
  memset(files, 0, row_len * sizeof(char *));
  std::fill_n(values, row_len * col_len, NAN);
}



Signatures::~Signatures()
{ 
  clear();
  delete [] sigs;
  delete [] files;
  delete [] values;
}



void Signatures::add_signature(const char *sig_name, const char *filename, double value)
{
  int col = get_signature_index(sig_name);
  if(col < 0)
    col = insert_new_signature(sig_name);

  int row = get_filename_index(filename);
  if(row < 0)
    row = insert_new_filename(filename);

  values[row * col_len + col] = value;
}



double Signatures::get_signature(const char *sig_name, const char *filename) const
{
  return get_signature(get_signature_index(sig_name),
                       get_filename_index (filename));
}



double Signatures::get_signature(int col, const char *filename) const
{
  return get_signature(col, get_filename_index(filename));
}



double Signatures::get_signature(const char *sig_name, int row) const
{
  return get_signature(get_signature_index(sig_name), row);
}



double Signatures::get_signature(int col, int row) const
{
  if(col < 0 || col >= col_len || row < 0 || row >= row_len)
    return NAN;

  return values[row * col_len + col];
}



int Signatures::get_signature_index(const char *name) const
{
  return find_in_array(sigs, col_n, name);
}



int Signatures::get_filename_index(const char *name) const
{
  return find_in_array(files, row_n, name);
}



void Signatures::get_sig_name(int col, char *output)
{
  if(col < col_len)
    strcpy(output, sigs[col]);
}



void Signatures::get_file_name(int row, char *output)
{
  if(row < row_len)
    strcpy(output, files[row]);
}



int Signatures::get_sig_count() const
{
  return row_n;
}



int Signatures::get_file_count() const
{
  return col_n;
}



void Signatures::clear()
{
  for(int i = 0; i < col_n; i++)
  {
    delete [] sigs[i];
    sigs[i] = 0;
  }

  for(int i = 0; i < row_n; i++)
  {
    delete [] files[i];
    files[i] = 0;
  }
  
  col_n = row_n = 0;

  std::fill_n(values, row_len * col_len, NAN);
}



std::vector<std::string> Signatures::get_sig_names() const
{
  return get_array_copy(sigs, col_n);
}



std::vector<std::string> Signatures::get_filenames() const
{
  return get_array_copy(files, row_n);
}



int Signatures::find_in_array(char **arr, int len, const char *element) const
{
  for(int i = 0; i < len; i++)
    if(strcmp(arr[i], element) == 0)
      return i;
  return -1;
}



int Signatures::insert_new_signature(const char *name) 
{
  if(col_n >= col_len)
    expand_signature_container();

  // Clear out values in this column
  for(int i = 0; i < row_n; i++)
    values[i * col_len + col_n] = NAN;

  sigs[col_n] = new char[SIGNATURE_NAME_LENGTH];
  strcpy(sigs[col_n], name);

  col_n++;
  return col_n - 1;
}



int Signatures::insert_new_filename(const char *name)
{
  if(row_n >= row_len)
    expand_filename_container();

  // Clear out values in the row
  for(int i = 0; i < col_n; i++)
    values[row_n * col_len + i] = NAN;

  files[row_n] = new char[FILENAME_MAX];
  strcpy(files[row_n], name);

  row_n++;
  return row_n - 1;
}



char **Signatures::expand_array(char **arr, int len, int new_len)
{
  char **new_arr = new char*[new_len];
  memset(new_arr, 0, new_len * sizeof(char *));

  for(int i = 0; i < len; i++) 
    new_arr[i] = arr[i];

  return new_arr;
}



void Signatures::expand_signature_container()
{
  int new_size = col_len * 2;
  char **new_sigs = expand_array(sigs, col_len, new_size);
  delete [] sigs;
  sigs = new_sigs;

  expand_value_array(row_len, new_size);
}



void Signatures::expand_filename_container()
{
  int new_size = row_len * 2;
  char **new_files = expand_array(files, row_len, new_size);
  delete [] files;
  files = new_files;

  expand_value_array(new_size, col_len);
}



void Signatures::expand_value_array(int d_row_len, int d_col_len)
{
  double *new_values = new double[d_row_len * d_col_len];
  std::fill_n(new_values, d_row_len * d_col_len, NAN);
  for(int row = 0; row < row_len; row++)
    for(int col = 0; col < col_len; col++)
      new_values[row * d_col_len + col] = values[row * col_len + col];
  delete [] values;
  col_len = d_col_len;
  row_len = d_row_len;
  values = new_values;
}



inline std::vector<std::string> Signatures::get_array_copy(char **arr, int len) const
{
  std::vector<std::string> arrcopy;
  for (int i = 0; i < len; i++)
    arrcopy.push_back(std::string(arr[i]));
  return arrcopy;
}


