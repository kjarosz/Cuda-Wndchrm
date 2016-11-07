#ifndef SIGNATURES_H
#define SIGNATURES_H


#include <string>
#include <vector>

#include "constants.h"


struct Signature
{
  std::string signature_name;
  double      value;
};

struct FileSignatures
{
  std::string            file_name;
  std::vector<Signature> signatures;
};

struct ClassSignatures
{
  std::string                 class_name;
  std::vector<FileSignatures> signatures;
};


#endif // SIGNATURES_H