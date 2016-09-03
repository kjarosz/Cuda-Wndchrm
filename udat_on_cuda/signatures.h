#ifndef SIGNATURES_H
#define SIGNATURES_H


#include <string>
#include <vector>

#include "constants.h"


struct Signature
{
  std::string signature_name;
  std::string file_name;
  double      value;
};

struct ClassSignatures
{
  std::string            class_name;
  std::vector<Signature> signatures;
};


#endif // SIGNATURES_H