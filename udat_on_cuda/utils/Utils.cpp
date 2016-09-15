#include "utils.h"

#include <Windows.h>
#include <iostream>

std::string print_windows_error()
{
  DWORD flags;
  flags  = FORMAT_MESSAGE_ALLOCATE_BUFFER;
  flags &= FORMAT_MESSAGE_FROM_SYSTEM;
  flags &= FORMAT_MESSAGE_IGNORE_INSERTS;

  LPTSTR lpMsgBuf; 
  FormatMessage(flags, NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), 
    (LPTSTR) &lpMsgBuf, 0, NULL);

  std::cerr << lpMsgBuf << std::endl;

  int msg_len = wcslen(lpMsgBuf);
  char *msg = new char[msg_len+1];
  for(int i = 0; i < msg_len; i++) {
    msg[i] = (char)lpMsgBuf[i];
  }
  msg[msg_len] = '\0';
  std::string error(msg);

  delete [] msg;
  LocalFree(lpMsgBuf);

  return error;
}



void print_cuda_error(cudaError error, std::string message)
{
  std::cout << message << " (" << error << ")"      << std::endl
    << "Name: "        << cudaGetErrorName(error)   << std::endl
    << "Description: " << cudaGetErrorString(error) << std::endl
    << std::endl;
}