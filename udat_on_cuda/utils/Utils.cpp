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

  std::string error(lpMsgBuf);

  LocalFree(lpMsgBuf);

  return error;
}