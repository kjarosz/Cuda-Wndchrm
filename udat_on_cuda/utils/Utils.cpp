/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*                                                                               */
/*    This file is part of Cuda-Wndchrm.                                         */
/*    Copyright (C) 2017 Kamil Jarosz, Christopher K. Horton and Tyler Wiersing  */
/*                                                                               */
/*    This library is free software; you can redistribute it and/or              */
/*    modify it under the terms of the GNU Lesser General Public                 */
/*    License as published by the Free Software Foundation; either               */
/*    version 2.1 of the License, or (at your option) any later version.         */
/*                                                                               */
/*    This library is distributed in the hope that it will be useful,            */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of             */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU          */
/*    Lesser General Public License for more details.                            */
/*                                                                               */
/*    You should have received a copy of the GNU Lesser General Public           */
/*    License along with this library; if not, write to the Free Software        */
/*    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA  */
/*                                                                               */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include "utils.h"

#include <Windows.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

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
