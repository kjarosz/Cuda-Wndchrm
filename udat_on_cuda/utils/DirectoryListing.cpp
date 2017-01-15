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

#include "DirectoryListing.h"

#include "Utils.h"
#include <stdexcept>
#include <sstream>



DirectoryListing::DirectoryListing()
:m_directory(0), m_subdirectory(0)
{}



DirectoryListing::DirectoryListing(std::string directory)
:m_subdirectory(0)
{
  open_directory(directory);
}



DirectoryListing::~DirectoryListing()
{
  delete m_subdirectory;
}



void DirectoryListing::open_directory(std::string directory)
{
  m_directory = directory;
  read_directory();
  reset();
}



void DirectoryListing::reset()
{
  m_iterator = m_files.begin();
}



std::string DirectoryListing::next_file()
{
  if (m_subdirectory)
  {
    try
    {
      return m_subdirectory->next_file();
    }
    catch( OutOfFilesException &exc )
    {
      delete m_subdirectory;
      m_subdirectory = 0;
      return next_file();
    }
  } else {
    if (m_iterator == m_files.end())
      throw OutOfFilesException(m_directory);

    auto find_data = *m_iterator;
    m_iterator++;

    int i = 0;
    char filename[256];
    while(find_data.cFileName[i])
    {
      filename[i] = (char)find_data.cFileName[i];
      i++;
    }
    filename[i] = '\0';

    std::stringstream path;
    path << m_directory << "\\" << filename;
    if ( find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY )
    {
      m_subdirectory = new DirectoryListing(path.str());
      return next_file();
    }
    return path.str();
  }
}



void add_to_list(std::list<WIN32_FIND_DATA> &list, WIN32_FIND_DATA &file_data)
{
  if (wcscmp(file_data.cFileName, L".")  == 0 ||
      wcscmp(file_data.cFileName, L"..") == 0)
    return;

  list.push_back(file_data);
}



void DirectoryListing::read_directory()
{
  WIN32_FIND_DATA file_data;
  HANDLE find_handle;

  m_files.clear();

  std::wstring wdir(m_directory.begin(), m_directory.end());

  std::wstringstream ss;
  ss << wdir << L"\\*";

  find_handle = FindFirstFile(ss.str().c_str(), &file_data);
  if ( find_handle == INVALID_HANDLE_VALUE )
  {
    if ( GetLastError() != ERROR_FILE_NOT_FOUND )
    {
      std::string what = print_windows_error();
      throw std::runtime_error(what);
    }

    return;
  }

  add_to_list(m_files, file_data);

  while( FindNextFile(find_handle, &file_data) )
    add_to_list(m_files, file_data);

  if ( GetLastError() != ERROR_NO_MORE_FILES )
  {
    std::string what = print_windows_error();
    throw std::runtime_error(what);
  }
}
