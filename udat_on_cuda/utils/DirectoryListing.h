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

#ifndef DIRECTORYLISTING_H
#define DIRECTORYLISTING_H



#include <Windows.h>
#include <string>
#include <list>
#include <iterator>
#include <exception>



class OutOfFilesException : public std::exception
{
public:
  OutOfFilesException(std::string source_directory)
    :m_directory(source_directory)
  {}

  const char * what () const throw ()
  {
    return "Out of files";
  }

  std::string directory() const
  {
    return std::string(m_directory);
  }

private:
  std::string m_directory;
};



class DirectoryListing
{
public:
  DirectoryListing();
  DirectoryListing(std::string directory);
  ~DirectoryListing();

  void        open_directory(std::string directory);

  void        reset();
  std::string next_file();


private:
  void        read_directory();



private:
  std::string                                m_directory;
  std::list<WIN32_FIND_DATA>                 m_files;
  std::list<WIN32_FIND_DATA>::const_iterator m_iterator;

  DirectoryListing                          *m_subdirectory;
};



#endif // DIRECTORYLISTING_H
