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