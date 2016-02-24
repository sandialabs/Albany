//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_MULTIVECTOROUTPUTFILE_HPP
#define MOR_MULTIVECTOROUTPUTFILE_HPP

#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"

#include "Teuchos_RCP.hpp"

#include <string>

namespace MOR {

class MultiVectorOutputFile {
public:
  std::string path() const { return path_; }

  virtual void write(const Epetra_MultiVector &mv) = 0;

  virtual ~MultiVectorOutputFile();

protected:
  explicit MultiVectorOutputFile(const std::string &path);

private:
  std::string path_;

  // Disallow copy and assignment
  MultiVectorOutputFile(const MultiVectorOutputFile &);
  MultiVectorOutputFile &operator=(const MultiVectorOutputFile &);
};

inline
MultiVectorOutputFile::MultiVectorOutputFile(const std::string &path) :
  path_(path)
{
  // Nothing to do
}

inline
MultiVectorOutputFile::~MultiVectorOutputFile()
{
  // Nothing to do
}

} // namespace MOR

#endif /* MOR_MULTIVECTOROUTPUTFILE_HPP */
