//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_MULTIVECTORINPUTFILE_HPP
#define MOR_MULTIVECTORINPUTFILE_HPP

#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"

#include "Teuchos_RCP.hpp"

#include <string>

namespace MOR {

class MultiVectorInputFile {
public:
  std::string path() const { return path_; }

  virtual int readVectorCount(const Epetra_Comm &comm) = 0;
  virtual Teuchos::RCP<Epetra_MultiVector> read(const Epetra_Map &map) = 0;

  virtual ~MultiVectorInputFile();

protected:
  explicit MultiVectorInputFile(const std::string &path);

private:
  std::string path_;

  // Disallow copy and assignment
  MultiVectorInputFile(const MultiVectorInputFile &);
  MultiVectorInputFile &operator=(const MultiVectorInputFile &);
};

inline
MultiVectorInputFile::MultiVectorInputFile(const std::string &path) :
  path_(path)
{
  // Nothing to do
}

inline
MultiVectorInputFile::~MultiVectorInputFile()
{
  // Nothing to do
}

} // namespace MOR

#endif /* MOR_MULTIVECTORINPUTFILE_HPP */
