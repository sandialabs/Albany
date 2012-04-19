/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#ifndef ALBANY_MULTIVECTORINPUTFILE_HPP
#define ALBANY_MULTIVECTORINPUTFILE_HPP

#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"

#include "Teuchos_RCP.hpp"

#include <string>

namespace Albany {

class MultiVectorInputFile {
public:
  std::string path() const { return path_; }

  virtual Teuchos::RCP<Epetra_MultiVector> vectorNew(const Epetra_Map &map) = 0;

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

} // end namespace Albany

#endif /* ALBANY_MULTIVECTORINPUTFILE_HPP */
