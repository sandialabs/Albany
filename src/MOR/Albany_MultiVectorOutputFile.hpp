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

#ifndef ALBANY_MULTIVECTOROUTPUTFILE_HPP
#define ALBANY_MULTIVECTOROUTPUTFILE_HPP

#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"

#include "Teuchos_RCP.hpp"

#include <string>

namespace Albany {

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

} // end namespace Albany

#endif /* ALBANY_MULTIVECTOROUTPUTFILE_HPP */
