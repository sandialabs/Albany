//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_HDF5MVINPUTFILE_HPP
#define ALBANY_HDF5MVINPUTFILE_HPP

#include "Albany_MultiVectorInputFile.hpp"

namespace Albany {

class Hdf5MVInputFile : public MultiVectorInputFile {
public:
  virtual Teuchos::RCP<Epetra_MultiVector> read(const Epetra_Map &map); // overriden

  Hdf5MVInputFile(const std::string &path, const std::string &groupName);

private:
  std::string groupName_;
};

} // end namespace Albany

#endif /* ALBANY_HDF5MVINPUTFILE_HPP */
