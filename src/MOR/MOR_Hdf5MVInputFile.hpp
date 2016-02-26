//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_HDF5MVINPUTFILE_HPP
#define MOR_HDF5MVINPUTFILE_HPP

#include "MOR_MultiVectorInputFile.hpp"

namespace MOR {

class Hdf5MVInputFile : public MultiVectorInputFile {
public:
  Hdf5MVInputFile(const std::string &path, const std::string &groupName);

  // Overridden
  virtual int readVectorCount(const Epetra_Comm &comm);
  virtual Teuchos::RCP<Epetra_MultiVector> read(const Epetra_Map &map);

private:
  std::string groupName_;
};

} // namespace MOR

#endif /* MOR_HDF5MVINPUTFILE_HPP */
