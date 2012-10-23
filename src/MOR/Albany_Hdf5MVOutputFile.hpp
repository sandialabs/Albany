//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_HDF5MVOUTPUTFILE_HPP
#define ALBANY_HDF5MVOUTPUTFILE_HPP

#include "Albany_MultiVectorOutputFile.hpp"

namespace Albany {

class Hdf5MVOutputFile : public MultiVectorOutputFile {
public:
  Hdf5MVOutputFile(const std::string &path, const std::string &groupName);

  virtual void write(const Epetra_MultiVector &mv); // overriden

private:
  std::string groupName_;
};

} // end namespace Albany

#endif /* ALBANY_HDF5MVOUTPUTFILE_HPP */
