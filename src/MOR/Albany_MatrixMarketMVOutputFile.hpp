//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_MATRIXMARKETMVOUTPUTFILE_HPP
#define ALBANY_MATRIXMARKETMVOUTPUTFILE_HPP

#include "Albany_MultiVectorOutputFile.hpp"

namespace Albany {

class MatrixMarketMVOutputFile : public MultiVectorOutputFile {
public:
  explicit MatrixMarketMVOutputFile(const std::string &path);

  virtual void write(const Epetra_MultiVector &mv); // overriden
};

} // end namespace Albany

#endif /* ALBANY_MATRIXMARKETMVOUTPUTFILE_HPP */
