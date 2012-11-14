//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_MATRIXMARKETMVINPUTFILE_HPP
#define ALBANY_MATRIXMARKETMVINPUTFILE_HPP

#include "Albany_MultiVectorInputFile.hpp"

namespace Albany {

class MatrixMarketMVInputFile : public MultiVectorInputFile {
public:
  virtual Teuchos::RCP<Epetra_MultiVector> read(const Epetra_Map &map); // overriden

  explicit MatrixMarketMVInputFile(const std::string &path);
};

} // end namespace Albany

#endif /* ALBANY_MATRIXMARKETMVINPUTFILE_HPP */
