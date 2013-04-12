//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_MATRIXMARKETMVINPUTFILE_HPP
#define MOR_MATRIXMARKETMVINPUTFILE_HPP

#include "MOR_MultiVectorInputFile.hpp"

namespace MOR {

class MatrixMarketMVInputFile : public MultiVectorInputFile {
public:
  virtual Teuchos::RCP<Epetra_MultiVector> read(const Epetra_Map &map); // overriden

  explicit MatrixMarketMVInputFile(const std::string &path);
};

} // namespace MOR

#endif /* MOR_MATRIXMARKETMVINPUTFILE_HPP */
