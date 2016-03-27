//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_FILEREDUCEDBASISSOURCE_HPP
#define MOR_FILEREDUCEDBASISSOURCE_HPP

#include "MOR_TruncatedReducedBasisSource.hpp"

#include "Epetra_Map.h"

namespace MOR {

class EpetraMVSourceInputFileProvider {
public:
  /*implicit*/ EpetraMVSourceInputFileProvider(const Epetra_Map &vectorMap);

  Teuchos::RCP<BasicEpetraMVSource> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Epetra_Map vectorMap_;
};


class FileReducedBasisSource : public TruncatedReducedBasisSource<EpetraMVSourceInputFileProvider> {
public:
  explicit FileReducedBasisSource(const Epetra_Map &basisMap);
};

} // namespace MOR

#endif /* MOR_FILEREDUCEDBASISSOURCE_HPP */
