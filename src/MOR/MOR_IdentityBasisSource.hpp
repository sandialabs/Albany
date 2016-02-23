//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_IDENTITYBASISSOURCE_HPP
#define MOR_IDENTITYBASISSOURCE_HPP

#include "MOR_ReducedBasisFactory.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Epetra_Map.h"

namespace MOR {

class IdentityBasisSource : public ReducedBasisSource {
public:
  explicit IdentityBasisSource(const Epetra_Map &basisMap);

  virtual ReducedBasisElements operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Epetra_Map basisMap_;
};

} // namespace MOR

#endif /* MOR_IDENTITYBASISSOURCE_HPP */
