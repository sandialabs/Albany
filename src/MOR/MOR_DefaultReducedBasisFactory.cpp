//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_DefaultReducedBasisFactory.hpp"

#include "MOR_IdentityBasisSource.hpp"
#include "MOR_FileReducedBasisSource.hpp"

namespace MOR {

Teuchos::RCP<ReducedBasisFactory> defaultReducedBasisFactoryNew(const Epetra_Map &basisMap)
{
  const Teuchos::RCP<ReducedBasisFactory> result = Teuchos::rcp(new ReducedBasisFactory);

  result->extend("Identity", Teuchos::rcp(new MOR::IdentityBasisSource(basisMap)));
  result->extend("File", Teuchos::rcp(new MOR::FileReducedBasisSource(basisMap)));

  return result;
}

} // end namespace MOR
