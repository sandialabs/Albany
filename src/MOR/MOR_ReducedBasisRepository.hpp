//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDBASISREPOSITORY_HPP
#define MOR_REDUCEDBASISREPOSITORY_HPP

#include "MOR_ReducedBasisFactory.hpp"

#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <string>
#include <map>

namespace MOR {

class ReducedBasisRepository {
public:
  explicit ReducedBasisRepository(const Teuchos::RCP<ReducedBasisFactory> &factory);

  Teuchos::RCP<const Epetra_Vector> getOrigin(const Teuchos::RCP<Teuchos::ParameterList> &params);
  Teuchos::RCP<const Epetra_MultiVector> getBasis(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Teuchos::RCP<ReducedBasisFactory> factory_;

  typedef std::map<std::string, ReducedBasisElements> InstanceMap;
  InstanceMap instances_;

  ReducedBasisElements getInstance(const Teuchos::RCP<Teuchos::ParameterList> &params);
};

} // end namepsace Albany

#endif /* MOR_REDUCEDBASISREPOSITORY_HPP */


