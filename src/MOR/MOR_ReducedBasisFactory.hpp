//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDBASISFACTORY_HPP
#define MOR_REDUCEDBASISFACTORY_HPP

#include "Epetra_Vector.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <string>
#include <map>

namespace MOR {

struct ReducedBasisElements {
  /* implicit */ ReducedBasisElements(
      const Teuchos::RCP<Epetra_MultiVector> &basis_in) :
    origin(), basis(basis_in)
  {}

  ReducedBasisElements(
      const Teuchos::RCP<Epetra_Vector> &origin_in,
      const Teuchos::RCP<Epetra_MultiVector> &basis_in) :
    origin(origin_in), basis(basis_in)
  {}

  Teuchos::RCP<Epetra_Vector> origin;
  Teuchos::RCP<Epetra_MultiVector> basis;
};


class ReducedBasisSource {
public:
  virtual ReducedBasisElements operator()(const Teuchos::RCP<Teuchos::ParameterList> &params) = 0;
  virtual ~ReducedBasisSource() {}
};


class ReducedBasisFactory {
public:
  ReducedBasisFactory();

  ReducedBasisElements create(const Teuchos::RCP<Teuchos::ParameterList> &params);

  void extend(const std::string &id, const Teuchos::RCP<ReducedBasisSource> &source);

private:
  typedef std::map<std::string, Teuchos::RCP<ReducedBasisSource> > BasisSourceMap;
  BasisSourceMap sources_;
};

} // end namepsace Albany

#endif /* MOR_REDUCEDBASISFACTORY_HPP */

