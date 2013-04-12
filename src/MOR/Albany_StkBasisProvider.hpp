//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_STKBASISPROVIDER_HPP
#define ALBANY_STKBASISPROVIDER_HPP

#include "MOR_ReducedBasisFactory.hpp"

namespace Albany {

class STKDiscretization;

class StkBasisProvider : public MOR::ReducedBasisFactory::BasisProvider {
public:
  explicit StkBasisProvider(const Teuchos::RCP<STKDiscretization> &disc);

  virtual Teuchos::RCP<Epetra_MultiVector> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Teuchos::RCP<STKDiscretization> disc_;
};

} // end namepsace Albany

#endif /* ALBANY_STKBASISPROVIDER_HPP */

