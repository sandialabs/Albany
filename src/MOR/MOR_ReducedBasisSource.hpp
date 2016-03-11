//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDBASISSOURCE_HPP
#define MOR_REDUCEDBASISSOURCE_HPP

#include "MOR_ReducedBasisElements.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace MOR {

class ReducedBasisSource {
public:
  virtual ReducedBasisElements operator()(const Teuchos::RCP<Teuchos::ParameterList> &params) = 0;
  virtual ~ReducedBasisSource() {}
};

} // end namepsace Albany

#endif /* MOR_REDUCEDBASISSOURCE_HPP */
