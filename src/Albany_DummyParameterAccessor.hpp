//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef ALBANY_DUMMYPARAMETERACCESSOR_H
#define ALBANY_DUMMYPARAMETERACCESSOR_H

#include <string>
#include "PHAL_AlbanyTraits.hpp"
#include "Sacado_ParameterAccessor.hpp"

// This dummy function allows us to register parameters for
// all evaluation types. This is needed for sensitivities with
// respect to shapeParams, where the parameters are only 
// accessed through the Residual fill type (getValue method in
// Albany_Application). These dummy accessors are created for
// other evaluation types, so that Sacado ParamLib has somewhere
// to assign them. But, they are never used.

namespace Albany {

template<typename EvalT, typename Traits>
class DummyParameterAccessor :
     public Sacado::ParameterAccessor<EvalT, Traits> {

   public:
     DummyParameterAccessor() { dummy = 0.0;};
     typename EvalT::ScalarT& getValue(const std::string &name)
     { return dummy;};
   private:
     typename EvalT::ScalarT dummy;
  };
}
#endif
