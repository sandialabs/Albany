/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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
