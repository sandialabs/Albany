//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ConstitutiveModelDriver<EvalT, Traits>::
ConstitutiveModelDriver(Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setName("ConstitutiveModelDriver" + PHX::TypeString<EvalT>::value);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelDriver<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelDriver<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

//------------------------------------------------------------------------------

}

