//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

namespace ANISO {

template<typename EvalT, typename Traits>
AdvectionResidual<EvalT, Traits>::
AdvectionResidual(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) {
}

template<typename EvalT, typename Traits>
void AdvectionResidual<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm) {
}

template<typename EvalT, typename Traits>
void AdvectionResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset) {
}

}
