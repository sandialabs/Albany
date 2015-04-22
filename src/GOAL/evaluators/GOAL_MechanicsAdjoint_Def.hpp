//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"

namespace GOAL {

template<typename EvalT, typename Traits>
MechanicsAdjointBase<EvalT, Traits>::
MechanicsAdjointBase (Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl,
    const Albany::MeshSpecsStruct* mesh_specs)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    mesh_specs == NULL, std::logic_error,
    "MechanicsAdjointBase needs access to"
    "mesh_specs->ebName and mesh_specs->sepEvalsByEB");

  // register with the state manager
  this->stateManager_ = p.get<Albany::StateManager*>("State Manager Ptr");

  // create field tag
  fieldTag_ =
    Teuchos::rcp(new PHX::Tag<ScalarT>("Project IP to Nodal Field", dl->dummy));
}

template<typename EvalT, typename Traits>
void MechanicsAdjointBase<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
}

/*************************
  RESIDUAL SPECIALIZATION
**************************/
template<typename Traits>
MechanicsAdjoint<PHAL::AlbanyTraits::Residual, Traits>::
MechanicsAdjoint (
    Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl,
    const Albany::MeshSpecsStruct* mesh_specs) :
  MechanicsAdjointBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl,mesh_specs)
{
}

template<typename Traits>
void MechanicsAdjoint<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate (typename Traits::PreEvalData workset)
{
}

template<typename Traits>
void MechanicsAdjoint<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
}

template<typename Traits>
void MechanicsAdjoint<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate (typename Traits::PostEvalData workset)
{
}

}
