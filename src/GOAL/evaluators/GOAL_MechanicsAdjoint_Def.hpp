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
  /* input
     TODO this should be generalized to include other scatter
     evaluators (e.g. scatter temperature), preferably by
     querying the mechanics problem directly. This is placed
     here, because I believe it is cleaner than injecting code
     directly into the Mechanics problem */
  PHX::Tag<typename EvalT::ScalarT> tag("Scatter",dl->dummy);
  this->addDependentField(tag);

  /* output */
  fieldTag_ =
    Teuchos::rcp(new PHX::Tag<ScalarT>("Mechanics Adjoint", dl->dummy));
  this->addEvaluatedField(*fieldTag_);
  this->setName("MechanicsAdjoint" + PHX::typeAsString<EvalT>());
}

template<typename EvalT, typename Traits>
void MechanicsAdjointBase<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
}

/* Specialization : Residual */
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
