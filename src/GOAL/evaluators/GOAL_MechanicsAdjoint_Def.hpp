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
    const Albany::MeshSpecsStruct* mesh_specs) :
  wBF_(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
  BF_(p.get<std::string>("BF Name"), dl->node_qp_scalar)
{
  std::cout << "MECHANICS ADJOINT: in constructor" << std::endl;
}

template<typename EvalT, typename Traits>
void MechanicsAdjointBase<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  std::cout << "MECHANICS ADJOINT: in postregistration" << std::endl;
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
  std::cout << "MECHANICS ADJOINT: in postregistration" << std::endl;
}

template<typename Traits>
void MechanicsAdjoint<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate (typename Traits::PreEvalData workset)
{
  std::cout << "MECHANICS ADJOINT: in preevaluate" << std::endl;
}

template<typename Traits>
void MechanicsAdjoint<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  std::cout << "MECHANICS ADJOINT: in evaluate" << std::endl;
}

template<typename Traits>
void MechanicsAdjoint<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate (typename Traits::PostEvalData workset)
{
  std::cout << "MECHANICS ADJOINT: in postevaluate" << std::endl;
}

}
