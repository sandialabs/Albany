//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Application.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Intrepid_MiniTensor.h"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

//define DEBUG_LCM_SCHWARZ

//
// Generic Template Code for Constructor and PostRegistrationSetup
//

namespace LCM {

template <typename EvalT, typename Traits>
SchwarzBC_Base<EvalT, Traits>::
SchwarzBC_Base(Teuchos::ParameterList & p) :
  PHAL::DirichletBase<EvalT, Traits>(p),
  app_(p.get<Teuchos::RCP<Albany::Application>>("Application", Teuchos::null)),
  coupled_app_name_(p.get<std::string>("Coupled Application", "self")),
  coupled_block_name_(p.get<std::string>("Coupled Block")),
  disc_(Teuchos::null)
{
  std::string const &
  nodeset_name = this->nodeSetID;

  app_->setCoupledAppBlockNodeset(
      coupled_app_name_,
      coupled_block_name_,
      nodeset_name);
}

//
// Specialization: Residual
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
  SchwarzBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

//
//
//
template<typename Traits>
void
SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: Jacobian
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
  SchwarzBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: Tangent
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Tangent, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
  SchwarzBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: DistParamDeriv
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
  SchwarzBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: Stochastic Galerkin Residual
//
#ifdef ALBANY_SG_MP
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::SGResidual, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: Stochastic Galerkin Jacobian
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: Stochastic Galerkin Tangent
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::SGTangent, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: Multi-point Residual
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::MPResidual, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: Multi-point Jacobian
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: Multi-point Tangent
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::MPTangent, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}
#endif //ALBANY_SG_MP

}
 // namespace LCM

