//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

//
// Generic Template Code for Constructor and PostRegistrationSetup
//

namespace LCM {

//
//
//
template<typename EvalT, typename Traits>
StrongDBC_Base<EvalT, Traits>::
StrongDBC_Base(Teuchos::ParameterList & p) :
    PHAL::DirichletBase<EvalT, Traits>(p)
{
  return;
}

//
// Specialization: Residual
//
template<typename Traits>
StrongDBC<PHAL::AlbanyTraits::Residual, Traits>::
StrongDBC(Teuchos::ParameterList & p) :
    StrongDBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void
StrongDBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: Jacobian
//
template<typename Traits>
StrongDBC<PHAL::AlbanyTraits::Jacobian, Traits>::
StrongDBC(Teuchos::ParameterList & p) :
    StrongDBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void StrongDBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: Tangent
//
template<typename Traits>
StrongDBC<PHAL::AlbanyTraits::Tangent, Traits>::
StrongDBC(Teuchos::ParameterList & p) :
    StrongDBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void StrongDBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: DistParamDeriv
//
template<typename Traits>
StrongDBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
StrongDBC(Teuchos::ParameterList & p) :
    StrongDBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void StrongDBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

} // namespace LCM
