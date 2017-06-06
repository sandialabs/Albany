//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Application.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "MiniTensor.h"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

#if defined(ALBANY_DTK)
#include "Albany_OrdinarySTKFieldContainer.hpp"
#endif

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
//
//
template<typename EvalT, typename Traits>
template<typename T>
void
StrongDBC_Base<EvalT, Traits>::
computeBCs(size_t const ns_node, T & x_val, T & y_val, T & z_val)
{
  x_val = 0.0;
  y_val = 0.0;
  z_val = 0.0;
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
