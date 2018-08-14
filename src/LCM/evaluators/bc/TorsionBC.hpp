//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TORSIONBC_HPP
#define TORSIONBC_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace LCM {
/** \brief Torsion BC Dirichlet evaluator

*/

// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template <typename EvalT, typename Traits>
class TorsionBC;

template <typename EvalT, typename Traits>
class TorsionBC_Base : public PHAL::DirichletBase<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;
  TorsionBC_Base(Teuchos::ParameterList& p);
  void
  computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval, const RealType time);

  RealType thetaDot;
  RealType X0;
  RealType Y0;
};

// **************************************************************
// Residual
// **************************************************************
template <typename Traits>
class TorsionBC<PHAL::AlbanyTraits::Residual, Traits>
    : public TorsionBC_Base<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template <typename Traits>
class TorsionBC<PHAL::AlbanyTraits::Jacobian, Traits>
    : public TorsionBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template <typename Traits>
class TorsionBC<PHAL::AlbanyTraits::Tangent, Traits>
    : public TorsionBC_Base<PHAL::AlbanyTraits::Tangent, Traits>
{
 public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template <typename Traits>
class TorsionBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public TorsionBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>
{
 public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif
