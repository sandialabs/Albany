//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TORSIONBC_HPP
#define TORSIONBC_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"

namespace LCM {
/** \brief Torsion BC Dirichlet evaluator

*/


// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits> class TorsionBC;

template <typename EvalT, typename Traits>
class TorsionBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
public:
  typedef typename EvalT::ScalarT ScalarT;
  TorsionBC_Base(Teuchos::ParameterList& p);
  void computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval,
                  const RealType time);

  RealType thetaDot;
  RealType X0;
  RealType Y0;
};

// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::Residual,Traits>
  : public TorsionBC_Base<PHAL::AlbanyTraits::Residual, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::Jacobian,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::Tangent,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG_MP
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::SGResidual,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::SGTangent,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::SGTangent, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Residual
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::MPResidual,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::MPTangent,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::MPTangent, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};
#endif //ALBANY_SG_MP

}

#endif
