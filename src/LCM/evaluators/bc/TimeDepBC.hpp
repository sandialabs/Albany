//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TIMEDEPBC_HPP
#define TIMEDEPBC_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include <vector>

namespace LCM {
/** \brief Time Dependent BC Dirichlet evaluator

*/


// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits> class TimeDepBC;

template <typename EvalT, typename Traits>
class TimeDepBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
public:
  typedef typename EvalT::ScalarT ScalarT;
  TimeDepBC_Base(Teuchos::ParameterList& p);
  ScalarT computeVal(RealType time);

protected:
  const int offset;
  std::vector< RealType > timeValues;
  std::vector< RealType > BCValues;
};

// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::Residual,Traits>
  : public TimeDepBC_Base<PHAL::AlbanyTraits::Residual, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::Jacobian,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::Tangent,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG_MP
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::SGResidual,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::SGTangent,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::SGTangent, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Residual
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::MPResidual,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::MPTangent,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::MPTangent, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};
#endif //ALBANY_SG_MP

}

#endif
