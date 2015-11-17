//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_EquilibriumConcentrationBC_hpp)
#define LCM_EquilibriumConcentrationBC_hpp

#include <Phalanx_config.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>

#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include <vector>


namespace LCM {
/** \brief Equilibrium Concentration BC Dirichlet evaluator
*/

//------------------------------------------------------------------------------
// Specialization of the DirichletBase class
//
template<typename EvalT, typename Traits> class EquilibriumConcentrationBC;

template <typename EvalT, typename Traits>
class EquilibriumConcentrationBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
public:
  typedef typename EvalT::ScalarT ScalarT;
  EquilibriumConcentrationBC_Base(Teuchos::ParameterList& p);
  void computeBCs(ScalarT& pressure, ScalarT& Cval);

  RealType term1_, term2_;

protected:
  const int offset_;
};

//------------------------------------------------------------------------------
// Residual
//
template<typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::Residual,Traits>
  : public EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Residual, Traits> {
public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Jacobian
//
template<typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::Jacobian,Traits>
   : public EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Tangent
//
template<typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::Tangent,Traits>
   : public EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Distributed Parameter Derivative
//
template<typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Stochastic Galerkin Residual
//

#ifdef ALBANY_SG
template<typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGResidual,Traits>
   : public EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Stochastic Galerkin Jacobian
//
template<typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Stochastic Galerkin Tangent
//
template<typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGTangent,Traits>
   : public EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::SGTangent, Traits> {
public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};
#endif 
#ifdef ALBANY_ENSEMBLE 

//------------------------------------------------------------------------------
// Multi-point Residual
//
template<typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPResidual,Traits>
   : public EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Multi-point Jacobian
//
template<typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Multi-point Tangent
//
template<typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPTangent,Traits>
   : public EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::MPTangent, Traits> {
public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};
#endif

}

#endif
