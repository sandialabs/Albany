//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DIRICHLET_HPP
#define PHAL_DIRICHLET_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Albany_UnivariateDistribution.hpp"

#include "PHAL_Dimension.hpp"

namespace PHAL {
/** \brief Gathers solution values from the Newton solution vector into
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Generic Template Implementation for constructor and PostReg
// **************************************************************

template<typename EvalT, typename Traits>
class DirichletBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>,
    public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{

private:

  //typedef typename Traits::Residual::ScalarT ScalarT;
  typedef typename EvalT::ScalarT ScalarT;

public:

  DirichletBase(Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  // This function will be overloaded with template specialized code
  void evaluateFields(typename Traits::EvalData d)=0;

  virtual ScalarT& getValue(const std::string &/* n */) { return value; }

protected:
  const int offset;
  ScalarT value;
  std::string nodeSetID;
  bool isRandom;
  Teuchos::RCP<Albany::UnivariatDistribution> distribution;
  PHX::MDField<const ScalarT,Dim>   theta_as_field;
};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits> class Dirichlet;

// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::Residual,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::Residual, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::Jacobian,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::Tangent,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);

  /**
    * @brief preEvaluate phase for PHAL::AlbanyTraits::DistParamDeriv EvaluationType.
    *
    * During the computation of the Distributed parameter derivative, for the transpose case, 
    * we use a preEvaluate phase to zero out the Lagrange multipliers associated to degrees of freedom constrained
    * by a Dirichlet boundary condition. It is necessary to do so in a pre-evaluate phase to ensure a correct implemetation when 
    * two or more Dirichlet bcs are applied to the same degree of freedom (the last one applied is the one that gets prescribed).
  */
  void preEvaluate(typename Traits::EvalData d);
  
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// HessianVec
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::HessianVec,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::HessianVec, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);

  /**
  * @brief preEvaluate phase for PHAL::AlbanyTraits::HessianVec EvaluationType.
  *
  * During the computation of the Hessian-vector product of the residual
  * multiplied by Lagrange multipliers, a preEvaluate phase is used to zero
  * out the Lagrange multipliers associated to degrees of freedom constrained
  * by a Dirichlet boundary condition.
  */
  void preEvaluate(typename Traits::EvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// **************************************************************
// Evaluator to aggregate all Dirichlet BCs into one "field"
// **************************************************************
template<typename EvalT, typename Traits>
class DirichletAggregator
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{
private:

  typedef typename EvalT::ScalarT ScalarT;

public:

  DirichletAggregator(Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  // This function will be overloaded with template specialized code
  void evaluateFields(typename Traits::EvalData /* d */) {};
};

} // namespace PHAL

#endif // PHAL_DIRICHLET_HPP
