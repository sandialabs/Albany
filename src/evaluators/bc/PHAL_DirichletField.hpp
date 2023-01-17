//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DIRICHLET_FIELD_HPP
#define PHAL_DIRICHLET_FIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "PHAL_IdentityCoordinateFunctionTraits.hpp"

namespace PHAL {
/** \brief BC Dirichlet evaluator for general coordinate functions

*/


// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits>
class DirichletField;

template <typename EvalT, typename Traits>
class DirichletField_Base : public PHAL::DirichletBase<EvalT, Traits> {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    DirichletField_Base(Teuchos::ParameterList& p);

  protected:
    std::string field_name;
};

// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::Residual, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::Residual, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::Jacobian, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::Tangent, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::Tangent, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
//  -- Currently assuming no parameter derivative
// **************************************************************
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;

    /**
    * @brief preEvaluate phase for PHAL::AlbanyTraits::DistParamDeriv EvaluationType.
    *
    * During the computation of the Distributed parameter derivative, for the transpose case, 
    * we use a preEvaluate phase to zero out the Lagrange multipliers associated to degrees of freedom constrained
    * by a Dirichlet boundary condition, unless we are differentiating with repsect to this field, in which case we do not zero it out until the evaluation phase.
    * It is necessary to do so in a pre-evaluate phase to ensure a correct implemetation when 
    * two or more Dirichlet bcs are applied to the same degree of freedom (the last one applied is the one that gets prescribed).
    */
    void preEvaluate(typename Traits::EvalData d);
    
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// HessianVec
//  -- Currently assuming no parameter derivative
// **************************************************************
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::HessianVec, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::HessianVec, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT ScalarT;

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

}

#endif
