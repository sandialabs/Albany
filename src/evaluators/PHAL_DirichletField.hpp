//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#ifndef PHAL_DIRICHLET_FIELD_HPP
#define PHAL_DIRICHLET_FIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Vector.h"
#endif

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
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::SGResidual, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::SGResidual, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::SGJacobian, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::SGJacobian, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::SGTangent, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::SGTangent, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};
#endif 
#ifdef ALBANY_ENSEMBLE 

// **************************************************************
// Multi-point Residual
// **************************************************************
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::MPResidual, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::MPResidual, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::MPJacobian, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::MPJacobian, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class DirichletField<PHAL::AlbanyTraits::MPTangent, Traits>
    : public DirichletField_Base<PHAL::AlbanyTraits::MPTangent, Traits> {
  public:
    DirichletField(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};
#endif

}

#endif
