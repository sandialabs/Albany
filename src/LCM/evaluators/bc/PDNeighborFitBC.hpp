//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#ifndef PHAL_PDNEIGHBORFIT_HPP
#define PHAL_PDNEIGHBORFIT_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#ifdef ALBANY_EPETRA
#include "Epetra_Vector.h"
#endif

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"


namespace LCM {
/** \brief BC Dirichlet evaluator for general coordinate functions

*/


// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits>
class PDNeighborFitBC;

template <typename EvalT, typename Traits>
class PDNeighborFitBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    PDNeighborFitBC_Base(Teuchos::ParameterList& p);

  protected:

  double perturbDirichlet;
  double timeStep;
};

// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::Residual, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::Residual, Traits> {
  public:
    PDNeighborFitBC(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::Jacobian, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
  public:
    PDNeighborFitBC(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::Tangent, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::Tangent, Traits> {
  public:
    PDNeighborFitBC(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
//  -- Currently assuming no parameter derivative
// **************************************************************
template<typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
  public:
    PDNeighborFitBC(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG
template<typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::SGResidual, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::SGResidual, Traits> {
  public:
    PDNeighborFitBC(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::SGJacobian, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits> {
  public:
    PDNeighborFitBC(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::SGTangent, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::SGTangent, Traits> {
  public:
    PDNeighborFitBC(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};
#endif 
#ifdef ALBANY_ENSEMBLE 

// **************************************************************
// Multi-point Residual
// **************************************************************
template<typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::MPResidual, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::MPResidual, Traits> {
  public:
    PDNeighborFitBC(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::MPJacobian, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits> {
  public:
    PDNeighborFitBC(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::MPTangent, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::MPTangent, Traits> {
  public:
    PDNeighborFitBC(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};
#endif

}

#endif
