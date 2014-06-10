//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DIRICHLET_COORDFUNC_HPP
#define PHAL_DIRICHLET_COORDFUNC_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

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

template<typename EvalT, typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<EvalT> >
class DirichletCoordFunction;

template <typename EvalT, typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<EvalT> >
class DirichletCoordFunction_Base : public PHAL::DirichletBase<EvalT, Traits> {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    DirichletCoordFunction_Base(Teuchos::ParameterList& p);

    //! Type of traits class being used
    typedef cfunc_traits cfunc_traits_type;

    cfunc_traits_type func;

};

// **************************************************************
// Residual
// **************************************************************
//template<typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<PHAL::AlbanyTraits::Residual> >
template<typename Traits, typename cfunc_traits >
class DirichletCoordFunction<PHAL::AlbanyTraits::Residual, Traits, cfunc_traits>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::Residual, Traits, cfunc_traits> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
//template<typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<PHAL::AlbanyTraits::Jacobian> >
template<typename Traits, typename cfunc_traits>
class DirichletCoordFunction<PHAL::AlbanyTraits::Jacobian, Traits, cfunc_traits>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::Jacobian, Traits, cfunc_traits> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
//template<typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<PHAL::AlbanyTraits::Tangent> >
template<typename Traits, typename cfunc_traits>
class DirichletCoordFunction<PHAL::AlbanyTraits::Tangent, Traits, cfunc_traits>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::Tangent, Traits, cfunc_traits> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
//  -- Currently assuming no parameter derivative
// **************************************************************
//template<typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<PHAL::AlbanyTraits::Tangent> >
template<typename Traits, typename cfunc_traits>
class DirichletCoordFunction<PHAL::AlbanyTraits::DistParamDeriv, Traits, cfunc_traits>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits, cfunc_traits> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG_MP
//template<typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<PHAL::AlbanyTraits::SGResidual> >
template<typename Traits, typename cfunc_traits>
class DirichletCoordFunction<PHAL::AlbanyTraits::SGResidual, Traits, cfunc_traits>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGResidual, Traits, cfunc_traits> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
//template<typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<PHAL::AlbanyTraits::SGJacobian> >
template<typename Traits, typename cfunc_traits>
class DirichletCoordFunction<PHAL::AlbanyTraits::SGJacobian, Traits, cfunc_traits>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGJacobian, Traits, cfunc_traits> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
//template<typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<PHAL::AlbanyTraits::SGTangent> >
template<typename Traits, typename cfunc_traits>
class DirichletCoordFunction<PHAL::AlbanyTraits::SGTangent, Traits, cfunc_traits>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGTangent, Traits, cfunc_traits> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Residual
// **************************************************************
//template<typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<PHAL::AlbanyTraits::MPResidual> >
template<typename Traits, typename cfunc_traits>
class DirichletCoordFunction<PHAL::AlbanyTraits::MPResidual, Traits, cfunc_traits>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPResidual, Traits, cfunc_traits> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
//template<typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<PHAL::AlbanyTraits::MPJacobian> >
template<typename Traits, typename cfunc_traits>
class DirichletCoordFunction<PHAL::AlbanyTraits::MPJacobian, Traits, cfunc_traits>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPJacobian, Traits, cfunc_traits> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
//template<typename Traits, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<PHAL::AlbanyTraits::MPTangent> >
template<typename Traits, typename cfunc_traits>
class DirichletCoordFunction<PHAL::AlbanyTraits::MPTangent, Traits, cfunc_traits>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPTangent, Traits, cfunc_traits> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};
#endif //ALBANY_SG_MP

}

#endif
