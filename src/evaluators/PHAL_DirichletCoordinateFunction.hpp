//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#ifndef PHAL_DIRICHLET_COORDFUNC_HPP
#define PHAL_DIRICHLET_COORDFUNC_HPP

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

// After the 3 Nov 2015 Phalanx refactor, the third template parameter doesn't
// work. That isn't actually a problem since one could never use anything except
// IdentityCoordFunctionTraits, anyway, because of the BC field evaluator
// factory mechanism. (Additionally, in any case, IdentityCoordFunctionTraits is
// the only such class.) In the future, when we get around to removing the
// factory (and all of FactoryTraits.hpp), we can go back to using it if it's of
// use.

template<typename EvalT, typename Traits/*, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<EvalT>*/ >
class DirichletCoordFunction;

template <typename EvalT, typename Traits/*, typename cfunc_traits = PHAL::IdentityCoordFunctionTraits<EvalT>*/ >
class DirichletCoordFunction_Base : public PHAL::DirichletBase<EvalT, Traits> {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    DirichletCoordFunction_Base(Teuchos::ParameterList& p);

    //! Type of traits class being used
    //typedef cfunc_traits cfunc_traits_type;
    typedef PHAL::IdentityCoordFunctionTraits<EvalT> cfunc_traits_type;

    cfunc_traits_type func;

};

// **************************************************************
// Residual
// **************************************************************

template<typename Traits/*, typename cfunc_traits*/ >
class DirichletCoordFunction<PHAL::AlbanyTraits::Residual, Traits/*, cfunc_traits*/>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::Residual, Traits/*, cfunc_traits*/> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************

template<typename Traits/*, typename cfunc_traits*/>
class DirichletCoordFunction<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************

template<typename Traits/*, typename cfunc_traits*/>
class DirichletCoordFunction<PHAL::AlbanyTraits::Tangent, Traits/*, cfunc_traits*/>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::Tangent, Traits/*, cfunc_traits*/> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
//  -- Currently assuming no parameter derivative
// **************************************************************

template<typename Traits/*, typename cfunc_traits*/>
class DirichletCoordFunction<PHAL::AlbanyTraits::DistParamDeriv, Traits/*, cfunc_traits*/>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits/*, cfunc_traits*/> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

#ifdef ALBANY_SG
// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************

template<typename Traits/*, typename cfunc_traits*/>
class DirichletCoordFunction<PHAL::AlbanyTraits::SGResidual, Traits/*, cfunc_traits*/>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGResidual, Traits/*, cfunc_traits*/> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************

template<typename Traits/*, typename cfunc_traits*/>
class DirichletCoordFunction<PHAL::AlbanyTraits::SGJacobian, Traits/*, cfunc_traits*/>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGJacobian, Traits/*, cfunc_traits*/> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************

template<typename Traits/*, typename cfunc_traits*/>
class DirichletCoordFunction<PHAL::AlbanyTraits::SGTangent, Traits/*, cfunc_traits*/>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGTangent, Traits/*, cfunc_traits*/> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};
#endif 

#ifdef ALBANY_ENSEMBLE 
// **************************************************************
// Multi-point Residual
// **************************************************************

template<typename Traits/*, typename cfunc_traits*/>
class DirichletCoordFunction<PHAL::AlbanyTraits::MPResidual, Traits/*, cfunc_traits*/>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPResidual, Traits/*, cfunc_traits*/> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************

template<typename Traits/*, typename cfunc_traits*/>
class DirichletCoordFunction<PHAL::AlbanyTraits::MPJacobian, Traits/*, cfunc_traits*/>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPJacobian, Traits/*, cfunc_traits*/> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************

template<typename Traits/*, typename cfunc_traits*/>
class DirichletCoordFunction<PHAL::AlbanyTraits::MPTangent, Traits/*, cfunc_traits*/>
    : public DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPTangent, Traits/*, cfunc_traits*/> {
  public:
    DirichletCoordFunction(Teuchos::ParameterList& p);
    typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
    void evaluateFields(typename Traits::EvalData d);
};
#endif

}

#endif
