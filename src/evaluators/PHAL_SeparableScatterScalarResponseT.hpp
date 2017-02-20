//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: DistParamDeriv not implemented.  SG and MP has Epetra.

#ifndef PHAL_SEPARABLE_SCATTER_SCALAR_RESPONSET_HPP
#define PHAL_SEPARABLE_SCATTER_SCALAR_RESPONSET_HPP

#include "PHAL_ScatterScalarResponse.hpp"

namespace PHAL {

/** \brief Handles scattering of separable scalar response functions into epetra
 * data structures.
 * 
 * Base implementation useable by specializations below
 */
template<typename EvalT, typename Traits> 
class SeparableScatterScalarResponseBaseT
  : public virtual PHX::EvaluatorWithBaseImpl<Traits>,
    public virtual PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  
  SeparableScatterScalarResponseBaseT(const Teuchos::ParameterList& p,
			       const Teuchos::RCP<Albany::Layouts>& dl);
  
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d) {}
  
protected:

  // Default constructor for child classes
  SeparableScatterScalarResponseBaseT() {}

  // Child classes should call setup once p is filled out
  void setup(const Teuchos::ParameterList& p,
	     const Teuchos::RCP<Albany::Layouts>& dl);

protected:

  typedef typename EvalT::ScalarT ScalarT;
  PHX::MDField<ScalarT> local_response;
};
  
/** \brief Handles scattering of separable scalar response functions into epetra
 * data structures.
 * 
 * A separable response function is one that is a sum of respones across cells.
 * In this case we can compute the Jacobian in a generic fashion.
 */
template <typename EvalT, typename Traits> 
class SeparableScatterScalarResponseT : 
    public ScatterScalarResponse<EvalT, Traits>,
    public SeparableScatterScalarResponseBaseT<EvalT,Traits> {
public:
  SeparableScatterScalarResponseT(const Teuchos::ParameterList& p,
			   const Teuchos::RCP<Albany::Layouts>& dl) :
    ScatterScalarResponse<EvalT,Traits>(p,dl) {}

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm) {
    ScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d,vm);
    SeparableScatterScalarResponseBaseT<EvalT,Traits>::postRegistrationSetup(d,vm);
  }

  void evaluateFields(typename Traits::EvalData d) {}
    
protected:
  SeparableScatterScalarResponseT() {}
  void setup(const Teuchos::ParameterList& p,
	     const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponse<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBaseT<EvalT,Traits>::setup(p,dl);
  }
};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class SeparableScatterScalarResponseT<PHAL::AlbanyTraits::Jacobian,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::Jacobian, Traits>,
    public SeparableScatterScalarResponseBaseT<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  SeparableScatterScalarResponseT(const Teuchos::ParameterList& p,
			   const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm) {
    ScatterScalarResponseBase<EvalT, Traits>::postRegistrationSetup(d,vm);
    SeparableScatterScalarResponseBaseT<EvalT,Traits>::postRegistrationSetup(d,vm);
  }
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::Jacobian EvalT;
  SeparableScatterScalarResponseT() {}
  void setup(const Teuchos::ParameterList& p,
	     const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBaseT<EvalT,Traits>::setup(p,dl);
    numNodes = dl->node_scalar->dimension(1);
  }
private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  int numNodes;
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class SeparableScatterScalarResponseT<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>,
    public SeparableScatterScalarResponseBaseT<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
public:
  SeparableScatterScalarResponseT(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) {
    ScatterScalarResponseBase<EvalT, Traits>::postRegistrationSetup(d,vm);
    SeparableScatterScalarResponseBaseT<EvalT,Traits>::postRegistrationSetup(d,vm);
  }
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::DistParamDeriv EvalT;
  SeparableScatterScalarResponseT() {}
  void setup(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBaseT<EvalT,Traits>::setup(p,dl);
    numNodes = dl->node_scalar->dimension(1);
  }
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  int numNodes;
}; 

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
#ifdef ALBANY_SG
template<typename Traits>
class SeparableScatterScalarResponseT<PHAL::AlbanyTraits::SGJacobian,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::SGJacobian, Traits>,
    public SeparableScatterScalarResponseBaseT<PHAL::AlbanyTraits::SGJacobian, Traits>{
public:
  SeparableScatterScalarResponseT(const Teuchos::ParameterList& p,
			   const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm) {
    ScatterScalarResponseBase<EvalT, Traits>::postRegistrationSetup(d,vm);
    SeparableScatterScalarResponseBaseT<EvalT,Traits>::postRegistrationSetup(d,vm);
  }
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::SGJacobian EvalT;
  SeparableScatterScalarResponseT() {}
  void setup(const Teuchos::ParameterList& p,
	     const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBaseT<EvalT,Traits>::setup(p,dl);
    numNodes = dl->node_scalar->dimension(1);
  }
private:
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  int numNodes;
};
#endif 
#ifdef ALBANY_ENSEMBLE 

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class SeparableScatterScalarResponseT<PHAL::AlbanyTraits::MPJacobian,Traits>
  : public ScatterScalarResponseBase<PHAL::AlbanyTraits::MPJacobian, Traits>,
    public SeparableScatterScalarResponseBaseT<PHAL::AlbanyTraits::MPJacobian, Traits>{
public:
  SeparableScatterScalarResponseT(const Teuchos::ParameterList& p,
			   const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm) {
    ScatterScalarResponseBase<EvalT, Traits>::postRegistrationSetup(d,vm);
    SeparableScatterScalarResponseBaseT<EvalT,Traits>::postRegistrationSetup(d,vm);
  }
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
protected:
  typedef PHAL::AlbanyTraits::MPJacobian EvalT;
  SeparableScatterScalarResponseT() {}
  void setup(const Teuchos::ParameterList& p,
	     const Teuchos::RCP<Albany::Layouts>& dl) {
    ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
    SeparableScatterScalarResponseBaseT<EvalT,Traits>::setup(p,dl);
    numNodes = dl->node_scalar->dimension(1);
  }
private:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  int numNodes;
};
#endif

// **************************************************************
}

#endif
