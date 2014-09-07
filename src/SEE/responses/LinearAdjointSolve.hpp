//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LINEARADJOINTSOLVE_HPP
#define LINEARADJOINTSOLVE_HPP

#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Teuchos_ParameterList.hpp>
#include "Albany_ProblemUtils.hpp"

namespace SEE
{

//*********************************************************************
template<typename EvalT, typename Traits>
class LinearAdjointSolveBase : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{

public:
  
  LinearAdjointSolveBase(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);
  
  void postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void preEvaluate(typename Traits::PreEvalData d) = 0;
  void postEvaluate(typename Traits::PostEvalData d) = 0;
  void evaluateFields(typename Traits::EvalData d) = 0;

  Teuchos::RCP<const PHX::FieldTag>
    getEvaluatedFieldTag() const { return field_tag_; }

  Teuchos::RCP<const PHX::FieldTag>
    getResponseFieldTag() const { return field_tag_; }
    
protected:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  Teuchos::RCP<const Teuchos::ParameterList>
    getValidLinearAdjointSolveParameters() const;

  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;

  Teuchos::RCP< PHX::Tag<ScalarT> > field_tag_;

};

//*********************************************************************
template<typename EvalT, typename Traits>
class LinearAdjointSolve :
  public LinearAdjointSolveBase<EvalT, Traits>
{
public:
  LinearAdjointSolve(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl) :
    LinearAdjointSolveBase<EvalT, Traits>(p, dl){}
  void preEvaluate(typename Traits::PreEvalData d){}
  void postEvaluate(typename Traits::PostEvalData d){}
  void evaluateFields(typename Traits::EvalData d){}
};

// residual specialization
//*********************************************************************
template<typename Traits>
class LinearAdjointSolve<PHAL::AlbanyTraits::Residual,Traits> :
public LinearAdjointSolveBase<PHAL::AlbanyTraits::Residual, Traits>
{
public:
  LinearAdjointSolve(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};
}

#endif
