//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_SCALARADVECTIONRESID_HPP
#define AERAS_SCALARADVECTIONRESID_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {
/** \brief ScalarAdvection equation Residual for atmospheric modeling

    This evaluator computes the residual of the ScalarAdvection equation for
    atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class ScalarAdvectionResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ScalarAdvectionResid(Teuchos::ParameterList& p,
                        const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);

private:

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  PHX::MDField<ScalarT,Cell,QuadPoint>     X;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> XGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint>     XDot;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> uXGrad;
  PHX::MDField<MeshScalarT,Cell,Point,Dim> coordVec;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> Residual;

  const int numNodes   ;
  const int numQPs     ;
  const int numDims    ;
  const int numLevels  ;
  const int numRank    ;
};
}

#endif
