//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_CALCINSTANTANEOUSCOORDS_HPP
#define PHAL_CALCINSTANTANEOUSCOORDS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
class CalcInstantaneousCoords : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  CalcInstantaneousCoords(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ScalarT ScalarT;
  int  numNodes, numDims, numQPs;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<ScalarT, Cell, Node, Dim> dispVec;
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;

  // Output:
  PHX::MDField<ScalarT, Cell, Node, Dim> instCoords;

};
}

#endif
