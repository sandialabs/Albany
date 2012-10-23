//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESRM_HPP
#define FELIX_STOKESRM_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesRm : public PHX::EvaluatorWithBaseImpl<Traits>,
	     public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  StokesRm(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);


private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> pGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> VGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> V;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> V_Dot;
  PHX::MDField<ScalarT,Cell,QuadPoint> T;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> force;  
  PHX::MDField<MeshScalarT,Cell,QuadPoint, Dim> coordVec;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> Rm;

  unsigned int numQPs, numDims, numNodes;
 
};
}

#endif
