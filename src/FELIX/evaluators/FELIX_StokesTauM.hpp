//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESTAUM_HPP
#define FELIX_STOKESTAUM_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesTauM : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  StokesTauM(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input: 
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> V;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> VGrad; //IK - added 7/19/2012
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> Gc;
  PHX::MDField<ScalarT,Cell,QuadPoint> mu;
  PHX::MDField<ScalarT,Cell,QuadPoint> muFELIX;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> jacobian_det; //jacobian determinant - for getting mesh size h 
  double delta; 
  ScalarT meshSize; //mesh size h 

  // Output:
  PHX::MDField<ScalarT,Cell,Node> TauM;

  unsigned int numQPs, numDims;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> normGc;
  
};
}

#endif
