//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKESTAUM_HPP
#define LANDICE_STOKESTAUM_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce {
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
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> V;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim,Dim> VGrad; //IK - added 7/19/2012
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim,Dim> Gc;
  PHX::MDField<const ScalarT,Cell,QuadPoint> mu;
  PHX::MDField<const ScalarT,Cell,QuadPoint> muLandIce;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint> jacobian_det; //jacobian determinant - for getting mesh size h 
  double delta; 
  ScalarT meshSize; //mesh size h 

  // Output:
  PHX::MDField<ScalarT,Cell,Node> TauM;

  unsigned int numQPs, numDims, numCells;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> normGc;
  
};
}

#endif
