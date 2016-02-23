//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ASSUMEDSTRAIN_HPP
#define ASSUMEDSTRAIN_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LCM {
/** \brief Deformation Gradient

    This evaluator computes the deformation gradient

*/

template<typename EvalT, typename Traits>
class AssumedStrain : public PHX::EvaluatorWithBaseImpl<Traits>,
		public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  AssumedStrain(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> GradU;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weights;


  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> assumedStrain;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;

  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  bool avgJ;
  bool volavgJ;
  bool weighted_Volume_Averaged_J;

};

}
#endif
