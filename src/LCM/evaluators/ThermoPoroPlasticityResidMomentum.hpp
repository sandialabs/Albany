//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef THERMOPOROPLASTICITYRESIDMOMENTUM_HPP
#define THERMOPOROPLASTICITYRESIDMOMENTUM_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief

    This evaluator calculate residual of the linear
    momentum balance equation
    for the thermoporomechanics problem.

*/

template<typename EvalT, typename Traits>
class ThermoPoroPlasticityResidMomentum : public PHX::EvaluatorWithBaseImpl<Traits>,
		        public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ThermoPoroPlasticityResidMomentum(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> TotalStress;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> uDotDot;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;

  PHX::MDField<ScalarT,Cell,QuadPoint> Bulk;
  PHX::MDField<ScalarT,Cell,QuadPoint> alphaSkeleton;

  PHX::MDField<ScalarT,Cell,QuadPoint> Temp;
  PHX::MDField<ScalarT,Cell,QuadPoint> TempRef;


  // Output:
  PHX::MDField<ScalarT,Cell,Node,Dim> ExResidual;

  int numNodes;
  int numQPs;
  int numDims;
  bool enableTransient;
  ScalarT dTemp;

  // Work space FCs
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> F_inv;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> F_invT;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> JF_invT;
//  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> P;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> thermoEPS;

};
}

#endif
