//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_NSTHERMALEQRESID_HPP
#define PHAL_NSTHERMALEQRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class NSThermalEqResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  NSThermalEqResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<ScalarT,Cell,QuadPoint> Temperature;
  PHX::MDField<ScalarT,Cell,QuadPoint> Tdot;
  PHX::MDField<ScalarT,Cell,QuadPoint> ThermalCond;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> TGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> V;
  PHX::MDField<ScalarT,Cell,QuadPoint> Source;
  PHX::MDField<ScalarT,Cell,QuadPoint> TauT;
  PHX::MDField<ScalarT,Cell,QuadPoint> rho;  
  PHX::MDField<ScalarT,Cell,QuadPoint> Cp;
  PHX::MDField<ScalarT,Cell,QuadPoint> phi;
  PHX::MDField<ScalarT,Cell,QuadPoint> Fission;
  PHX::MDField<ScalarT,Cell,QuadPoint> PropConst;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> TResidual;

  bool haveSource, haveFlow, haveSUPG, haveNeut; 
  bool enableTransient;
  unsigned int numQPs, numDims, numNodes;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> flux;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> convection;

 };
}

#endif
