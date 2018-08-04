//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TSUNAMI_BOUSSINESQRESID_HPP
#define TSUNAMI_BOUSSINESQRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace Tsunami {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class BoussinesqResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  BoussinesqResid(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<const ScalarT,Cell,QuadPoint,VecDim> EtaUE; 
  PHX::MDField<const ScalarT,Cell,QuadPoint,VecDim> EtaUEDot; 
  PHX::MDField<const ScalarT,Cell,QuadPoint,VecDim,Dim> EtaUEGrad;
  PHX::MDField<const ScalarT,Cell,QuadPoint,VecDim,Dim> EtaUEDotGrad;
  PHX::MDField<const ScalarT,Cell,QuadPoint,VecDim> force;
  PHX::MDField<const ScalarT,Cell,QuadPoint> waterDepthQP;
  PHX::MDField<const ScalarT,Cell,QuadPoint> betaQP;
  PHX::MDField<const ScalarT,Cell,QuadPoint> zalphaQP;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint, Dim> waterDepthGrad;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim> Residual;

  unsigned int numQPs, numDims, numNodes, vecDim;

  double muSqr, epsilon; 

  double C1, C2, C3; 
 
  Teuchos::RCP<Teuchos::FancyOStream> out;  
};
}

#endif
