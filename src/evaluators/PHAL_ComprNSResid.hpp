//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_COMPRNSRESID_HPP
#define PHAL_COMPRNSRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ComprNSResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		        public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ComprNSResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> qFluct; //vector q' containing fluid fluctuations in primitive variables
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> qFluctGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> qFluctDot;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> force;
  
  double gamma_gas; //1.4 typically 
  double Rgas; //Non-dimensional gas constant Rgas = R*Tref/(cref*cref), where R = nondimensional gas constant = 287.0 typically
  double Re;   //Reynolds number
  double Pr;   //Prandtl number, 0.72 typically 
  PHX::MDField<ScalarT,Cell,QuadPoint> mu;
  PHX::MDField<ScalarT,Cell,QuadPoint> kappa;
  PHX::MDField<ScalarT,Cell,QuadPoint> lambda;
  PHX::MDField<ScalarT,Cell,QuadPoint> tau11;
  PHX::MDField<ScalarT,Cell,QuadPoint> tau12;
  PHX::MDField<ScalarT,Cell,QuadPoint> tau13;
  PHX::MDField<ScalarT,Cell,QuadPoint> tau22;
  PHX::MDField<ScalarT,Cell,QuadPoint> tau23;
  PHX::MDField<ScalarT,Cell,QuadPoint> tau33;
  
  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim> Residual;


  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;
  bool enableTransient;
};
}

#endif
