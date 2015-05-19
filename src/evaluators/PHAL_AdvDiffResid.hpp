//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_ADVDIFFRESID_HPP
#define PHAL_ADVDIFFRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class AdvDiffResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		        public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  AdvDiffResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> U; 
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> UGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> UDot;
  
  double mu;   //viscosity coefficient
  double a;    //advection coefficient
  double b;    //advection coefficient
  bool useAugForm; //use augmented form?
  int formType; //augmented form type
  
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
