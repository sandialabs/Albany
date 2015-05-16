//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESFOTHICKRESID_HPP
#define FELIX_STOKESFOTHICKRESID_HPP

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
class StokesFOThicknessResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		        public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  StokesFOThicknessResid(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> force;

  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> U;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> Ugrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> H;
   PHX::MDField<ScalarT,Cell,QuadPoint,Dim> gradH0;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> UDot;
  PHX::MDField<ScalarT,Cell,QuadPoint> muFELIX;

  enum EQNTYPE {FELIX, POISSON, FELIX_XZ};
  EQNTYPE eqn_type;
  
  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim> Residual;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDimFO;
  bool enableTransient;

};
}

#endif
