//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_CAHNHILLELASTICITYRESID_HPP
#define PHAL_CAHNHILLELASTICITYRESID_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class CahnHillElasResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		        public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  CahnHillElasResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> Stress;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,Dim> ExResidual;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;

};
}

#endif
