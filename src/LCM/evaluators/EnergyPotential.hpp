//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ENERGYPOTENTIAL_HPP
#define ENERGYPOTENTIAL_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief Nonlinear Elasticity Energy Potential

    This evaluator computes a energy density for nonlinear elastic material

*/

template<typename EvalT, typename Traits>
class EnergyPotential : public PHX::EvaluatorWithBaseImpl<Traits>,
			public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  EnergyPotential(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  PHX::MDField<ScalarT,Cell,QuadPoint> elasticModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> poissonsRatio;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> energy;

  std::size_t numQPs;
  std::size_t numDims;
  std::size_t numCells;
};
}

#endif
